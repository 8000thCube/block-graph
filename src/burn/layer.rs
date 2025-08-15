impl Config{
	/// creates a embedding config
	pub fn embedding(input:usize,output:usize,wscale:f32)->Self{
		let mut l=EmbeddingConfig::new(input,output);
		if wscale!=1.0{l.initializer=w_scale(l.initializer,wscale)}
		Self::Embedding(l)
	}
	/// initializes the layer
	pub fn init<B:Backend>(&self,device:&B::Device)->Layer<B>{
		match self{Config::Cat(c)=>Layer::Cat(Ignored(*c)),Config::Dropout(c)=>Layer::Dropout(c.init()),Config::Embedding(c)=>Layer::Embedding(c.init(device)),Config::LayerNorm(c)=>Layer::LayerNorm(c.init(device)),Config::Linear(c)=>Layer::Linear(c.init(device)),Config::CrossEntropy(c)=>Layer::CrossEntropy(c.init(device)),Config::Mse=>Layer::Mse(MseLoss),Config::Relu=>Layer::Relu(Relu::new()),Config::Rotary(c)=>Layer::Rotary(c.init(device)),Config::Stack(d)=>Layer::Stack(*d),Config::Sum(c)=>Layer::Sum(Ignored(*c)),Config::Tanh=>Layer::Tanh(Tanh::new())}
	}
	/// creates a layer norm config
	pub fn layer_norm(dim:usize)->Self{Self::LayerNorm(LayerNormConfig::new(dim))}
	/// creates a linear config
	pub fn linear(bias:bool,input:usize,output:usize,wscale:f32)->Self{
		let mut l=LinearConfig::new(input,output).with_bias(bias);
		if wscale!=1.0{l.initializer=w_scale(l.initializer,wscale)}
		Self::Linear(l)
	}
	/// creates a relu config
	pub fn relu()->Self{Self::Relu}
	/// creates a rotary config
	pub fn rotary(distance:usize,head:usize)->Self{Self::Rotary(RotaryEncodingConfig::new(distance,head))}
	/// creates a tanh config
	pub fn tanh()->Self{Self::Tanh}
	/// scales the initializer
	pub fn w_scale(mut self,r:f32)->Self{
		match &mut self{Config::Cat(_c)=>(),Config::CrossEntropy(_c)=>(),Config::Dropout(_c)=>(),Config::Embedding(c)=>w_scale_mut(&mut c.initializer,r),Config::LayerNorm(_c)=>(),Config::Linear(c)=>w_scale_mut(&mut c.initializer,r),Config::Mse=>(),Config::Relu=>(),Config::Rotary(_c)=>(),Config::Stack(_d)=>(),Config::Sum(_c)=>(),Config::Tanh=>()}
		self
	}
}
impl Decompose for Config{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl From<CatLayer> for Config{
	fn from(value:CatLayer)->Self{Config::Cat(value)}
}
impl From<CrossEntropyLossConfig> for Config{
	fn from(value:CrossEntropyLossConfig)->Self{Config::CrossEntropy(value)}
}
impl From<DropoutConfig> for Config{
	fn from(value:DropoutConfig)->Self{Config::Dropout(value)}
}
impl From<EmbeddingConfig> for Config{
	fn from(value:EmbeddingConfig)->Self{Config::Embedding(value)}
}
impl From<LayerNormConfig> for Config{
	fn from(value:LayerNormConfig)->Self{Config::LayerNorm(value)}
}
impl From<LinearConfig> for Config{
	fn from(value:LinearConfig)->Self{Config::Linear(value)}
}
impl From<MseLoss> for Config{
	fn from(_value:MseLoss)->Self{Config::Mse}
}
impl From<Relu> for Config{
	fn from(_value:Relu)->Self{Config::Relu}
}
impl From<RotaryEncodingConfig> for Config{
	fn from(value:RotaryEncodingConfig)->Self{Config::Rotary(value)}
}
impl From<SumLayer> for Config{
	fn from(value:SumLayer)->Self{Config::Sum(value)}
}
impl From<Tanh> for Config{
	fn from(_value:Tanh)->Self{Config::Tanh}
}
impl<B:Backend,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for Layer<B> where Layer<B>:Into<M>{
	fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
}
impl<B:Backend> AI<(Value<B>,Value<B>),(Value<B>,Value<B>)> for CacheKV<B>{
	fn forward(&self,(k,v):(Value<B>,Value<B>))->(Value<B>,Value<B>){
		let (keys,values)=(self.keys.clone(),self.values.clone());
		(if keys.is_empty(){k}else{Value::from(vec![keys,k]).cat(1)},if values.is_empty(){v}else{Value::from(vec![values,v]).cat(1)})
	}
	fn forward_mut(&mut self,(k,v):(Value<B>,Value<B>))->(Value<B>,Value<B>){
		let (keys,values)=(mem::take(&mut self.keys),mem::take(&mut self.values));

		let (keys,values)=(if keys.is_empty(){k}else{Value::from(vec![keys,k]).cat(1)},if values.is_empty(){v}else{Value::from(vec![values,v]).cat(1)});
		(self.keys,self.values)=if keys.is_incompatible()||values.is_incompatible(){Default::default()}else{(keys.clone(),values.clone())};

		(keys,values)
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>,Value<B>),Value<B>> for Attention<B>{
	fn forward(&self,(k,q,v):(Value<B>,Value<B>,Value<B>))->Value<B>{// TODO support for other numbers of dimensions
		fn apply_mask<B:Backend,const D:usize>(a:Tensor<B,D>,mask:AttentionMask,value:f32)->Tensor<B,D>{
			match mask{AttentionMask::Causal=>mask_causal(a,value as f64),AttentionMask::None=>a,AttentionMask::Window(n)=>mask_window(a,n,value as f64)}
		}
		fn f_3d<B:Backend>(dropout:f32,heads:usize,mask:AttentionMask,k:Tensor<B,3>,q:Tensor<B,3>,v:Tensor<B,3>)->Result<Tensor<B,3>,String>{
			let (kdims,qdims,vdims)=(k.dims(),q.dims(),v.dims());

			if kdims!=qdims{return Err("mismatched dims".into())}
			if kdims!=vdims{return Err("mismatched dims".into())}
			let [batch,sequence,embed]=kdims;
			let dropout=Dropout{prob:dropout as f64};
			let head=if embed%heads==0{embed/heads}else{return Err("embed must be a multiple of heads".into())};

			let (k,q,v)=(k.reshape([batch,sequence,heads,head]).swap_dims(1,2),q.reshape([batch,sequence,heads,head]).swap_dims(1,2),v.reshape([batch,sequence,heads,head]).swap_dims(1,2));
			let a=activation::softmax(apply_mask(q.matmul(k.transpose())/(head as f32).sqrt(),mask,-9999.0),3);
			let a=dropout.forward(a);
			let s=a.matmul(v).swap_dims(1,2).reshape([0,0,-1]);

			Ok(s)
		}
		fn mask_causal<B:Backend,const D:usize>(a:Tensor<B,D>,value:f64)->Tensor<B,D>{
			if D<2{return mask_causal::<B,2>(a.unsqueeze(),value).squeeze(0)}									// shouldn't actually happen but if the dimension is less than 2 we can just treat it like it has a second dimension of size 1

			let (device,dims)=(a.device(),a.dims());
			let (key,query)=(dims[D-1],dims[D-2]);
			let extrakeys=key.saturating_sub(query);															// due to caching, there might be more keys than queries

			let causal:Tensor<B,2,Bool>=Tensor::tril_mask([query,key],extrakeys as i64,&device);
			let a=a.mask_fill(causal.unsqueeze(),value);
			a
		}
		/// fills the attention tensor with the value where the query position is less than the key position minus length, or greater than the key position. Assumes attention dimensions are [.., query, key]
		fn mask_window<B:Backend,const D:usize>(a:Tensor<B,D>,length:usize,value:f64)->Tensor<B,D>{
			if D<2{return mask_window::<B,2>(a.unsqueeze(),length,value).squeeze(0)}							// shouldn't actually happen but if the dimension is less than 2 we can just treat it like it has a second dimension of size 1

			let (device,dims)=(a.device(),a.dims());
			let (key,query)=(dims[D-1],dims[D-2]);
			let extrakeys=key.saturating_sub(query);															// due to caching, there might be more keys than queries

			let causal:Tensor<B,2,Bool>=Tensor::tril_mask([query,key],extrakeys as i64,&device);
			let window:Tensor<B,2,Bool>=Tensor::triu_mask([query,key],extrakeys as i64-length as i64,&device);
			let a=a.mask_fill(causal.unsqueeze(),value).mask_fill(window.unsqueeze(),value);
			a
		}
		let (dropout,heads,mask)=(self.dropout,self.heads,self.mask.0);

		match match (k.float(),q.float(),v.float()){
			(Value::F3(k),Value::F3(q),Value::F3(v))=>f_3d(dropout,heads,mask,k,q,v).map(Into::into),
			(Value::Multi(k),Value::Multi(q),Value::Multi(v))=>if k.len()==q.len()&&q.len()==v.len(){Ok(k.into_iter().zip(q).zip(v).map(|((k,q),v)|self.forward((k,q,v))).collect())}else{Err("incompatible lengths".into())}
			_=>Err("attention is currently only supported for 3d float inputs [batch, seq, embed]".into())
		}{
			Err(e)=>e.into(),
			Ok(x)=>x
		}
	}
}
impl<B:Backend> AI<Value<B>,(Value<B>,Value<B>,Value<B>)> for KQV<B>{
	fn forward(&self,input:Value<B>)->(Value<B>,Value<B>,Value<B>){
		let (k,q)=(input.clone(),input.clone());
		let v=input;

		(AI::forward(&self.key,k),AI::forward(&self.query,q),AI::forward(&self.value,v))
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Attention<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{
			Value::Incompatible(e)=>e.into(),
			Value::Multi(v) if v.len()>=3=>if v.len()==3{
				let [k,q,v]=v.try_into().unwrap();
				self.forward((k,q,v))
			}else{
				v.into_iter().map(|x|self.forward(x)).collect()
			},
			_=>"attention inputs must be in triples".into()
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Bias<B>{
	fn forward(&self,input:Value<B>)->Value<B>{input+Value::from(self.bias.val())}
}
impl<B:Backend> AI<Value<B>,Value<B>> for CacheKV<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{
			Value::Incompatible(e)=>e.into(),
			Value::Multi(v) if v.len()>=2=>match v.len(){
				2=>{
					let [k,v]=v.try_into().unwrap();

					let (k,v)=self.forward((k,v));
					vec![k,v].into()
				},
				3=>{
					let [k,q,v]=v.try_into().unwrap();

					let (k,v)=self.forward((k,v));
					vec![k,q,v].into()
				},
				_=>{
					v.into_iter().map(|x|self.forward(x)).collect()
				}
			},
			_=>"cache kv inputs must be in pairs or triples".into()
		}
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		match input{
			Value::Incompatible(e)=>e.into(),
			Value::Multi(v) if v.len()>=2=>match v.len(){
				2=>{
					let [k,v]=v.try_into().unwrap();

					let (k,v)=self.forward_mut((k,v));
					vec![k,v].into()
				},
				3=>{
					let [k,q,v]=v.try_into().unwrap();

					let (k,v)=self.forward_mut((k,v));
					vec![k,q,v].into()
				},
				_=>{
					v.into_iter().map(|x|self.forward_mut(x)).collect()
				}
			},
			_=>"cache kv inputs must be in pairs or triples".into()
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for KQV<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		let (k,q,v)=self.forward(input);
		vec![k,q,v].into()
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Layer<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match self{
			Layer::Attention(f)=>f.forward(input),
			Layer::Bias(f)=>f.forward(input),
			Layer::CacheKV(f)=>f.forward(input),
			Layer::Cat(f)=>f.forward(input),
			Layer::Conv2d(f)=>AI::forward(f,input),
			Layer::CrossEntropy(f)=>AI::forward(f,input),
			Layer::Dropout(f)=>AI::forward(f,input),
			Layer::Embedding(f)=>AI::forward(f,input),
			Layer::KQV(f)=>f.forward(input),
			Layer::LayerNorm(f)=>AI::forward(f,input),
			Layer::Linear(f)=>AI::forward(f,input),
			Layer::Mse(f)=>AI::forward(f,input),
			Layer::Relu(f)=>AI::forward(f,input),
			Layer::Rotary(f)=>AI::forward(f,input),
			Layer::Stack(dim)=>input.stack(*dim as i32),
			Layer::Sum(f)=>f.forward(input),
			Layer::Tanh(f)=>AI::forward(f,input),
		}
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		match self{
			Layer::Attention(f)=>f.forward_mut(input),
			Layer::Bias(f)=>f.forward_mut(input),
			Layer::CacheKV(f)=>f.forward_mut(input),
			Layer::Cat(f)=>f.0.forward_mut(input),
			Layer::Conv2d(f)=>f.forward_mut(input),
			Layer::CrossEntropy(f)=>AI::forward_mut(f,input),
			Layer::Dropout(f)=>AI::forward_mut(f,input),
			Layer::Embedding(f)=>AI::forward_mut(f,input),
			Layer::KQV(f)=>f.forward_mut(input),
			Layer::LayerNorm(f)=>AI::forward_mut(f,input),
			Layer::Linear(f)=>AI::forward_mut(f,input),
			Layer::Mse(f)=>AI::forward_mut(f,input),
			Layer::Relu(f)=>AI::forward_mut(f,input),
			Layer::Rotary(f)=>AI::forward_mut(f,input),
			Layer::Stack(dim)=>input.stack(*dim as i32),
			Layer::Sum(f)=>f.0.forward_mut(input),
			Layer::Tanh(f)=>AI::forward_mut(f,input),
		}
	}
}
impl<B:Backend> Decompose for Layer<B>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend> From<CatLayer> for Layer<B>{
	fn from(value:CatLayer)->Self{Layer::Cat(Ignored(value))}
}
impl<B:Backend> From<CrossEntropyLoss<B>> for Layer<B>{
	fn from(value:CrossEntropyLoss<B>)->Self{Layer::CrossEntropy(value)}
}
impl<B:Backend> From<Dropout> for Layer<B>{
	fn from(value:Dropout)->Self{Layer::Dropout(value)}
}
impl<B:Backend> From<Embedding<B>> for Layer<B>{
	fn from(value:Embedding<B>)->Self{Layer::Embedding(value)}
}
impl<B:Backend> From<LayerNorm<B>> for Layer<B>{
	fn from(value:LayerNorm<B>)->Self{Layer::LayerNorm(value)}
}
impl<B:Backend> From<Linear<B>> for Layer<B>{
	fn from(value:Linear<B>)->Self{Layer::Linear(value)}
}
impl<B:Backend> From<MseLoss> for Layer<B>{
	fn from(value:MseLoss)->Self{Layer::Mse(value)}
}
impl<B:Backend> From<Relu> for Layer<B>{
	fn from(value:Relu)->Self{Layer::Relu(value)}
}
impl<B:Backend> From<RotaryEncoding<B>> for Layer<B>{
	fn from(value:RotaryEncoding<B>)->Self{Layer::Rotary(value)}
}
impl<B:Backend> From<SumLayer> for Layer<B>{
	fn from(value:SumLayer)->Self{Layer::Sum(Ignored(value))}
}
impl<B:Backend> From<Tanh> for Layer<B>{
	fn from(value:Tanh)->Self{Layer::Tanh(value)}
}
impl<B:Backend> Layer<B>{
	/// creates a embedding layer
	pub fn embedding(input:usize,output:usize,wscale:f32)->Self{
		let mut l=EmbeddingConfig::new(input,output);
		if wscale!=1.0{l.initializer=w_scale(l.initializer,wscale)}
		let l=l.init(&Default::default());
		Self::Embedding(l)
	}
	/// creates a layer norm layer
	pub fn layer_norm(dim:usize)->Self{Self::LayerNorm(LayerNormConfig::new(dim).init(&Default::default()))}
	/// creates a linear layer
	pub fn linear(bias:bool,input:usize,output:usize,wscale:f32)->Self{
		let mut l=LinearConfig::new(input,output).with_bias(bias);
		if wscale!=1.0{l.initializer=w_scale(l.initializer,wscale)}
		let l=l.init(&Default::default());
		Self::Linear(l)
	}
	/// creates a relu layer
	pub fn relu()->Self{Self::Relu(Relu)}
	/// creates a rotary layer
	pub fn rotary(distance:usize,head:usize)->Self{Self::Rotary(RotaryEncodingConfig::new(distance,head).init(&Default::default()))}
	/// creates a tanh layer
	pub fn tanh()->Self{Self::Tanh(Tanh)}
}
impl<B:Backend> Op for Layer<B>{
	type Output=Value<B>;
}
#[derive(Clone,Copy,Debug)]
pub enum AttentionMask{Causal,None,Window(usize)}
#[derive(Config)]
/// enumerates config for some burn layers
pub enum Config{Cat(CatLayer),CrossEntropy(CrossEntropyLossConfig),Dropout(DropoutConfig),Embedding(EmbeddingConfig),LayerNorm(LayerNormConfig),Linear(LinearConfig),Mse,Relu,Rotary(RotaryEncodingConfig),Stack(usize),Sum(SumLayer),Tanh}
#[derive(Debug,Module)]//TODO more layers//TODO kqv, rotary, attention, bias
/// enumerates some burn layers
pub enum Layer<B:Backend>{Attention(Attention<B>),Bias(Bias<B>),CacheKV(CacheKV<B>),Cat(Ignored<CatLayer>),Conv2d(Conv2d<B>),CrossEntropy(CrossEntropyLoss<B>),Dropout(Dropout),Embedding(Embedding<B>),KQV(KQV<B>),LayerNorm(LayerNorm<B>),Linear(Linear<B>),Mse(MseLoss),Relu(Relu),Rotary(RotaryEncoding<B>),Stack(usize),Sum(Ignored<SumLayer>),Tanh(Tanh)}
/// scales the initializer
pub fn w_scale(initializer:Initializer,r:f32)->Initializer{
	let r=r as f64;// apparently
	match initializer{
		Initializer::Constant{value}=>Initializer::Constant{value:value*r},
		Initializer::KaimingNormal{gain,fan_out_only}=>Initializer::KaimingNormal{gain:gain*r,fan_out_only},
		Initializer::KaimingUniform{gain,fan_out_only}=>Initializer::KaimingUniform{gain:gain*r,fan_out_only},
		Initializer::Normal{mean,std}=>Initializer::Normal{mean:mean*r,std:std*r},
		Initializer::Ones=>Initializer::Constant{value:r},
		Initializer::Orthogonal{gain}=>Initializer::Orthogonal{gain:gain*r},
		Initializer::Uniform{min,max}=>Initializer::Uniform{min:min*r,max:max*r},
		Initializer::XavierNormal{gain}=>Initializer::XavierNormal{gain:gain*r},
		Initializer::XavierUniform{gain}=>Initializer::XavierUniform{gain:gain*r},
		Initializer::Zeros=>Initializer::Zeros
	}
}
/// scales the initializer
pub fn w_scale_mut(initializer:&mut Initializer,r:f32){*initializer=w_scale(initializer.clone(),r)}
#[derive(Debug,Module)]
/// layer for computing attention from [key,query,value] inputs
pub struct Attention<B:Backend>{dropout:f32,heads:usize,mask:Ignored<AttentionMask>,phantom:PhantomData<B>}
#[derive(Debug,Module)]
/// layer for adding bias anywhere
pub struct Bias<B:Backend>{bias:Param<Tensor<B,1>>}
#[derive(Debug,Default,Module)]
/// layer for caching kv values from kqv when run mutably. cats along d1 and outputs the concatenated keys and values. clears cache on forward_mut when new data is incompatible for concatenation
pub struct CacheKV<B:Backend>{keys:Value<B>,values:Value<B>}
#[derive(Debug,Module)]
/// layer for linear splitting into [key,query,value] for attention purposes
pub struct KQV<B:Backend>{key:Linear<B>,query:Linear<B>,value:Linear<B>}
use burn::{
	module::{Ignored,Param},
	nn::{
		Dropout,DropoutConfig,Embedding,EmbeddingConfig,Initializer,LayerNorm,LayerNormConfig,Linear,LinearConfig,Relu,RotaryEncoding,RotaryEncodingConfig,Tanh,conv::Conv2d,loss::{CrossEntropyLoss,CrossEntropyLossConfig,MseLoss}
	},
	prelude::*,
	tensor::activation
};
use crate::{
	ai::{AI,Decompose,IntoSequence,Op},builtin::{CatLayer,Sequential,SumLayer},burn::Value
};
use std::{marker::PhantomData,mem};
