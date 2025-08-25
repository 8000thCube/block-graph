fn derror<D:Display,E:Derror>(msg:D)->E{E::custom(msg)}
fn deserialize_batch_norm<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<BatchNorm<B,1>,D::Error>{
	let record:BatchNormRecord<B>=BatchNormRecord::deserialize(deserializer)?;

	let (beta,epsilon,gamma,mean,momentum,variance)=(record.beta,record.epsilon,record.gamma,record.mean,record.momentum,record.variance);
	let (beta,gamma)=if let (Ok(b),Ok(g))=(beta.try_into(),gamma.try_into()){(Param::from_tensor(b),Param::from_tensor(g))}else{return Err(derror("batch norm beta and gamma parameters must be rank 1 floats"))};
	let (mean,variance)=if let (Ok(m),Ok(v))=(mean.try_into(),variance.try_into()){(RunningState::new(m),RunningState::new(v))}else{return Err(derror("batch norm mean and variance states must be rank 1 floats"))};

	Ok(BatchNorm{beta,epsilon,gamma,momentum,running_mean:mean,running_var:variance})
}
fn deserialize_conv2d<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<Conv2d<B>,D::Error>{
	let record=Conv2dRecord::deserialize(deserializer)?;

	let (dilation,groups,kernelsize,stride)=(record.dilation,record.groups,record.kernelsize,record.stride);
	let bias=if let Some(b)=record.bias{
		if let Ok(b)=b.try_into(){Some(Param::from_tensor(b))}else{return Err(derror("linear bias parameter must be a rank 1 float"))}
	}else{
		None
	};
	let padding=record.padding.clone();
	let weight=Param::from_tensor(if let Ok(w)=record.weight.try_into(){w}else{return Err(derror("linear weight parameter must be a rank 2 float"))});

	Ok(Conv2d{bias,dilation,groups,kernel_size:kernelsize,padding,stride,weight})
}
fn deserialize_cross_entropy<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<CrossEntropyLoss<B>,D::Error>{
	let record=CrossEntropyRecord::deserialize(deserializer)?;

	let (logits,pad,smoothing)=(record.logits,record.pad,record.smoothing);
	let weights=if let Some(s)=record.weights{
		if let Ok(s)=s.try_into(){Some(s)}else{return Err(derror("cross entropy weights parameter must be a rank 1 float"))}
	}else{
		None
	};

	Ok(CrossEntropyLoss{logits,pad_tokens:pad,smoothing,weights})
}
fn deserialize_dropout<'a,D:Deserializer<'a>>(deserializer:D)->Result<Dropout,D::Error>{
	Ok(Dropout{prob:f64::deserialize(deserializer)?})
}
fn deserialize_embedding<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<Embedding<B>,D::Error>{
	let weight=deserialize_param(deserializer)?;
	Ok(Embedding{weight})
}
fn deserialize_ignored<'a,D:Deserializer<'a>,T:Deserialize<'a>>(deserializer:D)->Result<Ignored<T>,D::Error>{
	let data:T=T::deserialize(deserializer)?;
	Ok(Ignored(data))
}
fn deserialize_layer_norm<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<LayerNorm<B>,D::Error>{
	let mut layer=LayerNormConfig::new(1).init(&Default::default());
	let record=LayerNormRecord::deserialize(deserializer)?;

	if let Ok(b)=record.beta.try_into(){layer.beta=Param::from_tensor(b)}else{return Err(derror("beta parameter must be a rank 1 float"))}
	if let Ok(g)=record.gamma.try_into(){layer.gamma=Param::from_tensor(g)}else{return Err(derror("gamma parameter must be a rank 1 float"))}

	Ok(layer)
}
fn deserialize_linear<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<Linear<B>,D::Error>{
	let record=LinearRecord::deserialize(deserializer)?;

	let bias=if let Some(b)=record.bias{
		if let Ok(b)=b.try_into(){Some(Param::from_tensor(b))}else{return Err(derror("linear bias parameter must be a rank 1 float"))}
	}else{
		None
	};
	let weight=Param::from_tensor(if let Ok(w)=record.weight.try_into(){w}else{return Err(derror("linear weight parameter must be a rank 2 float"))});

	Ok(Linear{bias,weight})
}
fn deserialize_nothing<'a,D:Deserializer<'a>,T:Default>(_deserializer:D)->Result<T,D::Error>{Ok(T::default())}
fn deserialize_param<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Param<Tensor<B,N>>,D::Error>{
	let data:Value<B>=Value::deserialize(deserializer)?;
	if let Ok(t)=data.try_into(){Ok(Param::from_tensor(t))}else{Err(derror(format!("expected parameter to be a rank {N} float")))}
}
fn deserialize_rotary<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<RotaryEncoding<B>,D::Error>{Ok(RotaryEncodingConfig::deserialize(deserializer)?.init(&Default::default()))}
fn serialize_batch_norm<B:Backend,S:Serializer>(layer:&BatchNorm<B,1>,serializer:S)->Result<S::Ok,S::Error>{
	let (beta,gamma)=(Value::from(layer.beta.val()),Value::from(layer.gamma.val()));
	let (epsilon,momentum)=(layer.epsilon,layer.momentum);
	let (mean,variance)=(Value::from(layer.running_mean.value()),Value::from(layer.running_var.value()));

	BatchNormRecord{beta,epsilon,gamma,mean,momentum,variance}.serialize(serializer)
}
fn serialize_conv2d<B:Backend,S:Serializer>(layer:&Conv2d<B>,serializer:S)->Result<S::Ok,S::Error>{
	let (dilation,groups,kernelsize,stride)=(layer.dilation,layer.groups,layer.kernel_size,layer.stride);
	let bias=layer.bias.as_ref().map(|b|b.val().into());
	let padding=layer.padding.clone();
	let weight=layer.weight.val().into();

	Conv2dRecord{bias,dilation,groups,kernelsize,padding,stride,weight}.serialize(serializer)
}
fn serialize_cross_entropy<'a,B:Backend,S:Serializer>(layer:&CrossEntropyLoss<B>,serializer:S)->Result<S::Ok,S::Error>{
	let (logits,pad,smoothing)=(layer.logits.clone(),layer.pad_tokens.clone(),layer.smoothing.clone());
	let weights=layer.weights.clone().map(Into::into);

	CrossEntropyRecord{logits,pad,smoothing,weights}.serialize(serializer)
}
fn serialize_dropout<S:Serializer>(data:&Dropout,serializer:S)->Result<S::Ok,S::Error>{data.prob.serialize(serializer)}
fn serialize_embedding<B:Backend,S:Serializer>(layer:&Embedding<B>,serializer:S)->Result<S::Ok,S::Error>{serialize_param(&layer.weight,serializer)}
fn serror<D:Display,E:Serror>(msg:D)->E{E::custom(msg)}
fn serialize_ignored<S:Serializer,T:Serialize>(data:&Ignored<T>,serializer:S)->Result<S::Ok,S::Error>{
	let data:&T=data;
	data.serialize(serializer)
}
fn serialize_layer_norm<B:Backend,S:Serializer>(layer:&LayerNorm<B>,serializer:S)->Result<S::Ok,S::Error>{
	LayerNormRecord{beta:layer.beta.val().into(),gamma:layer.gamma.val().into()}.serialize(serializer)
}
fn serialize_linear<B:Backend,S:Serializer>(layer:&Linear<B>,serializer:S)->Result<S::Ok,S::Error>{
	let bias=layer.bias.as_ref().map(|b|b.val().into());
	let weight=layer.weight.val().into();

	LinearRecord{bias,weight}.serialize(serializer)
}
fn serialize_nothing<S:Serializer,T:Default>(_data:&T,serializer:S)->Result<S::Ok,S::Error>{().serialize(serializer)}
fn serialize_param<B:Backend,S:Serializer,const N:usize>(data:&Param<Tensor<B,N>>,serializer:S)->Result<S::Ok,S::Error>{
	if N>8{return Err(serror("tensor rank greater than 8 is not currently supported"))}
	let data:Value<B>=data.val().into();
	data.serialize(serializer)
}
fn serialize_rotary<B:Backend,S:Serializer>(data:&RotaryEncoding<B>,serializer:S)->Result<S::Ok,S::Error>{
	let [distance,head,_2]=data.freq_complex.dims();
	let theta:f32=data.theta.clone().into_scalar().elem();

	RotaryEncodingConfig::new(distance,head).with_theta(theta).serialize(serializer)
}
impl AttentionConfig{
	pub fn init<B:Backend>(&self,_device:&B::Device)->Attention<B>{
		let (dropout,heads,mask)=(self.dropout,self.heads,self.mask);
		let mask=Ignored(mask);
		let phantom=PhantomData;

		Attention{dropout,heads,mask,phantom}
	}
}
impl BiasConfig{
	pub fn init<B:Backend>(&self,device:&B::Device)->Bias<B>{
		let dim=self.dim;
		let shape=[dim];

		Bias{bias:self.initializer.init_with(shape,None,Some(dim),device)}
	}
}
impl Config{
	/// creates an attention config
	pub fn attention(heads:usize,mask:AttentionMask)->Self{Self::Attention(AttentionConfig::new(heads,mask))}
	/// creates a bias config
	pub fn bias(dim:usize)->Self{Self::Bias(BiasConfig::new(dim))}
	/// creates a embedding config
	pub fn embedding(input:usize,output:usize)->Self{Self::Embedding(EmbeddingConfig::new(input,output))}
	/// initializes the layer
	pub fn init<B:Backend>(&self,device:&B::Device)->Layer<B>{
		match self{Config::Attention(c)=>Layer::Attention(c.init(device)),Config::BatchNorm(c)=>Layer::BatchNorm(c.init(device)),Config::Bias(c)=>Layer::Bias(c.init(device)),Config::CacheKV=>Layer::CacheKV(CacheKV::default()),Config::Cat(c)=>Layer::Cat(Ignored(*c)),Config::Conv2d(c)=>Layer::Conv2d(c.init(device)),Config::Dropout(c)=>Layer::Dropout(c.init()),Config::Embedding(c)=>Layer::Embedding(c.init(device)),Config::LayerNorm(c)=>Layer::LayerNorm(c.init(device)),Config::Linear(c)=>Layer::Linear(c.init(device)),Config::KQV(c)=>Layer::KQV(c.init(device)),Config::CrossEntropy(c)=>Layer::CrossEntropy(c.init(device)),Config::Mse=>Layer::Mse(MseLoss),Config::Relu=>Layer::Relu(Relu::new()),Config::Rotary(c)=>Layer::Rotary(c.init(device)),Config::Stack(d)=>Layer::Stack(*d),Config::Sum(c)=>Layer::Sum(Ignored(*c)),Config::Tanh=>Layer::Tanh(Tanh::new())}
	}
	/// creates a layer norm config
	pub fn layer_norm(dim:usize)->Self{Self::LayerNorm(LayerNormConfig::new(dim))}
	/// creates a linear config
	pub fn linear(bias:bool,input:usize,output:usize)->Self{Self::Linear(LinearConfig::new(input,output).with_bias(bias))}
	/// creates a relu config
	pub fn relu()->Self{Self::Relu}
	/// creates a rotary config
	pub fn rotary(distance:usize,head:usize)->Self{Self::Rotary(RotaryEncodingConfig::new(distance,head))}
	/// creates a tanh config
	pub fn tanh()->Self{Self::Tanh}
	/// scales the initializer
	pub fn w_scale(mut self,r:f32)->Self{
		match &mut self{Config::Attention(_c)=>(),Config::BatchNorm(_c)=>(),Config::Bias(c)=>w_scale_mut(&mut c.initializer,r),Config::CacheKV=>(),Config::Cat(_c)=>(),Config::Conv2d(c)=>w_scale_mut(&mut c.initializer,r),Config::CrossEntropy(_c)=>(),Config::Dropout(_c)=>(),Config::Embedding(c)=>w_scale_mut(&mut c.initializer,r),Config::KQV(c)=>w_scale_mut(&mut c.initializer,r),Config::LayerNorm(_c)=>(),Config::Linear(c)=>w_scale_mut(&mut c.initializer,r),Config::Mse=>(),Config::Relu=>(),Config::Rotary(_c)=>(),Config::Stack(_d)=>(),Config::Sum(_c)=>(),Config::Tanh=>()}
		self
	}
}
impl Decompose for Config{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl From<AttentionConfig> for Config{
	fn from(value:AttentionConfig)->Self{Self::Attention(value)}
}
impl From<BiasConfig> for Config{
	fn from(value:BiasConfig)->Self{Self::Bias(value)}
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
impl KQVConfig{
	pub fn init<B:Backend>(&self,device:&B::Device)->KQV<B>{
		let (embed,initializer,kdim,vdim)=(self.embed.clone(),self.initializer.clone(),self.kdim.clone(),self.vdim.clone());
		let (key,value)=(LinearConfig::new(embed,kdim).with_initializer(initializer.clone()).init(device),LinearConfig::new(embed,vdim).with_initializer(initializer.clone()).init(device));
		let query=LinearConfig::new(embed,kdim).with_initializer(initializer).init(device);

		KQV{key,query,value}
	}
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
			Layer::BatchNorm(f)=>AI::forward(f,input),
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
			Layer::BatchNorm(f)=>AI::forward_mut(f,input),
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
#[derive(Clone,Copy,Debug,Deserialize,Serialize)]
pub enum AttentionMask{Causal,None,Window(usize)}
#[derive(Config)]
/// enumerates config for some burn layers
pub enum Config{Attention(AttentionConfig),BatchNorm(BatchNormConfig),Bias(BiasConfig),CacheKV,Cat(CatLayer),Conv2d(Conv2dConfig),CrossEntropy(CrossEntropyLossConfig),Dropout(DropoutConfig),Embedding(EmbeddingConfig),KQV(KQVConfig),LayerNorm(LayerNormConfig),Linear(LinearConfig),Mse,Relu,Rotary(RotaryEncodingConfig),Stack(usize),Sum(SumLayer),Tanh}
#[derive(Debug,Deserialize,Module,Serialize)]//TODO more layers
#[serde(bound="")]
/// enumerates some burn layers
pub enum Layer<B:Backend>{
	Attention(Attention<B>),
	Bias(Bias<B>),
	#[serde(deserialize_with="deserialize_batch_norm")]
	#[serde(serialize_with="serialize_batch_norm")]
	BatchNorm(BatchNorm<B,1>),
	CacheKV(CacheKV<B>),
	#[serde(deserialize_with="deserialize_ignored")]
	#[serde(serialize_with="serialize_ignored")]
	Cat(Ignored<CatLayer>),
	#[serde(deserialize_with="deserialize_conv2d")]
	#[serde(serialize_with="serialize_conv2d")]
	Conv2d(Conv2d<B>),
	#[serde(deserialize_with="deserialize_cross_entropy")]
	#[serde(serialize_with="serialize_cross_entropy")]
	CrossEntropy(CrossEntropyLoss<B>),
	#[serde(deserialize_with="deserialize_dropout")]
	#[serde(serialize_with="serialize_dropout")]
	Dropout(Dropout),
	#[serde(deserialize_with="deserialize_embedding")]
	#[serde(serialize_with="serialize_embedding")]
	Embedding(Embedding<B>),
	KQV(KQV<B>),
	#[serde(deserialize_with="deserialize_layer_norm")]
	#[serde(serialize_with="serialize_layer_norm")]
	LayerNorm(LayerNorm<B>),
	#[serde(deserialize_with="deserialize_linear")]
	#[serde(serialize_with="serialize_linear")]
	Linear(Linear<B>),
	#[serde(deserialize_with="deserialize_nothing")]
	#[serde(serialize_with="serialize_nothing")]
	Mse(MseLoss),
	#[serde(deserialize_with="deserialize_nothing")]
	#[serde(serialize_with="serialize_nothing")]
	Relu(Relu),
	#[serde(deserialize_with="deserialize_rotary")]
	#[serde(serialize_with="serialize_rotary")]
	Rotary(RotaryEncoding<B>),
	Stack(usize),
	#[serde(deserialize_with="deserialize_ignored")]
	#[serde(serialize_with="serialize_ignored")]
	Sum(Ignored<SumLayer>),
	#[serde(deserialize_with="deserialize_nothing")]
	#[serde(serialize_with="serialize_nothing")]
	Tanh(Tanh)
}
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
#[derive(Config,Debug)]
/// layer for computing attention from [key,query,value] inputs
pub struct AttentionConfig{
	#[config(default="0.2")]
	dropout:f32,
	heads:usize,
	mask:AttentionMask
}
#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// layer for computing attention from [key,query,value] inputs
pub struct Attention<B:Backend>{
	dropout:f32,
	heads:usize,
	#[serde(deserialize_with="deserialize_ignored")]
	#[serde(serialize_with="serialize_ignored")]
	mask:Ignored<AttentionMask>,
	phantom:PhantomData<B>
}
#[derive(Config,Debug)]
/// layer for adding bias somewhere
pub struct BiasConfig{
	dim:usize,
	#[config(default="Initializer::Normal{mean:0.0,std:1.0}")]
	initializer:Initializer
}
#[derive(Config,Debug)]
/// layer for linear splitting into [key,query,value] for attention purposes
pub struct KQVConfig{
	embed:usize,
	#[config(default="Initializer::XavierNormal{gain:1.0}")]
	initializer:Initializer,
	kdim:usize,
	vdim:usize
}
#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// layer for adding bias anywhere
pub struct Bias<B:Backend>{
	#[serde(deserialize_with="deserialize_param")]
	#[serde(serialize_with="serialize_param")]
	bias:Param<Tensor<B,1>>
}
#[derive(Debug,Default,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// layer for caching kv values from kqv when run mutably. cats along d1 and outputs the concatenated keys and values. clears cache on forward_mut when new data is incompatible for concatenation
pub struct CacheKV<B:Backend>{keys:Value<B>,values:Value<B>}
#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// layer for linear splitting into [key,query,value] for attention purposes
pub struct KQV<B:Backend>{
	#[serde(deserialize_with="deserialize_linear")]
	#[serde(serialize_with="serialize_linear")]
	key:Linear<B>,
	#[serde(deserialize_with="deserialize_linear")]
	#[serde(serialize_with="serialize_linear")]
	query:Linear<B>,
	#[serde(deserialize_with="deserialize_linear")]
	#[serde(serialize_with="serialize_linear")]
	value:Linear<B>
}
#[derive(Deserialize,Serialize)]
#[serde(bound="")]
struct Conv2dRecord<B:Backend>{
	bias:Option<Value<B>>,
	dilation:[usize;2],
	groups:usize,
	kernelsize:[usize;2],
	#[serde(deserialize_with="deserialize_ignored")]
	#[serde(serialize_with="serialize_ignored")]
	padding:Ignored<PaddingConfig2d>,
	stride:[usize;2],
	weight:Value<B>
}
#[derive(Deserialize,Serialize)]
#[serde(bound="")]
struct BatchNormRecord<B:Backend>{beta:Value<B>,epsilon:f64,gamma:Value<B>,mean:Value<B>,momentum:f64,variance:Value<B>}
#[derive(Deserialize,Serialize)]
#[serde(bound="")]
struct CrossEntropyRecord<B:Backend>{logits:bool,pad:Option<Vec<usize>>,weights:Option<Value<B>>,smoothing:Option<f32>}
#[derive(Deserialize,Serialize)]
#[serde(bound="")]
struct LayerNormRecord<B:Backend>{beta:Value<B>,gamma:Value<B>}
#[derive(Deserialize,Serialize)]
#[serde(bound="")]
struct LinearRecord<B:Backend>{bias:Option<Value<B>>,weight:Value<B>}
use burn::{
	module::{Ignored,Param,RunningState},
	nn::{
		BatchNorm,BatchNormConfig,Dropout,DropoutConfig,Embedding,EmbeddingConfig,Initializer,LayerNorm,LayerNormConfig,Linear,LinearConfig,PaddingConfig2d,Relu,RotaryEncoding,RotaryEncodingConfig,Tanh,conv::{Conv2d,Conv2dConfig},loss::{CrossEntropyLoss,CrossEntropyLossConfig,MseLoss}
	},
	prelude::*,
	tensor::activation
};
use crate::{
	ai::{AI,Decompose,IntoSequence,Op},
	builtin::{
		Sequential,math::SumLayer,structural::CatLayer
	},burn::Value
};
use serde::{Deserialize,Deserializer,Serialize,Serializer,de::Error as Derror,ser::Error as Serror};
use std::{fmt::Display,marker::PhantomData,mem};
