impl Config{
	/// initializes the layer
	pub fn init<B:Backend>(&self,device:&B::Device)->Layer<B>{
		match self{Config::Dropout(c)=>Layer::Dropout(c.init()),Config::Embedding(c)=>Layer::Embedding(c.init(device)),Config::LayerNorm(c)=>Layer::LayerNorm(c.init(device)),Config::Linear(c)=>Layer::Linear(c.init(device)),Config::CrossEntropy(c)=>Layer::CrossEntropy(c.init(device)),Config::Mse=>Layer::Mse(MseLoss),Config::Relu=>Layer::Relu(Relu::new()),Config::Stack(d)=>Layer::Stack(*d),Config::Tanh=>Layer::Tanh(Tanh::new())}
	}
	/// scales the initializer
	pub fn w_scale(mut self,r:f32)->Self{
		match &mut self{Config::CrossEntropy(_c)=>(),Config::Dropout(_c)=>(),Config::Embedding(c)=>w_scale_mut(&mut c.initializer,r),Config::LayerNorm(_c)=>(),Config::Linear(c)=>w_scale_mut(&mut c.initializer,r),Config::Mse=>(),Config::Relu=>(),Config::Stack(_d)=>(),Config::Tanh=>()}
		self
	}
}
impl Decompose for Config{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
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
impl From<Tanh> for Config{
	fn from(_value:Tanh)->Self{Config::Tanh}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Layer<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match self{
			//Layer::CrossEntropy(f)=>AI::forward(f,input),// TODO implement forward on these for one multi input with two values
			Layer::Dropout(f)=>AI::forward(f,input),
			Layer::Embedding(f)=>AI::forward(f,input),
			Layer::LayerNorm(f)=>AI::forward(f,input),
			Layer::Linear(f)=>AI::forward(f,input),
			//Layer::MseLoss(f)=>AI::forward(f,input),
			Layer::Relu(f)=>AI::forward(f,input),
			Layer::Stack(dim)=>input.stack(*dim as i32),
			Layer::Tanh(f)=>AI::forward(f,input),
			_=>todo!()
		}
	}
}
impl<B:Backend> Decompose for Layer<B>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend> From<CrossEntropyLoss<B>> for Layer<B>{
	fn from(value:CrossEntropyLoss<B>)->Self{Layer::CrossEntropy(value)}
}
impl<B:Backend> Layer<B>{// TODO direct bias layer might be good since I have a use case for it and sometimes you just unexpectedly need bias
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
	/// creates a tanh layer
	pub fn tanh()->Self{Self::Tanh(Tanh)}
}
impl<B:Backend> Op for Layer<B>{
	type Output=Value<B>;
}
#[derive(Config)]
/// enumerates config for some burn layers
pub enum Config{CrossEntropy(CrossEntropyLossConfig),Dropout(DropoutConfig),Embedding(EmbeddingConfig),LayerNorm(LayerNormConfig),Linear(LinearConfig),Mse,Relu,Stack(usize),Tanh}
#[derive(Debug,Module)]//TODO more layers
/// enumerates some burn layers
pub enum Layer<B:Backend>{CrossEntropy(CrossEntropyLoss<B>),Dropout(Dropout),Embedding(Embedding<B>),LayerNorm(LayerNorm<B>),Linear(Linear<B>),Mse(MseLoss),Relu(Relu),Stack(usize),Tanh(Tanh)}
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
use burn::{
	nn::{
		Dropout,DropoutConfig,Embedding,EmbeddingConfig,Initializer,LayerNorm,LayerNormConfig,Linear,LinearConfig,Relu,Tanh,loss::{CrossEntropyLoss,CrossEntropyLossConfig,MseLoss}
	},
	prelude::*
};
use crate::{
	ai::{AI,Decompose,Op},burn::Value
};
