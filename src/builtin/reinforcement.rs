impl AI<Vec<f32>,Vec<f32>> for AccQLayer{
	fn forward(&self,mut input:Vec<f32>)->Vec<f32>{
		let (dim,gamma)=(self.dim,self.gamma);
		assert!(dim==0||dim==-1,"Dimension index was {dim} but a vec only has one tensor dimension");

		input.iter_mut().rev().fold(0.0,|future,present|{
			*present+=future*gamma;
			*present
		});
		input
	}
}
impl AccQLayer{
	/// creates from the inner value
	pub fn new(gamma:f32)->Self{Self::default().with_gamma(gamma)}
	/// gets the dimension
	pub fn get_dim(&self)->i32{self.dim}
	/// gets the gamma
	pub fn get_gamma(&self)->f32{self.gamma}
	/// sets the dim
	pub fn set_dim(&mut self,dim:i32){self.dim=dim}
	/// sets the gamma
	pub fn set_gamma(&mut self,gamma:f32){self.gamma=gamma}
	/// sets the dim
	pub fn with_dim(mut self,dim:i32)->Self{
		self.dim=dim;
		self
	}
	/// withs the gamma
	pub fn with_gamma(mut self,gamma:f32)->Self{
		self.gamma=gamma;
		self
	}
}
impl Decompose for AccQLayer{
	fn compose((dim,gamma):Self::Decomposition)->Self{
		Self{dim,gamma}
	}
	fn decompose(self)->Self::Decomposition{(self.dim,self.gamma)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.dim,self.gamma)}
	type Decomposition=(i32,f32);
}
impl Default for AccQLayer{
	fn default()->Self{
		Self{dim:-1,gamma:0.9}
	}
}
impl Op for AccQLayer{
	type Output=Vec<f32>;
}
impl<A:AI<X,Y>,X,Y> AI<X,Y> for AccQ<A> where AccQLayer:AI<Y,Y>{
	fn forward(&self,input:X)->Y{self.layer.forward(self.inner.forward(input))}
	fn forward_mut(&mut self,input:X)->Y{self.layer.forward(self.inner.forward_mut(input))}
}
impl<A:Decompose> Decompose for AccQ<A>{
	fn compose((inner,(dim,gamma)):Self::Decomposition)->Self{
		Self{inner:A::compose(inner),layer:AccQLayer::compose((dim,gamma))}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.layer.decompose())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.layer.decompose_cloned())}
	type Decomposition=(A::Decomposition,(i32,f32));
}
impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for AccQ<A> where AccQLayer:Into<M>{
	fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
}
impl<A:Op<Output=Y>,Y> Op for AccQ<A> where AccQLayer:AI<Y,Y>{
	type Output=Y;
}
impl<A:UnwrapInner> UnwrapInner for AccQ<A>{
	fn unwrap_inner(self)->Self::Inner{self.into_inner().unwrap_inner()}
	type Inner=A::Inner;
}
impl<A> AccQ<A>{
	/// gets the dimension
	pub fn get_dim(&self)->i32{self.layer.dim}
	/// gets the gamma
	pub fn get_gamma(&self)->f32{self.layer.gamma}
	/// references the inner value
	pub fn inner(&self)->&A{&self.inner}
	/// references the inner value
	pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
	/// converts into the inner value
	pub fn into_inner(self)->A{self.inner}
	/// creates from the inner value
	pub fn new(gamma:f32,inner:A)->Self{
		let layer=AccQLayer::new(gamma);
		Self{inner,layer}
	}
	/// sets the dim
	pub fn set_dim(&mut self,dim:i32){self.layer.dim=dim}
	/// sets the gamma
	pub fn set_gamma(&mut self,gamma:f32){self.layer.gamma=gamma}
	/// sets the dim
	pub fn with_dim(mut self,dim:i32)->Self{
		self.layer.dim=dim;
		self
	}
	/// withs the gamma
	pub fn with_gamma(mut self,gamma:f32)->Self{
		self.layer.gamma=gamma;
		self
	}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->AccQ<B> where AccQ<B>:Op{AccQ::new(self.get_gamma(),inner).with_dim(self.get_dim())}
}
impl<M:AI<M::Output,M::Output>+Op> IntoSequence<M> for AccQLayer where AccQLayer:Into<M>{
	fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
}
#[derive(Clone,Copy,Debug,Deserialize,Default,PartialEq,Serialize)]
/// accumulates cumulative
pub struct AccQ<A>{inner:A,layer:AccQLayer}
#[derive(Clone,Copy,Debug,Deserialize,PartialEq,Serialize)]
/// accumulates cumulative
pub struct AccQLayer{dim:i32,gamma:f32}
use crate::{
	AI,Decompose,IntoSequence,Op,UnwrapInner
};
use serde::{Deserialize,Serialize};
use super::Sequential;
