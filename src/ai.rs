decompose_primitive!((),bool,char,f32,f64,i128,i16,i32,i64,i8,isize,u128,u16,u32,u64,u8,usize);
decompose_tuple!((A,B),(A,B,C),(A,B,C,D),(A,B,C,D,E),(A,B,C,D,E,F),(A,B,C,D,E,F,G),(A,B,C,D,E,F,G,H));
impl Decompose for Range<usize>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.0..decomposition.1}
	fn decompose(self)->Self::Decomposition{(self.start,self.end)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.start,self.end)}
	type Decomposition=(usize,usize);
}
impl Op for (){
	type Output=();
}
impl<A:?Sized+AI<X,Y>,X,Y> AI<X,Y> for &A{
	fn forward(&self,input:X)->Y{(**self).forward(input)}
}
impl<A:?Sized+AI<X,Y>,X,Y> AI<X,Y> for &mut A{
	fn forward(&self,input:X)->Y{(**self).forward(input)}
	fn forward_mut(&mut self,input:X)->Y{(**self).forward_mut(input)}
}
impl<A:?Sized+Op<Output=Y>,Y> Op for &A{
	type Output=Y;
}
impl<A:?Sized+Op<Output=Y>,Y> Op for &mut A{
	type Output=Y;
}
impl<A:AI<X,X>+Op<Output=X>,X> Op for Option<A>{
	type Output=X;
}
impl<A:AI<X,X>,X> AI<X,X> for Option<A>{
	fn forward(&self,x:X)->X{
		if let Some(a)=self{a.forward(x)}else{x}
	}
	fn forward_mut(&mut self,x:X)->X{
		if let Some(a)=self{a.forward_mut(x)}else{x}
	}
}
impl<A:AI<X,Y>,X,Y> AI<X,Y> for Inner<A>{
	fn forward(&self,input:X)->Y{self.0.forward(input)}
	fn forward_mut(&mut self,input:X)->Y{self.0.forward_mut(input)}
}
impl<A:Decompose> Decompose for Inner<A>{
	fn compose(decomposition:Self::Decomposition)->Self{Self(A::compose(decomposition))}
	fn decompose(self)->Self::Decomposition{self.0.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.0.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Option<A>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.map(A::compose)}
	fn decompose(self)->Self::Decomposition{self.map(A::decompose)}
	fn decompose_cloned(&self)->Self::Decomposition{self.as_ref().map(A::decompose_cloned)}
	type Decomposition=Option<A::Decomposition>;
}
impl<A:Decompose> Decompose for Vec<A>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.into_iter().map(A::compose).collect()}
	fn decompose(self)->Self::Decomposition{self.into_iter().map(A::decompose).collect()}
	fn decompose_cloned(&self)->Self::Decomposition{self.iter().map(A::decompose_cloned).collect()}
	type Decomposition=Vec<A::Decomposition>;
}
impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for Inner<A>{
	fn into_sequence(self)->Sequential<Vec<M>>{self.0.into_sequence()}
}
impl<A:Op> Op for Inner<A>{
	type Output=A::Output;
}
impl<A> From<A> for Inner<A>{
	fn from(inner:A)->Self{Self(inner)}
}
impl<A> Inner<A>{
	/// references the inner value
	pub fn inner(&self)->&A{&self.0}
	/// references the inner value
	pub fn inner_mut(&mut self)->&mut A{&mut self.0}
	/// converts into the inner value
	pub fn into_inner(self)->A{self.0}
}
impl<A> Op for [A]{
	type Output=();
}
impl<A> Op for Vec<A>{
	type Output=();
}
impl<A> UnwrapInner for Inner<A>{
	fn unwrap_inner(self)->Self::Inner{self.0}
	type Inner=A;
}
impl<K:Decompose+Eq+Hash,V:Decompose,S:Default+BuildHasher> Decompose for HashMap<K,V,S> where K::Decomposition:Ord{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.into_iter().map(Decompose::compose).collect()}
	fn decompose(self)->Self::Decomposition{
		let mut v:Vec<_>=self.into_iter().map(Decompose::decompose).collect();
		v.sort_unstable_by(|(k,_v),(k2,_v2)|k.cmp(k2));
		v
	}
	fn decompose_cloned(&self)->Self::Decomposition{
		let mut v:Vec<_>=self.iter().map(|(k,v)|(k.decompose_cloned(),v.decompose_cloned())).collect();
		v.sort_unstable_by(|(k,_v),(k2,_v2)|k.cmp(k2));
		v
	}
	type Decomposition=Vec<(K::Decomposition,V::Decomposition)>;
}
impl<X:Into<Y>,Y> AI<X,Y> for (){
	fn forward(&self,input:X)->Y{input.into()}
}
/// implements decompose for primitive types
macro_rules! decompose_primitive{
	($($type:ty),*)=>($(impl Decompose for $type{
		fn compose(decomposition:Self::Decomposition)->Self{decomposition}
		fn decompose(self)->Self::Decomposition{self}
		fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
		type Decomposition=Self;
	})*);
}
macro_rules! decompose_tuple{
	($(($($type:ident),+)),*)=>($(impl<$($type:Decompose),+> Decompose for ($($type),+){
		#[allow(non_snake_case)]
		fn compose(($($type),+):Self::Decomposition)->Self{($(Decompose::compose($type)),+)}
		#[allow(non_snake_case)]
		fn decompose(self)->Self::Decomposition{
			let ($($type),+)=self;
			($($type.decompose()),+)
		}
		#[allow(non_snake_case)]
		fn decompose_cloned(&self)->Self::Decomposition{
			let ($($type),+)=self;
			($($type.decompose_cloned()),+)
		}
		type Decomposition=($($type::Decomposition),+);
	})*);
}
/// implements op for tuples
macro_rules! op_tuple{
	($(($($type:ident),+)),*)=>($(impl<$($type:Op),+> Op for ($($type),+){
		type Output=();
	})*);
}
op_tuple!((A,B),(A,B,C),(A,B,C,D),(A,B,C,D,E),(A,B,C,D,E,F),(A,B,C,D,E,F,G),(A,B,C,D,E,F,G,H));
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// wraps an inner value so it can be unwrapped with unwrap inner
pub struct Inner<A>(pub A);
/// general ai trait
pub trait AI<X,Y>{
	/// applies to the input
	fn forward(&self,input:X)->Y;
	/// applies to the input, possibly updating internal caches
	fn forward_mut(&mut self,input:X)->Y{self.forward(input)}
}
/// trait to decompose AI modules into components that implement other libraries' traits
pub trait Decompose{// TODO derive macros, make decompose cloned and decompose take and into sequence cloned and into sequence take
	/// recreates from the decomposition
	fn compose(decomposition:Self::Decomposition)->Self where Self:Sized;
	/// owned decomposition
	fn decompose(self)->Self::Decomposition where Self:Sized;
	/// decomposition that copies data
	fn decompose_cloned(&self)->Self::Decomposition;
	/// the decomposed type
	type Decomposition;
}
/// conversion from a composite module into a sequential list of modules
pub trait IntoSequence<M:AI<M::Output,M::Output>+Op>{
	/// converts into a sequential module list
	fn into_sequence(self)->Sequential<Vec<M>>;
}
/// composition trait
pub trait Op{
	/// wraps with a softmax operation
	fn abnormal_softmax(self,dim:i32)->AbnormalSoftmax<Self> where Self:Sized,AbnormalSoftmax<Self>:Op{AbnormalSoftmax::new(dim,self)}
	/// wraps with an absolute value operation
	fn abs(self)->Abs<Self> where Self:Sized,Abs<Self>:Op{Abs::new(self)}
	/// wraps with a accq operation
	fn acc_q(self,dim:i32,gamma:f32)->AccQ<Self> where AccQ<Self>:Op,Self:Sized{AccQ::new(dim,gamma,self)}
	/// wraps with a cat operation
	fn cat(self,dim:i32)->Cat<Self> where Cat<Self>:Op,Self:Sized{Cat::new(dim,self)}
	/// sequences with another ai operation
	fn chain<B>(self,b:B)->Sequential<(Self,B)> where Self:Sized,Sequential<(Self,B)>:Op{Sequential::new((self,b))}
	/// wraps with a cross entropy operation
	fn cross_entropy(self,dim:i32)->CrossEntropy<Self> where CrossEntropy<Self>:Op,Self:Sized{CrossEntropy::new(dim,self)}
	/// wraps with a duplicate operation
	fn duplicate(self)->Duplicate<Self> where Duplicate<Self>:Op,Self:Sized{Duplicate::new(self)}
	/// set type but with the same input and output
	fn fix_type<Z>(self)->SetType<Self,Z,Z> where Self:AI<Z,Z>+Sized{self.set_type()}
	/// applies to the input
	fn forward_fixed<Z>(&self,input:Z)->Z where Self:AI<Z,Z>+Sized{self.forward(input)}
	/// applies to the input
	fn forward_fixed_mut<Z>(&mut self,input:Z)->Z where Self:AI<Z,Z>+Sized{self.forward(input)}
	/// applies to the input
	fn forward_typed<W,Z>(&self,input:W)->Z where Self:AI<W,Z>+Sized{self.forward(input)}
	/// applies to the input, possibly updating internal caches
	fn forward_typed_mut<W,Z>(&mut self,input:W)->Z where Self:AI<W,Z>+Sized{self.forward(input)}
	/// creates an autoregressive inference
	fn infer_autoregressive<X,Y>(self,input:X)->Autoregression<Self,Y> where Self:AI<X,Y>+AI<Y,Y>+Sized,Y:Clone{Autoregression::new(self,input)}
	/// wraps with a softmax operation
	fn log_softmax(self,dim:i32)->LogSoftmax<Self> where Self:Sized,LogSoftmax<Self>:Op{LogSoftmax::new(dim,self)}
	/// applies the operation to every output
	fn map<B>(self,b:B)->Map<Sequential<(Self,B)>> where Map<Sequential<(Self,B)>>:Op,Self:Sized,Sequential<(Self,B)>:Op{self.chain(b).to_each()}
	/// wraps with a mean operation
	fn mean(self)->Mean<Self> where Mean<Self>:Op,Self:Sized{Mean::new(self)}
	/// creates an optional operation
	fn optional(self)->Option<Self> where Self:Sized{Some(self)}
	/// produces a zip module
	fn separately(self)->Zip<Self> where Self:Sized,Zip<Self>:Op{Zip::new(self)}
	/// produces a sequential module
	fn sequential(self)->Sequential<Self> where Self:Sized,Sequential<Self>:Op{Sequential::new(self)}
	/// sets the input output types
	fn set_type<W,Z>(self)->SetType<Self,W,Z> where Self:AI<W,Z>+Sized{SetType::new(self)}
	/// wraps with a choose operation
	fn soft_choose(self,dim:i32)->Choose<Self> where Self:Sized,Choose<Self>:Op{Choose::new(dim,self)}
	/// wraps with a softmax operation
	fn softmax(self,dim:i32)->Argmax<Self> where Self:Sized,Argmax<Self>:Op{Argmax::new(dim,self)}
	/// wraps with a mse operation
	fn squared_error(self)->SquaredError<Self> where SquaredError<Self>:Op,Self:Sized{SquaredError::new(self)}
	/// wraps with a map operation
	fn to_each(self)->Map<Self> where Map<Self>:Op,Self:Sized{Map::new(self)}
	/// wraps with a sum operation
	fn sum(self)->Sum<Self> where Sum<Self>:Op,Self:Sized{Sum::new(self)}
	/// wraps the inner value so it can be unwrapped with unwrap inner
	fn wrap_inner(self)->Inner<Self> where Self:Sized{Inner(self)}
	/// zips with another ai operation
	fn zip<B>(self,b:B)->Zip<(Self,B)> where Self:Sized,Zip<(Self,B)>:Op{Zip::new((self,b))}
	/// suggested output type to help with composition coherence. Ideally, Self should implement AI<X,Self::Output> for some X
	type Output;
}
/// trait for unwrapping nested wrapped values
pub trait UnwrapInner{
	/// unwraps the inner value
	fn unwrap_inner(self)->Self::Inner;
	/// the inner type
	type Inner;
}
use {op_tuple,decompose_primitive,decompose_tuple};
use crate::builtin::{AbnormalSoftmax,Abs,AccQ,Argmax,Autoregression,Cat,Choose,CrossEntropy,Duplicate,LogSoftmax,Map,Mean,Sequential,SetType,SquaredError,Sum,Zip};
use std::{
	collections::HashMap,cmp::Ord,hash::{BuildHasher,Hash},ops::Range
};
