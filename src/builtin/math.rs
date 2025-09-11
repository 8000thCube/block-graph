bicop_like!(AddLayer,Add);
bicop_like!(MulLayer,Mul);
bicop_like!(SquaredErrorLayer,SquaredError);
impl AI<(Vec<f32>,Vec<f32>),Vec<f32>> for SquaredErrorLayer{// TODO some kind of namable map iter might make some of these compose better
	fn forward(&self,(output,target):(Vec<f32>,Vec<f32>))->Vec<f32>{
		let (ol,tl)=(output.len(),target.len());
		assert!(ol==tl,"output len {ol} should match target len {tl}");

		output.into_iter().zip(target).map(|(o,t)|o-t).map(|x|x*x).collect()
	}
}
impl AI<(Vec<f32>,Vec<f32>),f32> for SquaredErrorLayer{
	fn forward(&self,(output,target):(Vec<f32>,Vec<f32>))->f32{
		let (ol,tl)=(output.len(),target.len());
		assert!(ol==tl,"output len {ol} should match target len {tl}");

		output.into_iter().zip(target).map(|(o,t)|o-t).map(|x|x*x).sum::<f32>()/ol as f32
	}
}
impl AI<Vec<f32>,f32> for MeanLayer{
	fn forward(&self,input:Vec<f32>)->f32{
		let sum:f32=input.iter().sum();

		sum/input.len() as f32
	}
}
impl AI<Vec<f32>,f32> for SumLayer{
	fn forward(&self,input:Vec<f32>)->f32{input.into_iter().sum()}//TODO check dim
}
impl AI<f32,f32> for MeanLayer{
	fn forward(&self,input:f32)->f32{input}
}
impl AI<f32,f32> for SumLayer{
	fn forward(&self,input:f32)->f32{input}
}
impl<L:OpsAdd<R>,R,Y> AI<(L,R),Y> for AddLayer where L::Output:Into<Y>{
	fn forward(&self,(left,right):(L,R))->Y{(left+right).into()}
}
impl<L:OpsMul<R>,R,Y> AI<(L,R),Y> for MulLayer where L::Output:Into<Y>{
	fn forward(&self,(left,right):(L,R))->Y{(left*right).into()}
}
impl<X:OpsAbs,Y> AI<X,Y> for AbsLayer where X::Output:Into<Y>{
	fn forward(&self,input:X)->Y{input.abs().into()}
}
impl<X:OpsNeg,Y> AI<X,Y> for NegLayer where X::Output:Into<Y>{
	fn forward(&self,input:X)->Y{input.neg().into()}
}
/// declares layer and wrapper structs and implements accessor functions, decompose and op for binary componentwise operations. ai will still have to be externally implemented for the layer stuct
macro_rules! bicop_like{// TODO separate parts of this like in one of the other likes and make squared error specifically have vec output
	($layer:ident,$wrap:ident)=>{
		impl $layer{
			/// creates a new layer
			pub fn new()->Self{Self::default()}
		}
		impl<A:AI<X,L>+Op<Output=L>,L,R,X,Y> AI<(X,R),Y> for $wrap<A> where $layer:AI<(L,R),Y>{
			fn forward(&self,(input,right):(X,R))->Y{self.layer.forward((self.inner.forward(input),right))}// TODO swap operation
			fn forward_mut(&mut self,(input,right):(X,R))->Y{self.layer.forward_mut((self.inner.forward_mut(input),right))}
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose(inner:Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:Default::default()}
			}
			fn decompose(self)->Self::Decomposition{self.inner.decompose()}
			fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
			type Decomposition=A::Decomposition;
		}
		impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $wrap<A> where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
		}
		impl<A:UnwrapInner> UnwrapInner for $wrap<A>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<(Y,Y),Y>{
			type Output=Y;
		}
		impl<A> $wrap<A>{/// references the inner value
			pub fn inner(&self)->&A{&self.inner}
			/// references the inner value
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// converts into the inner value
			pub fn into_inner(self)->A{self.inner}
			/// creates a new layer
			pub fn new(inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new()}
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
		}
		impl<M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $layer where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
		}
		impl Decompose for $layer{
			fn compose(_decomposition:Self::Decomposition)->Self{Self::new()}
			fn decompose(self){}
			fn decompose_cloned(&self){}
			type Decomposition=();
		}
		impl Op for $layer{
			type Output=Vec<f32>;
		}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{seal:PhantomData<()>}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	}
}
/// declares layer and wrapper structs and implements accessor functions, decompose and op for reduction operations that have a reduction mode as configuration fields. ai will still have to be externally implemented for the layer stuct
macro_rules! sum_like{
	($layer:ident,$wrap:ident)=>{
		impl $layer{
			/// gets the reduction mode
			pub fn get_reduction_mode(&self)->ReductionMode{self.reductionmode}
			/// creates a new layer
			pub fn new()->Self{Self::default()}
			/// sets the reduction mode
			pub fn set_reduction_mode(&mut self,mode:ReductionMode){self.reductionmode=mode}
			/// sets the reduction mode
			pub fn with_reduction_mode(mut self,mode:ReductionMode)->Self{
				self.reductionmode=mode;
				self
			}
		}
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose((inner,layer):Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:$layer::compose(layer)}
			}
			fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.layer.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.layer.decompose_cloned())}
			type Decomposition=(A::Decomposition,<$layer as Decompose>::Decomposition);
		}
		impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $wrap<A> where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
		}
		impl<A:UnwrapInner> UnwrapInner for $wrap<A>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<Y,f32>{
			type Output=f32;
		}
		impl<A> $wrap<A>{
			/// gets the reduction mode
			pub fn get_reduction_mode(&self)->ReductionMode{self.layer.reductionmode}
			/// references the inner value
			pub fn inner(&self)->&A{&self.inner}
			/// references the inner value
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// converts into the inner value
			pub fn into_inner(self)->A{self.inner}
			/// creates a new layer
			pub fn new(inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new()}
			}
			/// sets the reduction mode
			pub fn set_reduction_mode(&mut self,mode:ReductionMode){self.layer.reductionmode=mode}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
				self.layer.dim=dim;
				self
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
			/// sets the reduction mode
			pub fn with_reduction_mode(mut self,mode:ReductionMode)->Self{
				self.layer.reductionmode=mode;
				self
			}
		}
		impl Decompose for $layer{
			fn compose((dim,reductionmode):Self::Decomposition)->Self{
				Self{dim,reductionmode:ReductionMode::compose(reductionmode)}
			}
			fn decompose(self)->Self::Decomposition{(self.dim,self.reductionmode.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.dim,self.reductionmode.decompose_cloned())}
			type Decomposition=(i32,u64);
		}
		impl Op for $layer{
			type Output=f32;
		}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{dim:i32,reductionmode:ReductionMode}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	}
}
/// declares layer and wrapper structs and implements accessor functions, decompose and op for unary componentwise operations. ai will still have to be externally implemented for the layer stuct
macro_rules! uncop_like{
	($layer:ident,$wrap:ident)=>{
		impl $layer{
			/// creates a new layer
			pub fn new()->Self{Self::default()}
		}
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose(inner:Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:Default::default()}
			}
			fn decompose(self)->Self::Decomposition{self.inner.decompose()}
			fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
			type Decomposition=A::Decomposition;
		}
		impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $wrap<A> where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
		}
		impl<A:UnwrapInner> UnwrapInner for $wrap<A>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<Y,Y>{
			type Output=Y;
		}
		impl<A> $wrap<A>{/// references the inner value
			pub fn inner(&self)->&A{&self.inner}
			/// references the inner value
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// converts into the inner value
			pub fn into_inner(self)->A{self.inner}
			/// creates a new layer
			pub fn new(inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new()}
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
		}
		impl<M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $layer where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
		}
		impl Decompose for $layer{
			fn compose(_decomposition:Self::Decomposition)->Self{Self::new()}
			fn decompose(self){}
			fn decompose_cloned(&self){}
			type Decomposition=();
		}
		impl Op for $layer{
			type Output=Vec<f32>;
		}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{seal:PhantomData<()>}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	}
}
sum_like!(MeanLayer,Mean);
sum_like!(SumLayer,Sum);
uncop_like!(AbsLayer,Abs);
uncop_like!(NegLayer,Neg);
use {bicop_like,sum_like};
use crate::{
	AI,Decompose,IntoSequence,Op,UnwrapInner,ops::Abs as OpsAbs
};
use serde::{Deserialize,Serialize};
use std::{
	marker::PhantomData,ops::{Add as OpsAdd,Mul as OpsMul,Neg as OpsNeg}
};
use super::{ReductionMode,Sequential};
