cat_like!(CatLayer,Cat);
cat_like!(SqueezeLayer,Squeeze);
cat_like!(StackLayer,Stack);
cat_like!(UnsqueezeLayer,Unsqueeze);
flat_like!(FlattenLayer,Flatten);
flat_like!(ReshapeLayer,Reshape);
/// declares layer and wrapper structs and implements accessor functions, decompose and op for reshaping operations that have dim
macro_rules! cat_like{
	(@ai $layer:ident,$wrap:ident)=>{
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
		impl<X:crate::ops::$wrap,Y> AI<X,Y> for $layer where X::Output:Into<Y>{
			fn forward(&self,input:X)->Y{input._apply(self.dim).into()}
		}
	};
	(@declare $layer:ident,$wrap:ident)=>{
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{dim:i32}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	};
	(@decompose $layer:ident,$wrap:ident)=>{
		impl Decompose for $layer{
			fn compose(dim:Self::Decomposition)->Self{
				Self{dim}
			}
			fn decompose(self)->Self::Decomposition{self.dim}
			fn decompose_cloned(&self)->Self::Decomposition{self.dim}
			type Decomposition=i32;
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose((inner,layer):Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:$layer::compose(layer)}
			}
			fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.layer.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.layer.decompose_cloned())}
			type Decomposition=(A::Decomposition,<$layer as Decompose>::Decomposition);
		}
	};
	(@impl $layer:ident,$wrap:ident)=>{
		impl $layer{
			/// gets the dimension
			pub fn get_dim(&self)->i32{self.dim}
			/// creates a new layer
			pub fn new(dim:i32)->Self{Self::default().with_dim(dim)}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:i32){self.dim=dim}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
				self.dim=dim;
				self
			}
		}
		impl<A:UnwrapInner> UnwrapInner for $wrap<A>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A> $wrap<A>{
			pub fn get_dim(&self)->i32{self.layer.dim}
			/// references the inner value
			pub fn inner(&self)->&A{&self.inner}
			/// references the inner value
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// converts into the inner value
			pub fn into_inner(self)->A{self.inner}
			/// creates a new layer
			pub fn new(dim:i32,inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new(dim)}
			}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:i32){self.layer.dim=dim}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
				self.layer.dim=dim;
				self
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
		}
		impl<M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $layer where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
		}
	};
	(@op $layer:ident,$wrap:ident)=>{
		impl Op for $layer{
			type Output=Vec<f32>;
		}
		impl<A:Op<Output=Y>,Y:crate::ops::$wrap<Output=Z>,Z> Op for $wrap<A> where $layer:AI<Y,Z>{
			type Output=Z;
		}
	};
	($layer:ident,$wrap:ident)=>{
		cat_like!(@ai @declare @decompose @impl @op $layer,$wrap);
	};
	($(@$command:tt)* $layer:ident,$wrap:ident)=>{
		$(cat_like!(@$command $layer,$wrap);)*
	};
}
/// declares layer and wrapper structs and implements accessor functions, decompose and op for reshaping operations that have generic
macro_rules! flat_like{
	(@ai $layer:ident,$wrap:ident)=>{
		impl<A:AI<X,Y>+Op<Output=Y>,R:Clone,X,Y,Z> AI<X,Z> for $wrap<A,R> where $layer<R>:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
		impl<X:crate::ops::$wrap<R>,R:Clone,Y> AI<X,Y> for $layer<R> where X::Output:Into<Y>{
			fn forward(&self,input:X)->Y{input._apply(self.args.clone()).into()}
		}
	};
	(@declare $layer:ident,$wrap:ident)=>{
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer<R:Clone>{args:R}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A,R:Clone>{inner:A,layer:$layer<R>}
	};
	(@decompose $layer:ident,$wrap:ident)=>{
		impl<R:Clone+Decompose> Decompose for $layer<R>{
			fn compose(args:Self::Decomposition)->Self{
				Self{args:R::compose(args)}
			}
			fn decompose(self)->Self::Decomposition{self.args.decompose()}
			fn decompose_cloned(&self)->Self::Decomposition{self.args.decompose_cloned()}
			type Decomposition=R::Decomposition;
		}
		impl<A:Decompose,R:Clone+Decompose> Decompose for $wrap<A,R>{
			fn compose((inner,layer):Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:$layer::compose(layer)}
			}
			fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.layer.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.layer.decompose_cloned())}
			type Decomposition=(A::Decomposition,<$layer<R> as Decompose>::Decomposition);
		}
	};
	(@impl $layer:ident,$wrap:ident)=>{
		impl<R:Clone> $layer<R>{
			/// gets the args
			pub fn args(&self)->&R{&self.args}
			/// gets the args
			pub fn args_mut(&mut self)->&mut R{&mut self.args}
			/// creates a new layer
			pub fn new(args:R)->Self{
				Self{args}
			}
			/// sets the dimension
			pub fn with_args(mut self,args:R)->Self{
				self.args=args;
				self
			}
		}
		impl<A:UnwrapInner,R:Clone> UnwrapInner for $wrap<A,R>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A,R:Clone> $wrap<A,R>{
			/// gets the args
			pub fn args(&self)->&R{&self.layer.args}
			/// gets the args
			pub fn args_mut(&mut self)->&mut R{&mut self.layer.args}
			/// references the inner value
			pub fn inner(&self)->&A{&self.inner}
			/// references the inner value
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// converts into the inner value
			pub fn into_inner(self)->A{self.inner}
			/// creates a new layer
			pub fn new(args:R,inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new(args)}
			}
			/// sets the args
			pub fn with_args(mut self,args:R)->Self{
				self.layer.args=args;
				self
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B,R> where $wrap<B,R>:Op{
				$wrap{inner,layer:self.layer}
			}
		}
		impl<M:AI<M::Output,M::Output>+Op,R:Clone> IntoSequence<M> for $layer<R> where $layer<R>:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
		}
	};
	(@op $layer:ident,$wrap:ident)=>{
		impl<R:Clone> Op for $layer<R>{
			type Output=Vec<f32>;
		}
		impl<A:Op<Output=Y>,R:Clone,Y:crate::ops::$wrap<R,Output=Z>,Z> Op for $wrap<A,R> where $layer<R>:AI<Y,Z>{
			type Output=Z;
		}
	};
	($layer:ident,$wrap:ident)=>{
		flat_like!(@ai @declare @decompose @impl @op $layer,$wrap);
	};
	($(@$command:tt)* $layer:ident,$wrap:ident)=>{
		$(flat_like!(@$command $layer,$wrap);)*
	};
}
use {cat_like,flat_like};
use crate::{AI,Decompose,IntoSequence,Op,UnwrapInner,builtin::Sequential};
use serde::{Deserialize,Serialize};
