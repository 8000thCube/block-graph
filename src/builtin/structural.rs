cat_like!(CatLayer,Cat);
cat_like!(StackLayer,Stack);
impl<E> AI<Vec<Vec<E>>,Vec<E>> for StackLayer{// TODO squeeze unsqueeze so we can properly implement this
	fn forward(&self,_input:Vec<Vec<E>>)->Vec<E>{todo!()}
}
impl<E> AI<Vec<Vec<E>>,Vec<Vec<E>>> for StackLayer{
	fn forward(&self,_input:Vec<Vec<E>>)->Vec<Vec<E>>{todo!()}
}
impl<X> AI<Vec<Vec<X>>,Vec<X>> for CatLayer{
	fn forward(&self,input:Vec<Vec<X>>)->Vec<X>{
		let dim=self.dim;
		assert!(dim==0,"Dimension index was {dim} but a vec only has one tensor dimension");

		let acc=Vec::with_capacity(input.iter().map(|x|x.len()).sum());
		input.into_iter().fold(acc,|mut acc,x|{
			acc.extend(x);
			acc
		})
	}
}
/// declares layer and wrapper structs and implements accessor functions, decompose and op for reduction operations that have dim and mismatch behavior as configuration fields. ai will still have to be externally implemented for the layer stuct
macro_rules! cat_like{
	(@aiwrap $layer:ident,$wrap:ident)=>{
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
	};
	(@declare $layer:ident,$wrap:ident)=>{
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{dim:i32,mismatchbehavior:OnMismatch}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	};
	(@decompose $layer:ident,$wrap:ident)=>{
		impl Decompose for $layer{
			fn compose((dim,mismatchbehavior):Self::Decomposition)->Self{
				Self{dim,mismatchbehavior:OnMismatch::compose(mismatchbehavior)}
			}
			fn decompose(self)->Self::Decomposition{(self.dim,self.mismatchbehavior.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.dim,self.mismatchbehavior.decompose_cloned())}
			type Decomposition=(i32,usize);
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
			/// gets the mismatch behavior
			pub fn get_mismatch_behavior(&self)->OnMismatch{self.mismatchbehavior}
			/// creates a new layer
			pub fn new(dim:i32)->Self{Self::default().with_dim(dim)}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:i32){self.dim=dim}
			/// sets the mismatch behavior
			pub fn set_mismatch_behavior(&mut self,behavior:OnMismatch){self.mismatchbehavior=behavior}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
				self.dim=dim;
				self
			}
			/// sets the mismatch behavior
			pub fn with_mismatch_behavior(mut self,behavior:OnMismatch)->Self{
				self.mismatchbehavior=behavior;
				self
			}
		}
		impl<A> $wrap<A>{
			pub fn get_dim(&self)->i32{self.layer.dim}
			/// gets the mismatch behavior
			pub fn get_mismatch_behavior(&self)->OnMismatch{self.layer.mismatchbehavior}
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
			/// sets the mismatch behavior
			pub fn set_mismatch_behavior(&mut self,behavior:OnMismatch){self.layer.mismatchbehavior=behavior}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
				self.layer.dim=dim;
				self
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
			/// sets the mismatch behavior
			pub fn with_mismatch_behavior(mut self,behavior:OnMismatch)->Self{
				self.layer.mismatchbehavior=behavior;
				self
			}
		}
	};
	(@op $layer:ident,$wrap:ident)=>{
		impl Op for $layer{
			type Output=Vec<()>;
		}
		impl<A:Op<Output=Y>,Y:IntoIterator<Item=Z>,Z> Op for $wrap<A> where $layer:AI<Y,Z>{
			type Output=Z;
		}
	};
	($layer:ident,$wrap:ident)=>{
		cat_like!(@aiwrap @declare @decompose @impl @op $layer,$wrap);
	};
	($(@$command:tt)* $layer:ident,$wrap:ident)=>{
		$(cat_like!(@$command $layer,$wrap);)*
	};
}
use cat_like;
use crate::{AI,Decompose,Op};
use serde::{Deserialize,Serialize};
use super::OnMismatch;
