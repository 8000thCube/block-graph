impl AI<(Vec<f32>,Vec<f32>),f32> for CrossEntropyLayer{
	fn forward(&self,(_output,_target):(Vec<f32>,Vec<f32>))->f32{
		//let t=self.temperature;
		//-if t.i

		//-new().fix_type::<Vec<f32>>().log_softmax().forward_fixed(output).iter().zip(target.iter()).map(|(o,t)|o*t).fold(0.0,|acc,x|acc+x)
		todo!()
	}
}
impl AI<(Vec<f32>,u32),f32> for CrossEntropyLayer{
	fn forward(&self,(output,target):(Vec<f32>,u32))->f32{
		let t=self.temperature;
		-if t.is_nan(){output[target as usize].ln()}else{LogSoftmaxLayer::new(t).forward_fixed(output)[target as usize]}
	}
}
impl AI<Vec<f32>,Vec<f32>> for AbnormalSoftmaxLayer{
	fn forward(&self,input:Vec<f32>)->Vec<f32>{
		let max=input.iter().fold(f32::NEG_INFINITY,|x,&y|if x<y{y}else{x});
		input.into_iter().map(|x|if x==max{1.0}else{(x-max).exp()}).collect()
	}
}
impl AI<Vec<f32>,Vec<f32>> for LogSoftmaxLayer{
	fn forward(&self,input:Vec<f32>)->Vec<f32>{
		let t=self.temperature.recip();
		let mut sum=0.0;
		input.iter().for_each(|x|sum+=(t*x).exp());
		let r=sum.ln();
		let output:Vec<f32>=input.into_iter().map(|x|t*x-r).collect();
		output
	}
}
impl AI<Vec<f32>,Vec<f32>> for SoftmaxLayer{
	fn forward(&self,input:Vec<f32>)->Vec<f32>{
		let t=self.temperature.recip();
		if t.is_nan(){
			let mut count=0;
			let max=input.iter().fold(f32::NEG_INFINITY,|x,&y|if x<y{
				count=0;
				y
			}else{
				if x==y{count+=1}
				x
			});
			let r=(count as f32).recip();
			return input.into_iter().map(|x|if x==max{r}else{0.0}).collect();
		}
		let max=input.iter().fold(f32::NEG_INFINITY,|x,&y|if x<y{y}else{x});
		let mut sum=0.0;
		let intermediate:Vec<f32>=input.into_iter().map(|x|if x==max{1.0}else{((x-max)*t).exp()}).inspect(|y|sum+=y).collect();
		let r=sum.recip();
		let output:Vec<f32>=intermediate.into_iter().map(|y|r*y).collect();
		output
	}
}
impl Op for ChooseLayer{
	type Output=u32;
}
impl Op for CrossEntropyLayer{
	type Output=Vec<f32>;
}
impl<A:AI<X,Y>+Op<Output=Y>,T,X,Y,Z> AI<(X,T),Z> for CrossEntropy<A> where CrossEntropyLayer:AI<(Y,T),Z>{
	fn forward(&self,(input,target):(X,T))->Z{self.layer.forward((self.inner.forward(input),target))}
	fn forward_mut(&mut self,(input,target):(X,T))->Z{self.layer.forward_mut((self.inner.forward_mut(input),target))}
}
impl<A:Op<Output=Y>,Y> Op for Choose<A> where ChooseLayer:AI<Y,u32>{
	type Output=u32;
}
impl<A:Op<Output=Y>,Y> Op for CrossEntropy<A> where CrossEntropyLayer:AI<(Y,Y),Vec<f32>>{
	type Output=Vec<f32>;
}

/// declares layer and wrapper structs and implements accessor functions, decompose and op for reduction operations that have dim and temperature as configuration fields. ai will still have to be externally implemented for the layer stuct
macro_rules! soft_like{
	(@aiwrap $layer:ident,$wrap:ident)=>{
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
	};
	(@declare $layer:ident,$wrap:ident)=>{
		impl Default for $layer{
			fn default()->Self{
				Self{dim:-1,temperature:1.0}
			}
		}
		#[derive(Clone,Copy,Debug,Deserialize,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{dim:i32,temperature:f32}
		#[derive(Clone,Copy,Debug,Default,Deserialize,PartialEq,Serialize)]// TODO eq and hash that do something about the float
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	};
	(@decompose $layer:ident,$wrap:ident)=>{
		impl Decompose for $layer{
			fn compose((dim,temperature):Self::Decomposition)->Self{
				Self{dim,temperature}
			}
			fn decompose(self)->Self::Decomposition{(self.dim,self.temperature)}
			fn decompose_cloned(&self)->Self::Decomposition{(self.dim,self.temperature)}
			type Decomposition=(i32,f32);
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
			/// gets the temperature
			pub fn get_temperature(&self)->f32{self.temperature}
			/// creates a new layer
			pub fn new(temperature:f32)->Self{
				Self{dim:-1,temperature}
			}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:i32){self.dim=dim}
			/// sets the mismatch behavior. A temperature of NaN will make the non soft version if possible. A finite temperature will make the soft version
			pub fn set_temperature(&mut self,temperature:f32){self.temperature=temperature}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
				self.dim=dim;
				self
			}
			/// sets the temperature. A temperature of NaN will make the non soft version if possible. A finite temperature will make the soft version
			pub fn with_temperature(mut self,temperature:f32)->Self{
				self.temperature=temperature;
				self
			}
		}
		impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $wrap<A> where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
		}
		impl<A:UnwrapInner> UnwrapInner for $wrap<A>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A> $wrap<A>{
			pub fn get_dim(&self)->i32{self.layer.dim}
			/// gets the temperature
			pub fn get_temperature(&self)->f32{self.layer.temperature}
			/// references the inner value
			pub fn inner(&self)->&A{&self.inner}
			/// references the inner value
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// converts into the inner value
			pub fn into_inner(self)->A{self.inner}
			/// creates a new layer
			pub fn new(inner:A,temperature:f32)->Self where Self:Op{
				Self{inner,layer:$layer::new(temperature)}
			}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:i32){self.layer.dim=dim}
			/// sets the temperature
			pub fn set_temperature(&mut self,temperature:f32){self.layer.temperature=temperature}
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
			pub fn with_temperature(mut self,temperature:f32)->Self{
				self.layer.temperature=temperature;
				self
			}
		}
	};
	(@op $layer:ident,$wrap:ident)=>{
		impl Op for $layer{
			type Output=Vec<f32>;
		}
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<Y,Vec<f32>>{
			type Output=Vec<f32>;
		}
	};
	($layer:ident,$wrap:ident)=>{
		soft_like!(@aiwrap @declare @decompose @impl @op $layer,$wrap);
	};
	($(@$command:tt)* $layer:ident,$wrap:ident)=>{
		$(soft_like!(@$command $layer,$wrap);)*
	};
}
soft_like!(@aiwrap @declare @decompose @impl ChooseLayer,Choose);
soft_like!(AbnormalSoftmaxLayer,AbnormalSoftmax);
soft_like!(SoftmaxLayer,Softmax);
soft_like!(@declare @decompose @impl CrossEntropyLayer,CrossEntropy);
soft_like!(LogSoftmaxLayer,LogSoftmax);
use soft_like;
use crate::{
	AI,Decompose,IntoSequence,Op,UnwrapInner
};
use serde::{Deserialize,Serialize};
use super::Sequential;
