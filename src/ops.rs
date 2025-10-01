				// TODO macro primitive implementations
impl Abs for f32{// TODO implement operations for result types
	fn abs(self)->Self::Output{f32::abs(self)}
	type Output=f32;
}
impl Rank for f32{
	fn dynamic_rank(&self)->usize{0}
	fn type_rank()->usize{0}
}
impl Squeeze for Vec<f32>{
	fn squeeze(self,dim:i32)->Self::Output{
		if dim!=-1&&dim!=0{panic!("squeeze dim out of bounds")}
		if self.len()!=1{panic!("cannot squeeze a dim whose size is not 1")}
		self[0]
	}
	type Output=f32;
}
impl SquaredError for f32{
	fn squared_error(self,rhs:f32)->Self::Output{
		let d=self-rhs;
		d*d
	}
	type Output=f32;
}
impl Unsqueeze for f32{
	fn unsqueeze(self,dim:i32)->UnsqueezeScalar<f32>{
		if dim==-1||dim==0{UnsqueezeScalar(self)}else{panic!("unsqueeze dim out of bounds")}
	}
	type Output=UnsqueezeScalar<f32>;
}
impl<T:Rank> Rank for Vec<T>{
	fn dynamic_rank(&self)->usize{self.first().map(T::dynamic_rank).unwrap_or_else(T::type_rank)+1}
	fn type_rank()->usize{T::type_rank()+1}
}
impl<T:Squeeze> Squeeze for Vec<Vec<T>> where Vec<T>:Squeeze<Output=T>+Rank{
	fn squeeze(self,mut dim:i32)->Self::Output{
		let rank=self.rank() as i32;

		if !(-rank..rank).contains(&dim){panic!("squeeze dim out of bounds")}
		if dim==0||dim==-rank{
			if self.len()!=1{panic!("cannot squeeze a dim whose size is not 1")}
			self.into_iter().next().unwrap()
		}else{
			if dim>0{dim-=1}
			self.into_iter().map(|x|x.squeeze(dim)).collect()
		}
	}
	type Output=Vec<T>;
}
impl<T:Unsqueeze<Output=U>,U> Stack for Vec<T> where Vec<U>:Cat<Output=Vec<T>>{
	fn stack(self,dim:i32)->Self::Output{
		let unsqueezed:Vec<U>=self.into_iter().map(|x|x.unsqueeze(dim)).collect();
		unsqueezed.cat(dim)
	}
	type Output=Self;
}
impl<T:Unsqueeze> Unsqueeze for Vec<T> where T::Output:Into<Vec<T>>,Vec<T>:Rank{
	fn unsqueeze(self,mut dim:i32)->Self::Output{
		let rank=self.rank() as i32;

		if !(-rank..rank+1).contains(&dim){panic!("unsqueeze dim out of bounds")}
		if dim==0||dim==-rank{return vec![self]}else if dim>0{dim-=1}
		self.into_iter().map(|x|x.unsqueeze(dim).into()).collect()
	}
	type Output=Vec<Vec<T>>;
}
impl<T> From<UnsqueezeScalar<T>> for Vec<T>{
	fn from(value:UnsqueezeScalar<T>)->Vec<T>{vec![value.0]}
}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
/// unsqueezed scalar that can be converted to vector type
pub struct UnsqueezeScalar<T>(pub T);
/// trait to represent the operation
pub trait Abs{
	/// macro convenience version of the primary method
	fn _apply(self)->Self::Output where Self:Sized{self.abs()}
	/// computes the operation
	fn abs(self)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait Cat{
	/// macro convenience version of the primary method
	fn _apply(self,dim:i32)->Self::Output where Self:Sized{self.cat(dim)}
	/// concatenates the data along the given axis
	fn cat(self,dim:i32)->Self::Output;
	/// the output type
	type Output;
}
// flatten
pub trait Flatten<R>{
	/// macro convenience version of the primary method
	fn _apply(self,args:R)->Self::Output where Self:Sized{self.flatten(args)}
	/// flattens
	fn flatten(self,args:R)->Self::Output;
	/// the output type
	type Output;
}
/// get tensor rank
pub trait Rank{
	/// gets the rank
	fn dynamic_rank(&self)->usize;
	/// gets the rank
	fn rank(&self)->usize{self.dynamic_rank()}
	/// gets the rank at a type level. this may be some kind of default if there isn't a clear rank associated with the type
	fn type_rank()->usize where Self:Sized;
}
// reshape
pub trait Reshape<R>{
	/// macro convenience version of the primary method
	fn _apply(self,args:R)->Self::Output where Self:Sized{self.reshape(args)}
	/// reshapes
	fn reshape(self,args:R)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait Squeeze{
	/// macro convenience version of the primary method
	fn _apply(self,dim:i32)->Self::Output where Self:Sized{self.squeeze(dim)}
	/// computes the operation
	fn squeeze(self,dim:i32)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait SwapDims{
	/// macro convenience version of the primary method
	fn _apply(self,a:i32,b:i32)->Self::Output where Self:Sized{self.swap_dims(a,b)}
	/// computes the operation
	fn swap_dims(self,a:i32,b:i32)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait SquaredError<R=Self>{
	/// macro convenience version of the primary method
	fn _apply(self,rhs:R)->Self::Output where Self:Sized{self.squared_error(rhs)}
	/// computes the operation
	fn squared_error(self,rhs:R)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait Stack{
	/// macro convenience version of the primary method
	fn _apply(self,dim:i32)->Self::Output where Self:Sized{self.stack(dim)}
	/// stacks the data along the given axis
	fn stack(self,dim:i32)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait Unsqueeze{
	/// macro convenience version of the primary method
	fn _apply(self,dim:i32)->Self::Output where Self:Sized{self.unsqueeze(dim)}
	/// computes the operation
	fn unsqueeze(self,dim:i32)->Self::Output;
	/// the output type
	type Output;
}
