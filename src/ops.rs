impl Abs for f32{// TODO implement operations for result types
	fn abs(self)->Self::Output{f32::abs(self)}
	type Output=f32;
}
impl Cat for Vec<f32>{
	fn cat(self,_dim:i32)->Self::Output{panic!("cannot cat scalars. it's only implemented to make vector generic work correctly")}
	type Output=Self;
}
impl Rank for f32{
	fn dynamic_rank(&self)->usize{0}
	fn type_rank()->usize{0}
}
impl Squeeze for f32{
	fn squeeze(self,_dim:i32)->Self::Output{panic!("cannot squeeze a scalar. it's only implemented to make vector generic work correctly")}
	type Output=Self;
}
impl SquaredError for f32{
	fn squared_error(self,rhs:f32)->Self::Output{
		let d=self-rhs;
		d*d
	}
	type Output=f32;
}
impl Stack for f32{
	fn stack(self,_dim:i32)->Self::Output{panic!("cannot stack a scalar. it's only implemented to make vector generic work correctly")}
	type Output=Self;
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
pub fn unsqueezde_vec_test(input:Vec<Vec<f32>>)->Vec<Vec<Vec<f32>>>{input.unsqueeze(0)}
impl<T:Squeeze> Squeeze for Vec<T> where Vec<T::Output>:Into<T>,Vec<T>:Rank{
	fn squeeze(self,mut dim:i32)->Self::Output{
		let rank=self.rank() as i32;

		if !(-rank..rank).contains(&dim){panic!("squeeze dim out of bounds")}
		if dim<0{dim+=rank}
		if dim==0{
			if self.len()!=1{panic!("cannot squeeze a dim whose size is not 1")}
			self.into_iter().next().unwrap()
		}else{
			dim-=1;
			let output:Vec<T::Output>=self.into_iter().map(|x|x.squeeze(dim)).collect();

			output.into()
		}
	}
	type Output=T;
}
impl<T:Unsqueeze> Unsqueeze for Vec<T> where T::Output:Into<Vec<T>>,Vec<T>:Rank{
	fn unsqueeze(self,mut dim:i32)->Self::Output{
		let rank=self.rank() as i32;

		if !(-rank..rank).contains(&dim){panic!("squeeze dim out of bounds")}
		if dim<0{dim+=rank+1}
		if dim==0{return vec![self]}
		self.into_iter().map(|x|x.unsqueeze(dim).into()).collect()
	}
	type Output=Vec<Vec<T>>;
}
impl<T> From<UnsqueezeScalar<T>> for Vec<T>{
	fn from(value:UnsqueezeScalar<T>)->Vec<T>{vec![value.0]}
}
impl<T> SwapDims for Vec<Vec<T>> where Vec<T>:Rank+SwapDims<Output=Vec<T>>{
	fn swap_dims(self,mut a:i32,mut b:i32)->Self::Output{
		let rank=self.rank() as i32;

		if !(-rank..rank).contains(&a){panic!("swap dim out of bounds")}
		if !(-rank..rank).contains(&b){panic!("swap dim out of bounds")}
		if a<0{a+=rank}
		if b<0{b+=rank}
		if a==b{return self}

		(a,b)=(a.min(b),a.max(b));

		let recur=if a>0{return self.into_iter().map(|x|x.swap_dims(a-1,b-1)).collect()}else if b>1{self.into_iter().map(|x|x.swap_dims(0,b-1)).collect()}else{self};

		let mut iterators:Vec<_>=recur.into_iter().map(Vec::into_iter).collect();
		(0..).map_while(|_|{
			let v:Vec<T>=iterators.iter_mut().filter_map(Iterator::next).collect();
			(v.len()>0).then_some(v)
		}).collect()
	}
	type Output=Self;
}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
/// unsqueezed scalar that can be converted to vector type
pub struct UnsqueezeScalar<T>(pub T);
/// trait to represent the operation
pub trait Abs{
	/// computes the operation
	fn abs(self)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait Cat{
	/// concatenates the data along the given axis
	fn cat(self,dim:i32)->Self::Output;
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
/// trait to represent the operation
pub trait Squeeze{
	/// computes the operation
	fn squeeze(self,dim:i32)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait SwapDims{
	/// computes the operation
	fn swap_dims(self,a:i32,b:i32)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait SquaredError<R=Self>{
	/// computes the operation
	fn squared_error(self,rhs:R)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait Stack{
	/// stacks the data along the given axis
	fn stack(self,dim:i32)->Self::Output;
	/// the output type
	type Output;
}
/// trait to represent the operation
pub trait Unsqueeze{
	/// computes the operation
	fn unsqueeze(self,dim:i32)->Self::Output;
	/// the output type
	type Output;
}
