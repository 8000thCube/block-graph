impl Abs for f32{
	fn abs(self)->Self::Output{f32::abs(self)}
	type Output=f32;
}
impl SquaredError for f32{
	fn squared_error(self,rhs:f32)->Self::Output{
		let d=self-rhs;
		d*d
	}
	type Output=f32;
}
/// trait to represent the operation
pub trait Abs{
	/// computes the operation
	fn abs(self)->Self::Output;
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
