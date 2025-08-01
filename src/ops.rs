impl Abs for f32{
	fn abs(self)->Self::Output{f32::abs(self)}
	type Output=f32;
}
/// trait to represent the operation
pub trait Abs{
	/// computes the operation
	fn abs(self)->Self::Output;
	/// the output type
	type Output;
}
