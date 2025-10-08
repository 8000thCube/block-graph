impl AsRef<Self> for Shape{//TODO more reref stuff
	fn as_ref(&self)->&Self{self}
}
impl Default for Reshape{
	fn default()->Self{Self::Recursive(Vec::new())}
}
impl Default for Shape{
	fn default()->Self{Self::Recursive(Vec::new())}
}
impl From<[isize;1]> for Reshape{
	fn from(dims:[isize;1])->Self{R1(dims)}
}
impl From<[isize;2]> for Reshape{
	fn from(dims:[isize;2])->Self{R2(dims)}
}
impl From<[isize;3]> for Reshape{
	fn from(dims:[isize;3])->Self{R3(dims)}
}
impl From<[isize;4]> for Reshape{
	fn from(dims:[isize;4])->Self{R4(dims)}
}
impl From<[isize;5]> for Reshape{
	fn from(dims:[isize;5])->Self{R5(dims)}
}
impl From<[isize;6]> for Reshape{
	fn from(dims:[isize;6])->Self{R6(dims)}
}
impl From<[isize;7]> for Reshape{
	fn from(dims:[isize;7])->Self{R7(dims)}
}
impl From<[isize;8]> for Reshape{
	fn from(dims:[isize;8])->Self{R8(dims)}
}
impl From<[usize;1]> for Reshape{
	fn from(dims:[usize;1])->Self{R1(dims.map(|d|d as isize))}
}
impl From<[usize;2]> for Reshape{
	fn from(dims:[usize;2])->Self{R2(dims.map(|d|d as isize))}
}
impl From<[usize;3]> for Reshape{
	fn from(dims:[usize;3])->Self{R3(dims.map(|d|d as isize))}
}
impl From<[usize;4]> for Reshape{
	fn from(dims:[usize;4])->Self{R4(dims.map(|d|d as isize))}
}
impl From<[usize;5]> for Reshape{
	fn from(dims:[usize;5])->Self{R5(dims.map(|d|d as isize))}
}
impl From<[usize;6]> for Reshape{
	fn from(dims:[usize;6])->Self{R6(dims.map(|d|d as isize))}
}
impl From<[usize;7]> for Reshape{
	fn from(dims:[usize;7])->Self{R7(dims.map(|d|d as isize))}
}
impl From<[usize;8]> for Reshape{
	fn from(dims:[usize;8])->Self{R8(dims.map(|d|d as isize))}
}
impl Reshape{
	/// counts the recursive depth
	pub fn depth(&self)->usize{
		match self{
			R1(_)=>1,
			R2(_)=>1,
			R3(_)=>1,
			R4(_)=>1,
			R5(_)=>1,
			R6(_)=>1,
			R7(_)=>1,
			R8(_)=>1,
			Reshape::Recursive(v)=>v.iter().map(Reshape::depth).max().unwrap_or(0)
		}
	}
	/// converts to the eight dimensional array type by extending with ones. The original data will be placed according to the alignment. Multi and incompatible types will be all ones
	pub fn to_array(self,alignment:Alignment)->[isize;8]{
		let mut result=[1;8];
		let slice=match &self{R1(x)=>x.as_slice(),R2(x)=>x.as_slice(),R3(x)=>x.as_slice(),R4(x)=>x.as_slice(),R5(x)=>x.as_slice(),R6(x)=>x.as_slice(),R7(x)=>x.as_slice(),R8(x)=>x.as_slice(),Reshape::Recursive(_r)=>return result};
		let l=slice.len();
		match alignment{Alignment::Center=>result[4-l/2..][..l].copy_from_slice(slice),Alignment::Left=>result[..l].copy_from_slice(slice),Alignment::Right=>result[8-l..].copy_from_slice(slice)}
		result
	}
}
impl Shape{
	/// counts the number of components if possible. returns none if incompatible or if a non recursive multi shape of more than 0 tensors
	pub fn count(&self)->Option<usize>{
		match self{
			Shape::Incompatible(_e)=>None,
			Shape::Multi(n)=>if *n==0{Some(0)}else{None},
			Shape::Recursive(v)=>{
				let mut s=0;
				for v in v{s+=v.count()?}
				Some(s)
			},
			X1(x)=>Some(x.iter().product()),
			X2(x)=>Some(x.iter().product()),
			X3(x)=>Some(x.iter().product()),
			X4(x)=>Some(x.iter().product()),
			X5(x)=>Some(x.iter().product()),
			X6(x)=>Some(x.iter().product()),
			X7(x)=>Some(x.iter().product()),
			X8(x)=>Some(x.iter().product())
		}
	}
	/// converts to the eight dimensional array type by extending with ones. The original data will be placed according to the alignment. Multi and incompatible types will be all ones
	pub fn to_array(self,alignment:Alignment)->[usize;8]{
		let mut result=[1;8];
		let slice=match &self{Shape::Incompatible(_e)=>return result,Shape::Multi(_v)=>return result,Shape::Recursive(_r)=>return result,X1(x)=>x.as_slice(),X2(x)=>x.as_slice(),X3(x)=>x.as_slice(),X4(x)=>x.as_slice(),X5(x)=>x.as_slice(),X6(x)=>x.as_slice(),X7(x)=>x.as_slice(),X8(x)=>x.as_slice()};
		let l=slice.len();
		match alignment{Alignment::Center=>result[4-l/2..][..l].copy_from_slice(slice),Alignment::Left=>result[..l].copy_from_slice(slice),Alignment::Right=>result[8-l..].copy_from_slice(slice)}
		result
	}
}
#[derive(Clone,Copy,Debug,Eq,PartialEq,Deserialize,Serialize)]
/// enumerates kinds for values
pub enum Kind{Bool,Float,Incompatible,Int,Multi}
#[derive(Clone,Debug,Deserialize,Serialize)]
/// value reshaping arguments
pub enum Reshape{R1([isize;1]),R2([isize;2]),R3([isize;3]),R4([isize;4]),R5([isize;5]),R6([isize;6]),R7([isize;7]),R8([isize;8]),Recursive(Vec<Reshape>)}
#[derive(Clone,Debug,Deserialize,Serialize)]// TODO eq that doesn't include the payload of incompatible
/// tensor shapes for Value
pub enum Shape{Incompatible(String),Multi(usize),Recursive(Vec<Shape>),X1([usize;1]),X2([usize;2]),X3([usize;3]),X4([usize;4]),X5([usize;5]),X6([usize;6]),X7([usize;7]),X8([usize;8])}
use Reshape::{R1,R2,R3,R4,R5,R6,R7,R8};
use Shape::{X1,X2,X3,X4,X5,X6,X7,X8};
use crate::builtin::Alignment;
use serde::{Deserialize,Serialize};
