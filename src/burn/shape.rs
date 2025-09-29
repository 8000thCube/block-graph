impl AsRef<Self> for Shape{//TODO more reref stuff
	fn as_ref(&self)->&Self{self}
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
#[derive(Clone,Debug,Deserialize,Serialize)]// TODO eq that doesn't include the payload of incompatible
/// tensor shapes for Value
pub enum Shape{Incompatible(String),Multi(usize),Recursive(Vec<Shape>),X1([usize;1]),X2([usize;2]),X3([usize;3]),X4([usize;4]),X5([usize;5]),X6([usize;6]),X7([usize;7]),X8([usize;8])}
use Shape::{X1,X2,X3,X4,X5,X6,X7,X8};
use crate::builtin::Alignment;
use serde::{Deserialize,Serialize};
