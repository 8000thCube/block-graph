impl Block for (){
	fn duplicate(&self)->Box<dyn AnyBlock>{Box::new(*self)}
}
impl Block for &dyn AnyBlock{
	fn cast<T:Any>(&self)->Option<&T>{(**self).as_any().downcast_ref()}
	fn cast_mut<T:Any>(&mut self)->Option<&mut T>{None}
	fn config(&self)->Box<dyn AnyBlock>{(**self).config()}
	fn duplicate(&self)->Box<dyn AnyBlock>{(**self).duplicate()}
	fn forward(&self,input:Box<dyn AnyBlock>)->Box<dyn AnyBlock>{(**self).forward(input)}
	fn init(&self)->Box<dyn AnyBlock>{(**self).init()}
}
impl Block for &mut dyn AnyBlock{
	fn cast<T:Any>(&self)->Option<&T>{(**self).as_any().downcast_ref()}
	fn cast_mut<T:Any>(&mut self)->Option<&mut T>{(**self).as_any_mut().downcast_mut()}
	fn config(&self)->Box<dyn AnyBlock>{(**self).config()}
	fn duplicate(&self)->Box<dyn AnyBlock>{(**self).duplicate()}
	fn forward(&self,input:Box<dyn AnyBlock>)->Box<dyn AnyBlock>{(**self).forward(input)}
	fn init(&self)->Box<dyn AnyBlock>{(**self).init()}
}
impl Block for Box<dyn AnyBlock>{
	fn cast<T:Any>(&self)->Option<&T>{(**self).as_any().downcast_ref()}
	fn cast_mut<T:Any>(&mut self)->Option<&mut T>{(**self).as_any_mut().downcast_mut()}
	fn config(&self)->Box<dyn AnyBlock>{(**self).config()}
	fn duplicate(&self)->Box<dyn AnyBlock>{(**self).duplicate()}
	fn forward(&self,input:Box<dyn AnyBlock>)->Box<dyn AnyBlock>{(**self).forward(input)}
	fn init(&self)->Box<dyn AnyBlock>{(**self).init()}
}
impl Block for usize{
	fn duplicate(&self)->Box<dyn AnyBlock>{Box::new(*self)}
}
impl<T:Any+Block> AnyBlock for T{
	fn as_any(&self)->&dyn Any{self}
	fn as_any_mut(&mut self)->&mut dyn Any{self}
}
#[test]
fn cast_check(){
	let block:Box<dyn AnyBlock>=Box::new(());
	assert_eq!(block.cast(),Some(&()));
	let a=Box::new(5_usize);
	let ap:*const dyn AnyBlock=&*a;
	dbg!(ap);
	let b=block.forward(a);
	let bp:*const dyn AnyBlock=&*b;
	dbg!(bp);
	assert_eq!(b.cast(),Some(&5_usize));
}
/// helps convert block to any for casting
pub trait AnyBlock:Any+Block{
	/// converts the reference to an any
	fn as_any(&self)->&dyn Any;
	/// converts the reference to an any
	fn as_any_mut(&mut self)->&mut dyn Any;
}
/// trait for dyanmic graph nn components and values
pub trait Block:Send+Sync{
	/// attempts to cast the reference
	fn cast<T:Any>(&self)->Option<&T> where Self:AnyBlock+Sized{self.as_any().downcast_ref()}
	/// attempts to cast the reference
	fn cast_mut<T:Any>(&mut self)->Option<&mut T> where Self:AnyBlock+Sized{self.as_any_mut().downcast_mut()}
	/// converts initialized layer back into the configuration, should be an inverse of init. this may just duplicate if already a config
	fn config(&self)->Box<dyn AnyBlock>{self.duplicate()}
	/// clones the block dynamically
	fn duplicate(&self)->Box<dyn AnyBlock>;
	/// applies the layer operation if this is a layer. By default it returns self.duplicate() if the input is (), otherwise returns the input
	fn forward(&self,input:Box<dyn AnyBlock>)->Box<dyn AnyBlock>{
		if let Some(_x)=input.cast::<()>(){self.duplicate()}else{input}
	}
	/// converts the configuration to an initialized layer, should be approximate inverse of config. this should just duplicate if already initialized
	fn init(&self)->Box<dyn AnyBlock>{self.duplicate()}
}

use std::any::Any;
