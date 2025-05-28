

impl Block for Connection{
	fn config(&self)->Box<dyn AnyBlock>{Box::new(Self::from_existing_data(self.block.config(),self.inputname.clone(),self.outputname.clone()))}
	fn duplicate(&self)->Box<dyn AnyBlock>{Box::new(self.clone())}
	fn forward(&self,input:Box<dyn AnyBlock>)->Box<dyn AnyBlock>{self.inner.forward(input)}
	fn init(&self)->Box<dyn AnyBlock>{Box::new(Self::from_existing_data(self.block.init(),self.inputname.clone(),self.outputname.clone()))}
}
impl Clone for Connection{
	fn clone(&self)->Self{Self::from_existing_data(self.block.duplicate(),self.inputname.clone(),self.outputname.clone())}
}
impl Connection{
	fn from_existing_data(block:Box<dyn AnyBlock>,inputname:String,outputname:String)->Self{
		Self{block,inputname,outputname}
	}
}

/// block graph connection
pub struct Connection{block:Box<dyn AnyBlock>,inputname:String,outputname:String}

/// block graph
pub struct Graph{connections:Vec<Connection>,indexstructure:Vec<(bool,usize,usize)>}

use crate::block::{AnyBlock,Block};
