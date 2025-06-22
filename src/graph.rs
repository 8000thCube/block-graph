
impl<C,V> AI<Vec<Vec<V>>,Vec<Vec<V>>> for Graph<C,V>{
	fn forward(&self,input:Vec<Vec<V>>)->Vec<Vec<V>>{
		let (connections,connectivity)=(&self.connections,&self.connectivity);
		let (hidden,inputs,outputs)=(self.hidden,self.inputs,self.outputs);
		let mut slots=input;
		slots.resize_with(hidden+inputs+outputs,Default::default);

		connectivity.iter().for_each(|&(c,x,y)|{
			let x=take(&mut slots[x]);
			slots[y].extend(connections[c].forward(x));
		});
		slots.drain(0..hidden+inputs);
		slots
	}
	fn forward_mut(&mut self,input:Vec<Vec<V>>)->Vec<Vec<V>>{
		let (connections,connectivity)=(&mut self.connections,&self.connectivity);
		let (hidden,inputs,outputs)=(self.hidden,self.inputs,self.outputs);
		let mut slots=input;
		slots.resize_with(hidden+inputs+outputs,Default::default);

		connectivity.iter().for_each(|&(c,x,y)|{
			let x=take(&mut slots[x]);
			slots[y].extend(connections[c].forward_mut(x));
		});
		slots.drain(0..hidden+inputs);
		slots
	}
}


/// graphs like ai operation structure
pub struct Graph<C,V>{connections:Vec<Box<dyn DynAI<C,Vec<V>,Vec<V>>>>,connectivity:Vec<(usize,usize,usize)>,hidden:usize,inputs:usize,outputs:usize}


use crate::ai::{AI,DynAI};
use std::mem::take;
