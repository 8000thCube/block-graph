impl<C:AI<V,V>,V:Default+Merge> AI<Vec<V>,Vec<V>> for Graph<C>{
	fn forward(&self,input:Vec<V>)->Vec<V>{//TODO fix hio somehow
		let (connections,connectivity)=(&self.connections,&self.connectivity);
		let (hidden,inputs,outputs)=(self.hidden,self.inputs,self.outputs);
		let mut slots=input;
		slots.resize_with(hidden+inputs+outputs,Default::default);

		connectivity.iter().for_each(|&(c,x,y)|{
			let x=take(&mut slots[x]);
			slots[y].merge(connections[c].forward(x));
		});
		slots.drain(0..hidden+inputs);
		slots
	}
	fn forward_mut(&mut self,input:Vec<V>)->Vec<V>{
		let (connections,connectivity)=(&mut self.connections,&self.connectivity);
		let (hidden,inputs,outputs)=(self.hidden,self.inputs,self.outputs);
		let mut slots=input;
		slots.resize_with(hidden+inputs+outputs,Default::default);

		connectivity.iter().for_each(|&(c,x,y)|{
			let x=take(&mut slots[x]);
			slots[y].merge(connections[c].forward_mut(x));
		});
		slots.drain(0..hidden+inputs);
		slots
	}
}
impl<C:Decompose> Decompose for Graph<C>{
	fn compose((connections,connectivity):Self::Decomposition)->Self{
		let connections:Vec<C>=connections.into_iter().map(C::compose).collect();
		let mut connectivity:Vec<(usize,usize,usize)>=connectivity.chunks_exact(3).map(|x|(x[0],x[1],x[2])).collect();
		let (hidden,inputs,outputs)=connectivity.pop().unwrap();
		Self{connections,connectivity,hidden,inputs,outputs}
	}
	fn decompose(self)->Self::Decomposition{(self.connections.into_iter().map(C::decompose).collect(),self.connectivity.into_iter().chain([(self.hidden,self.inputs,self.outputs)]).flat_map(|(x,y,z)|[x,y,z]).collect())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.connections.iter().map(C::decompose_cloned).collect(),self.connectivity.iter().copied().chain([(self.hidden,self.inputs,self.outputs)]).flat_map(|(x,y,z)|[x,y,z]).collect())}
	type Decomposition=(Vec<C::Decomposition>,Vec<usize>);
}
impl<C:Op> Op for Graph<C>{
	type Output=Vec<C::Output>;
}
impl<C> Default for Graph<C>{
	fn default()->Self{
		Self{connections:Vec::new(),connectivity:Vec::new(),hidden:0,inputs:0,outputs:0}
	}
}
impl<C> Graph<C>{
	/// creates a new empty graph
	pub const fn new()->Self{
		Self{connections:Vec::new(),connectivity:Vec::new(),hidden:0,inputs:0,outputs:0}
	}
	/// adds a connection between two vertices
	pub fn add_connection(&mut self,connection:C,input:usize,output:usize)->usize{
		(self.hidden,self.inputs,self.outputs)=(0,0,0);
		let cx=self.connectivity.len();
		self.connections.push(connection);
		self.connectivity.push((cx,input,output));
		cx
	}
	/// connects two graph between vertices with the connection specified by index. The connections run in the order they are connected
	pub fn connect(&mut self,connection:usize,input:usize,output:usize){
		(self.hidden,self.inputs,self.outputs)=(0,0,0);
		self.connectivity.push((connection,input,output))
	}
	/// adds a connection that doesn't anything yet
	pub fn push(&mut self,connection:C)->usize{
		let cx=self.connections.len();
		self.connections.push(connection);
		cx
	}
}
impl<E> Merge for Vec<E>{
	fn merge(&mut self,other:Self){self.extend(other)}
}
#[derive(Clone,Debug,Eq,PartialEq)]
/// graph like ai operation structure. The connections run in the order they are connected
pub struct Graph<C>{connections:Vec<C>,connectivity:Vec<(usize,usize,usize)>,hidden:usize,inputs:usize,outputs:usize}
/// trait to allow merging multiple outputs into one graph node
pub trait Merge{
	/// merges the other into self, taking out of other if convenient
	fn merge(&mut self,other:Self);
}
use crate::ai::{AI,Decompose,Op};
use std::mem::take;
