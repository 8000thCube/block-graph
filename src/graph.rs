impl Default for GraphIO{
	fn default()->Self{Self::Hidden(0)}
}
impl From<GraphIn> for GraphIO{
	fn from(GraphIn(v):GraphIn)->Self{Self::Input(v)}
}
impl From<GraphOut> for GraphIO{
	fn from(GraphOut(v):GraphOut)->Self{Self::Output(v)}
}
impl From<usize> for GraphIO{
	fn from(v:usize)->Self{Self::Hidden(v)}
}
impl GraphIO{
	#[track_caller]
	/// encodes as an isize with !n for inputs or outputs and n for hidden
	fn as_input_isize(self)->isize{
		(match self{Self::Hidden(n)=>n,Self::Input(n)=>!n,Self::Output(_n)=>panic!("cannot read from graph output. try specifying an input instead")}) as isize
	}
	#[track_caller]
	/// encodes as an isize with !n for inputs or outputs and n for hidden
	fn as_output_isize(self)->isize{
		(match self{Self::Hidden(n)=>n,Self::Input(_n)=>panic!("cannot write to graph input. try specifying an output instead"),Self::Output(n)=>!n}) as isize
	}
}
impl VertexLabels{
	fn _produce_index<S:Into<Cow<'static,str>>>(&mut self,label:S)->usize{
		let Self(map)=self;
		let nextid=map.len();

		*map.entry(label.into()).or_insert(nextid)
	}
	/// gets or adds an index associated with the str.
	pub fn labeled(&mut self,label:&'static str)->usize{self._produce_index(label)}
	/// creates a new listing of vertex labels
	pub fn new()->Self{Self(HashMap::new())}
	/// gets or adds an index associated with the string
	pub fn produce_index(&mut self,label:String)->usize{self._produce_index(label)}
}
impl<B:Backend> Merge for BurnValue<B>{
	fn merge(&mut self,other:Self){
		match (take(self),other){
			(BurnValue::Multi(mut u),BurnValue::Multi(v))=>{
				u.extend(v);
				*self=u.into();
			},
			(BurnValue::Multi(mut u),v)=>if u.len()==0{
				*self=v;
			}else{
				u.push(v);
				*self=u.into();
			},
			(u,BurnValue::Multi(mut v))=>if v.len()==0{
				*self=u;
			}else{
				v.push(u);
				*self=v.into();
			},
			(u,v)=>*self=vec![u,v].into()
		}
	}
}
impl<C:AI<V,V>,V:Default+Merge> AI<Vec<V>,Vec<V>> for Graph<C>{
	fn forward(&self,input:Vec<V>)->Vec<V>{
		let (connections,connectivity)=(&self.connections,&self.connectivity);
		let inputs=connectivity.iter().map(|&(_connection,input,_output)|!input).filter(|&x|x>=0).max().map(|x|x as usize+1).unwrap_or(0);
		let hidden=connectivity.iter().flat_map(|&(_connection,input,output)|[input,output]).filter(|&x|x>=0).max().map(|x|x as usize+1).unwrap_or(0);
		let outputs=connectivity.iter().map(|&(_connection,_input,output)|!output).filter(|&x|x>=0).max().map(|x|x as usize+1).unwrap_or(0);
		let mut slots=input;

		slots.resize_with(hidden+inputs+outputs,Default::default);
		let (i,o)=slots.split_at_mut(inputs);
		let (h,o)=o.split_at_mut(hidden);
		connectivity.iter().for_each(|&(c,x,y)|{
			let x=take(if x<0{&mut i[!x as usize]}else{&mut h[x as usize]});
			if y<0{&mut o[!y as usize]}else{&mut h[y as usize]}.merge(connections[c].forward(x));
		});
		slots.drain(0..hidden+inputs);
		slots
	}
	fn forward_mut(&mut self,input:Vec<V>)->Vec<V>{
		let connections=&mut self.connections;
		let connectivity=&self.connectivity;
		let hidden=connectivity.iter().flat_map(|&(_connection,input,output)|[input,output]).filter(|&x|x>=0).max().map(|x|x as usize+1).unwrap_or(0);
		let inputs=connectivity.iter().map(|&(_connection,input,_output)|!input).filter(|&x|x>=0).max().map(|x|x as usize+1).unwrap_or(0);
		let outputs=connectivity.iter().map(|&(_connection,_input,output)|!output).filter(|&x|x>=0).max().map(|x|x as usize+1).unwrap_or(0);
		let mut slots=input;

		slots.resize_with(hidden+inputs+outputs,Default::default);
		let (i,o)=slots.split_at_mut(inputs);
		let (h,o)=o.split_at_mut(hidden);
		connectivity.iter().for_each(|&(c,x,y)|{
			let x=take(if x<0{&mut i[!x as usize]}else{&mut h[x as usize]});
			if y<0{&mut o[!y as usize]}else{&mut h[y as usize]}.merge(connections[c].forward_mut(x));
		});
		slots.drain(0..hidden+inputs);
		slots
	}
}
impl<C:Decompose> Decompose for Graph<C>{
	fn compose((connections,connectivity):Self::Decomposition)->Self{
		let connections:Vec<C>=connections.into_iter().map(C::compose).collect();
		let connectivity:Vec<(usize,isize,isize)>=connectivity.chunks_exact(3).map(|x|(x[0],x[1] as isize,x[2] as isize)).collect();
		Self{connections,connectivity}
	}
	fn decompose(self)->Self::Decomposition{(self.connections.into_iter().map(C::decompose).collect(),self.connectivity.into_iter().flat_map(|(x,y,z)|[x,y as usize,z as usize]).collect())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.connections.iter().map(C::decompose_cloned).collect(),self.connectivity.iter().copied().flat_map(|(x,y,z)|[x,y as usize,z as usize]).collect())}
	type Decomposition=(Vec<C::Decomposition>,Vec<usize>);
}
impl<C:Op> Op for Graph<C>{
	type Output=Vec<C::Output>;
}
impl<C> Default for Graph<C>{
	fn default()->Self{
		Self{connections:Vec::new(),connectivity:Vec::new()}
	}
}
impl<C> Graph<C>{
	/// creates a new empty graph
	pub const fn new()->Self{
		Self{connections:Vec::new(),connectivity:Vec::new()}
	}
	/// adds a connection that connects two vertices. The connections run in the order they are connected.
	pub fn add_connection<A:Into<C>,I:Into<GraphIO>,J:Into<GraphIO>>(&mut self,connection:A,input:I,output:J)->usize{
		let connection=self.push(connection);
		self.connect(connection,input,output);
		connection
	}
	#[track_caller]
	/// connects two graph between vertices with the connection specified by index. The connections run in the order they are connected.
	pub fn connect<I:Into<GraphIO>,J:Into<GraphIO>>(&mut self,connection:usize,input:I,output:J){
		let connections=self.connections.len();
		let connectivity=&mut self.connectivity;

		assert!(connection<connections,"connection index {connection} must exist. connections length was {connections}");
		let (input,output)=(input.into().as_input_isize(),output.into().as_output_isize());
		connectivity.push((connection,input,output));
	}
	/// adds a connection that doesn't anything yet
	pub fn push<A:Into<C>>(&mut self,connection:A)->usize{
		let cx=self.connections.len();
		self.connections.push(connection.into());
		cx
	}
}
impl<E:Merge> Merge for Option<E>{
	fn merge(&mut self,other:Self){
		match (self,other){(Some(this),Some(other))=>this.merge(other),(this,Some(other))=>*this=Some(other),_=>()}
	}
}
impl<E> Merge for Vec<E>{
	fn merge(&mut self,other:Self){self.extend(other)}
}
#[macro_export]
/// macro to get indices from formatted labels
macro_rules! labeled{
	($labels:ident,$($arg:tt)*)=>($labels.produce_index(format!($($arg)*)));
}
mod tests{
	#[test]
	fn learn_xor(){
		let mut graph:Graph<BurnLayer<Wgpu>>=Graph::new();
		let mut l=VertexLabels::new();
		graph.add_connection(BurnLayer::linear(true,2,5,1.0),GraphIn(0),labeled!(l,"intermediate 1"));
		graph.add_connection(BurnLayer::relu(),labeled!(l,"intermediate 1"),labeled!(l,"intermediate 2"));
		graph.add_connection(BurnLayer::linear(false,5,1,1.0),labeled!(l,"intermediate 2"),GraphOut(0));
		let inputval=BurnValue::from(Tensor::<Wgpu,2>::from_data(TensorData::new([1.0,1.0].to_vec(),[1,2]),&Default::default()));
		let outputval=graph.forward(vec![inputval]).into_iter().next().unwrap();
		if let BurnValue::F2(o)=outputval{
			println!("{o}");
		}

		//panic!("h");
	}
	use burn::backend::Wgpu;
	use crate::ai::BurnLayer;
	use super::*;
}
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
/// enum for whether a node index is a hidden input or output node
pub enum GraphIO{Hidden(usize),Input(usize),Output(usize)}
#[derive(Clone,Debug,Eq,PartialEq)]
/// graph like ai operation structure. The connections run in the order they are connected
pub struct Graph<C>{connections:Vec<C>,connectivity:Vec<(usize,isize,isize)>}
#[derive(Clone,Copy,Debug,Default,Eq,PartialEq)]
/// wrapper for graph input node indices
pub struct GraphIn(usize);
#[derive(Clone,Copy,Debug,Default,Eq,PartialEq)]
/// wrapper for graph output node indices
pub struct GraphOut(usize);
#[derive(Clone,Debug,Default)]
/// formatted string to id mapper for node naming convenience
pub struct VertexLabels(HashMap<Cow<'static,str>,usize>);
/// trait to allow merging multiple outputs into one graph node
pub trait Merge{
	/// merges the other into self, taking out of other if convenient
	fn merge(&mut self,other:Self);
}
pub use labeled;
use burn::prelude::*;
use crate::ai::{AI,BurnValue,Decompose,Op};
use std::{borrow::Cow,collections::HashMap,mem::take};
