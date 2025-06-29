impl VertexLabels{
	/// creates a new vertex label map
	pub fn new()->Self{
		Self{labelmap:HashMap::new()}
	}
	/// produces an index label for the string
	pub fn label<S:AsRef<str>>(&mut self,s:S)->usize{
		let l=self.labelmap.len();
		*self.labelmap.entry(s.as_ref().to_string()).or_insert(l)
	}
}
impl<A:AI<Vec<X>,Vec<Y>>,X,Y> AI<X,Y> for Unvec<A>{
	fn forward(&self,input:X)->Y{self.0.forward(vec![input]).into_iter().next().unwrap()}
}
impl<A:Decompose> Decompose for Unvec<A>{
	fn compose(decomposition:Self::Decomposition)->Self{Self(A::compose(decomposition))}
	fn decompose(self)->Self::Decomposition{self.0.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.0.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Op<Output=Vec<Y>>,Y> Op for Unvec<A>{
	type Output=Y;
}
impl<C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> AI<Vec<V>,Vec<V>> for Graph<C>{
	fn forward(&self,input:Vec<V>)->Vec<V>{
		let (connections,nodes,layers)=(&self.connections,&self.nodes,&self.layers);
		let (inputcount,nodecount)=(input.len(),nodes.len());
		let mut slots=input;

		slots.resize_with(inputcount+nodecount,Default::default);
		let (input,hidden)=slots.split_at_mut(inputcount);
		input.iter_mut().zip(hidden.iter_mut().zip(nodes.iter()).filter_map(|(h,&(icount,_ocount))|(icount==0).then_some(h))).for_each(|(i,h)|*h=take(i));
		connections.iter().for_each(|&(clear,layer,input,output)|{
			let x=if clear{take(&mut hidden[input])}else{hidden[input].clone()};
			let y=layers[layer].forward(x);
			hidden[output].merge(y);
		});
		let mut n=0;
		slots.retain(|_x|{
			let remove=n<inputcount||nodes[n-inputcount].1>0;
			n+=1;
			!remove
		});
		slots
	}
	//TODO forward_mut
}
impl<C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> Default for Graph<C>{
	fn default()->Self{
		Self{connections:Vec::default(),nodes:Vec::default(),layers:Vec::default()}
	}
}
impl<C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> Graph<C>{
	/// creates a new empty graph
	pub fn new()->Self{
		Self{connections:Vec::new(),nodes:Vec::new(),layers:Vec::new()}
	}
	/// adds a connection between vertices, returning the connection index
	pub fn connect<A:Into<C>>(&mut self,clear:bool,input:usize,layer:A,output:usize)->usize{
		let (connections,layers,nodes)=(&mut self.connections,&mut self.layers,&mut self.nodes);
		let (layercount,nodecount)=(layers.len(),nodes.len());

		connections.push((clear,layercount,input,output));
		layers.push(layer.into());
		nodes.resize((input+1).max(nodecount).max(output+1),(0,0));
		nodes[input].1+=1;
		nodes[output].0+=1;
		layercount
	}
}
impl<C:Decompose> Decompose for Graph<C>{
	fn compose((layers,connections,nodes):Self::Decomposition)->Self{
		let layers:Vec<C>=layers.into_iter().map(C::compose).collect();
		Self{connections,layers,nodes}
	}
	fn decompose(self)->Self::Decomposition{(self.layers.into_iter().map(C::decompose).collect(),self.connections.clone(),self.nodes.clone())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.layers.iter().map(C::decompose_cloned).collect(),self.connections.clone(),self.nodes.clone())}
	type Decomposition=(Vec<C::Decomposition>,Vec<(bool,usize,usize,usize)>,Vec<(usize,usize)>);
}
impl<C:Op> Op for Graph<C>{
	type Output=Vec<C::Output>;
}
impl<E:Merge> Merge for Option<E>{
	fn merge(&mut self,other:Self){
		match (self,other){(Some(this),Some(other))=>this.merge(other),(this,Some(other))=>*this=Some(other),_=>()}
	}
}
impl<E> Merge for Vec<E>{
	fn merge(&mut self,other:Self){self.extend(other)}
}
#[derive(Clone,Debug)]
/// graph like ai operation structure. The connections run in the order they are connected
pub struct Graph<C>{connections:Vec<(bool,usize,usize,usize)>,nodes:Vec<(usize,usize)>,layers:Vec<C>}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
/// wraps the graph so it can take singular io
pub struct Unvec<A>(pub A);
#[derive(Clone,Debug,Default)]
/// formatted string to id mapper for node naming convenience
pub struct VertexLabels{labelmap:HashMap<String,usize>}
/// trait to allow merging multiple outputs into one graph node
pub trait Merge{
	/// merges the other into self, taking out of other if convenient
	fn merge(&mut self,other:Self);
}
use crate::ai::{AI,Decompose,Op};
use std::{collections::HashMap,mem::take};
