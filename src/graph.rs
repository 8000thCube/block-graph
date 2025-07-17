impl VertexLabels{
	/// creates a new vertex label map
	pub fn new()->Self{
		Self{labelmap:HashMap::new()}
	}
	/// converts into a list of strings where the index corresponds to the index label
	pub fn into_list(self)->Vec<String>{
		let map=self.labelmap;
		let mut v=vec![String::new();map.len()];

		map.into_iter().for_each(|(s,n)|v[n]=s);
		v
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
			let x=if clear>0{take(&mut hidden[input])}else{hidden[input].clone()};
			let y=layers[layer].forward(x);
			hidden[output].merge(y);
		});
		let mut n=0;
		slots.retain(|_x|{
			let remove=n<inputcount||nodes[n-inputcount].0==0||nodes[n-inputcount].1>0;
			n+=1;
			!remove
		});
		slots
	}
	fn forward_mut(&mut self,input:Vec<V>)->Vec<V>{
		let (connections,nodes)=(&self.connections,&self.nodes);
		let (inputcount,nodecount)=(input.len(),nodes.len());
		let layers=&mut self.layers;
		let mut slots=input;

		slots.resize_with(inputcount+nodecount,Default::default);
		let (input,hidden)=slots.split_at_mut(inputcount);
		input.iter_mut().zip(hidden.iter_mut().zip(nodes.iter()).filter_map(|(h,&(icount,_ocount))|(icount==0).then_some(h))).for_each(|(i,h)|*h=take(i));
		connections.iter().for_each(|&(clear,layer,input,output)|{
			let x=if clear>0{take(&mut hidden[input])}else{hidden[input].clone()};
			let y=layers[layer].forward_mut(x);
			hidden[output].merge(y);
		});
		let mut n=0;
		slots.retain(|_x|{
			let remove=n<inputcount||nodes[n-inputcount].0==0||nodes[n-inputcount].1>0;
			n+=1;
			!remove
		});
		slots
	}
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
	#[track_caller]
	/// adds a connection between two vertices reusing a layer index
	pub fn add_connection(&mut self,clear:bool,input:usize,layer:usize,output:usize){
		assert!(layer<self.layers.len(),"must use an index of an existing layer");
		let (connections,nodes)=(&mut self.connections,&mut self.nodes);
		let nodecount=nodes.len();

		connections.push((clear as usize,layer,input,output));
		nodes.resize((input+1).max(nodecount).max(output+1),(0,0));
		nodes[input].1+=1;
		nodes[output].0+=1;
	}
	/// adds a connection between vertices, returning the layer index
	pub fn connect<A:Into<C>>(&mut self,clear:bool,input:usize,layer:A,output:usize)->usize{
		let (connections,layers,nodes)=(&mut self.connections,&mut self.layers,&mut self.nodes);
		let (layercount,nodecount)=(layers.len(),nodes.len());

		connections.push((clear as usize,layercount,input,output));
		layers.push(layer.into());
		nodes.resize((input+1).max(nodecount).max(output+1),(0,0));
		nodes[input].1+=1;
		nodes[output].0+=1;
		layercount
	}
	/// topologically sorts the graph. connections in the same topological position will remain in the same relative order. node clearing will be moved to the last output of each node
	pub fn sort(&mut self){//TODO test
		let connections=&mut self.connections;
		let nodes=&self.nodes;
		let mut nodeconnectiondata=vec![0;connections.len()+nodes.len()];
		let mut nodeconnectiondata=nodeconnectiondata.as_mut_slice();
		let mut nodeconnections:Vec<(&mut [usize],&mut [usize])>=Vec::with_capacity(nodes.len());
		for &(inputs,_outputs) in nodes.iter(){
			let (filled,nc)=nodeconnectiondata.split_at_mut(1);
			let (inputs,nc)=nc.split_at_mut(inputs);

			nodeconnectiondata=nc;
			nodeconnections.push((filled,inputs));
		}
		for (n,(_clear,_input,_layer,output)) in connections.iter().enumerate(){
			let (filled,inputs)=&mut nodeconnections[*output];
			inputs[filled[0]]=n;
			filled[0]+=1;
		}

		let mut newconnections:Vec<(usize,usize,usize,usize)>=Vec::with_capacity(connections.len());
		for (&(_inputcount,outputcount),(_filled,inputs)) in nodes.iter().zip(nodeconnections.iter()){
			if outputcount>0{continue}
			newconnections.extend(inputs.iter().rev().map(|&n|connections[n]));
		}
		for n in 0..connections.len(){
			let (clear,input,_layer,_output)=&mut newconnections[n];
			let (filled,inputs)=&nodeconnections[*input];

			if filled[0]==0{
				*clear=0;
			}else{
				*clear=1;
				newconnections.extend(inputs.iter().rev().map(|&n|connections[n]));
			}
		}
		newconnections.reverse();
		*connections=newconnections;
	}
}
impl<C:Decompose> Decompose for Graph<C>{
	fn compose((layers,connections,nodes):Self::Decomposition)->Self{
		let layers:Vec<C>=layers.into_iter().map(C::compose).collect();
		Self{connections,layers,nodes}
	}
	fn decompose(self)->Self::Decomposition{(self.layers.into_iter().map(C::decompose).collect(),self.connections.clone(),self.nodes.clone())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.layers.iter().map(C::decompose_cloned).collect(),self.connections.clone(),self.nodes.clone())}
	type Decomposition=(Vec<C::Decomposition>,Vec<(usize,usize,usize,usize)>,Vec<(usize,usize)>);
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
pub struct Graph<C>{connections:Vec<(usize,usize,usize,usize)>,nodes:Vec<(usize,usize)>,layers:Vec<C>}
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
