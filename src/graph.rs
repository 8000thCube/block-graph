fn set_bit(x: u64, idx: u8, b: bool) -> u64{(x & !(1 << idx)) | ((b as u64) << idx)}
impl BuildHasher for H{
	fn build_hasher(&self)->Self::Hasher{*self}
	type Hasher=H;
}
impl Decompose for Label{
	fn compose(label:Self::Decomposition)->Self{label.into()}
	fn decompose(self)->Self::Decomposition{self.to_string()}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone().decompose()}
	type Decomposition=String;
}
impl Default for Label{
	fn default()->Self{
		Self{id:rand::random(),name:None}
	}
}
impl Display for Label{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		if self.id!=0{UpperHex::fmt(&self.id,f)?}
		if let Some(text)=&self.name{
			write!(f,"{}{text}",if self.id==0{""}else{": "})?;
		}
		Ok(())
	}
}
impl From<Option<String>> for Label{
	fn from(value:Option<String>)->Self{
		if let Some(v)=value{v.into()}else{Self::new()}
	}
}
impl From<String> for Label{
	fn from(value:String)->Self{Self::from_str(&value).unwrap()}
}
impl From<i32> for Label{
	fn from(value:i32)->Self{(value as u64).into()}
}
impl From<u64> for Label{
	fn from(value:u64)->Self{
		Self{id:value,name:None}
	}
}
impl From<usize> for Label{
	fn from(value:usize)->Self{Self::from(value as u64)}
}
impl FromStr for Label{
	fn from_str(s:&str)->Result<Self,Self::Err>{
		let parsewithid:Option<Self>=(||{
			let idstop=s.find(": ").unwrap_or(s.len());
			let id=u64::from_str_radix(&s[..idstop],16).ok()?;
			let name=(idstop<s.len()).then(||Arc::from(&s[idstop+2..]));
			Some(Self{id,name})
		})();
		Ok(if let Some(l)=parsewithid{l}else{Self::from(0_u64).with_name(s)})
	}
	type Err=();
}
impl Hasher for H{
	#[inline]
	fn finish(&self)->u64{self.0}
	#[inline]
	fn write(&mut self,bytes:&[u8]){
		let H(h)=self;
		for &byte in bytes.iter(){*h=h.rotate_left(8)^byte as u64}
	}
    #[inline]
    fn write_u64(&mut self,n:u64){self.0^=n}
}
impl Label{
	/// creates a new label
	pub fn new()->Self{
		Self{id:rand::random(),name:None}
	}
	/// sets the label id
	pub fn with_id(mut self,id:u64)->Self{
		self.id=id;
		self
	}
	/// names the label
	pub fn with_name<S:AsRef<str>>(mut self,name:S)->Self{
		let name=name.as_ref();
		self.name=if name.len()==0{None}else{Some(name.into())};
		self
	}
	/// sets the label id
	pub fn with_random_id(mut self)->Self{
		self.id=rand::random();
		self
	}
}
impl<'a,C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> ConnectionEditor<'a,C,V>{
	/// gets the index, or usize::MAX to insert at the end of the order regardless of the number of connections
	pub fn get_index(&self)->usize{self.index}
	/// references the input label
	pub fn input(&self)->&Label{&self.input}
	/// adds a layer to the associated graph if there is one, returning the layer if not or the previous layer associated with the layer id
	pub fn insert_layer<L:Into<C>>(&mut self,layer:L)->Option<C>{self.graph.as_mut().and_then(|g|g.layers.insert(self.layer.clone(),layer.into()))}
	/// checks if clear
	pub fn is_clear(&self)->bool{self.clear}
	/// references the connection label
	pub fn label(&self)->&Label{&self.connection}
	/// references the layer label
	pub fn layer(&self)->&Label{&self.layer}
	/// references the input label
	pub fn output(&self)->&Label{&self.output}
	/// inserts the layer into the graph using the current layer id
	pub fn with<L:Into<C>>(mut self,layer:L)->Self{
		self.insert_layer(layer);
		self
	}
	/// sets the flag for whether the input should be cleared after use
	pub fn with_clear(mut self,clear:bool)->Self{
		self.clear=clear;
		self
	}
	/// sets the index to insert in the run order, or usize::MAX to insert at the end of the order regardless of the number of connections
	pub fn with_index(mut self,index:usize)->Self{
		self.index=index;
		self
	}
	/// sets the input label
	pub fn with_input<L:Into<Label>>(mut self,label:L)->Self{
		self.input=label.into();
		self
	}
	/// sets the connection label
	pub fn with_label<L:Into<Label>>(mut self,label:L)->Self{
		self.connection=label.into();
		self
	}
	/// sets the layer label
	pub fn with_layer<L:Into<Label>>(mut self,label:L)->Self{
		self.layer=label.into();
		self
	}
	/// sets the output label
	pub fn with_output<L:Into<Label>>(mut self,label:L)->Self{
		self.output=label.into();
		self
	}
}
impl<'a,C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge>ConnectionInfo<'a,C,V>{
	/// gets the index in the connection processing order
	pub fn get_index(&self)->usize{
		let connection=self.connection;
		self.graph.order.iter().position(|label|connection==label).expect("order should contain all connection labels")
	}
	/// gets the underlying layer associated with the layer label
	pub fn get_layer(&self)->Option<&C>{self.graph.layers.get(&self.layer)}
	/// references the input label
	pub fn input(&self)->&Label{&self.input}
	/// checks if clear
	pub fn is_clear(&self)->bool{self.clear}
	/// references the connection label
	pub fn label(&self)->&Label{self.connection}
	/// references the layer label
	pub fn layer(&self)->&Label{self.layer}
	/// references the input label
	pub fn output(&self)->&Label{self.output}
}
impl<'a,C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> Default for ConnectionEditor<'a,C,V>{
	fn default()->Self{
		Self{clear:false,connection:Label::new(),graph:None,index:usize::MAX,input:Label::new(),layer:Label::new(),output:Label::new()}
	}
}
impl<'a,C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> Drop for ConnectionEditor<'a,C,V>{
	fn drop(&mut self){
		if let Some(g)=self.graph.take(){g.add_connection(self)}
	}
}
impl<A:AI<Vec<X>,Vec<Y>>,X,Y> AI<X,Y> for Unvec<A>{
	fn forward(&self,input:X)->Y{self.0.forward(vec![input]).into_iter().next().unwrap()}
	fn forward_mut(&mut self,input:X)->Y{self.0.forward_mut(vec![input]).into_iter().next().unwrap()}
}
impl<A:Decompose> Decompose for Unvec<A>{
	fn compose(decomposition:Self::Decomposition)->Self{Self(A::compose(decomposition))}
	fn decompose(self)->Self::Decomposition{self.0.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.0.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Into<C>,C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> Extend<Graph<A>> for Graph<C>{
	fn extend<I:IntoIterator<Item=Graph<A>>>(&mut self,iter:I){iter.into_iter().for_each(|graph|self.merge(graph))}
}
impl<A:Into<C>,C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> FromIterator<Graph<A>> for Graph<C>{
	fn from_iter<I:IntoIterator<Item=Graph<A>>>(iter:I)->Self{
		let mut graph=Graph::default();
		graph.extend(iter);
		graph
	}
}
impl<A:Op<Output=Vec<Y>>,Y> Op for Unvec<A>{
	type Output=Y;
}
impl<C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> AI<Vec<V>,Vec<V>> for Graph<C>{
	fn forward(&self,mut values:Vec<V>)->Vec<V>{
		let mut map=HashMap::with_capacity_and_hasher(values.len(),H(0));
		values.reverse();
		for (_clear,input,_layer,_output) in self.order.iter().filter_map(|c|self.connections.get(c)){
			if values.len()==0{break}
			map.entry(input.clone()).or_insert_with(||values.pop().unwrap());
		}
		map=self.forward(map);
		for (_clear,_input,_layer,output) in self.order.iter().rev().filter_map(|c|self.connections.get(c)){
			if map.len()==0{break}
			if let Some(y)=map.remove(output){values.push(y)}
		}
		values.reverse();
		values
	}
	fn forward_mut(&mut self,mut values:Vec<V>)->Vec<V>{
		let mut map=HashMap::with_capacity_and_hasher(values.len(),H(0));
		values.reverse();
		for (_clear,input,_layer,_output) in self.order.iter().filter_map(|c|self.connections.get(c)){
			if values.len()==0{break}
			map.entry(input.clone()).or_insert_with(||values.pop().unwrap());
		}
		map=self.forward_mut(map);
		for (_clear,_input,_layer,output) in self.order.iter().rev().filter_map(|c|self.connections.get(c)){
			if map.len()==0{break}
			if let Some(y)=map.remove(output){values.push(y)}
		}
		values.reverse();
		values
	}
}
impl<C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge,S:BuildHasher> AI<HashMap<Label,V,S>,HashMap<Label,V,S>> for Graph<C>{
	fn forward(&self,mut map:HashMap<Label,V,S>)->HashMap<Label,V,S>{
		let (connections,order)=(&self.connections,&self.order);
		let layers=&self.layers;

		order.iter().filter_map(|c|connections.get(c)).for_each(|(clear,input,layer,output)|{
			let x=if *clear>0{map.remove(input)}else{map.get(input).cloned()}.unwrap_or_default();
			let y=if let Some(f)=layers.get(layer){f.forward(x)}else{x};
			map.entry(output.clone()).or_default().merge(y);
		});
		map
	}
	fn forward_mut(&mut self,mut map:HashMap<Label,V,S>)->HashMap<Label,V,S>{
		let (connections,order)=(&self.connections,&self.order);
		let layers=&mut self.layers;

		order.iter().filter_map(|c|connections.get(c)).for_each(|(clear,input,layer,output)|{
			let x=if *clear>0{map.remove(input)}else{map.get(input).cloned()}.unwrap_or_default();
			let y=if let Some(f)=layers.get_mut(layer){f.forward_mut(x)}else{x};
			map.entry(output.clone()).or_default().merge(y);
		});
		map
	}
}
impl<C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> Default for Graph<C>{
	fn default()->Self{Self::new()}
}
impl<C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge> Graph<C>{
	/// creates a new empty graph
	pub fn new()->Self{
		Self{connections:HashMap::with_hasher(H(0)),layers:HashMap::with_hasher(H(0)),order:Vec::new()}
	}
	/// adds a connection between two vertices
	pub fn add_connection<'a>(&mut self,config:&ConnectionEditor<'a,C,V>){
		let (connections,order)=(&mut self.connections,&mut self.order);
		let (connection,input,layer,output)=(config.connection.clone(),config.input.clone(),config.layer.clone(),config.output.clone());
		let (clear,index)=(config.clear,config.index);

		connections.insert(connection.clone(),(clear as u64,input,layer,output));
		if index<order.len(){order.insert(index,connection)}else{order.push(connection)}
	}
	/// adds a layer without connecting it
	pub fn add_layer<X:Into<C>,L:Into<Label>>(&mut self,label:L,layer:X){
		self.layers.insert(label.into(),layer.into());
	}
	/// adds a connection between vertices
	pub fn connect<I:Into<Label>,O:Into<Label>>(&mut self,input:I,output:O)->ConnectionEditor<'_,C,V>{
		let (connection,layer)=(Label::new(),Label::new());
		let (input,output)=(input.into(),output.into());
		let clear=false;
		let index=self.order.len();

		let graph=Some(self);
		ConnectionEditor{clear,connection,graph,index,input,layer,output}
	}
	/// returns an iterator over the connections in an arbitrary order
	pub fn connections<'a>(&'a self)->impl Iterator<Item=ConnectionInfo<'a,C,V>>{self.connections.keys().filter_map(move|k|self.get_connection(k))}
	/// gets connection information by label
	pub fn get_connection<'a>(&'a self,label:&Label)->Option<ConnectionInfo<'a,C,V>>{
		let (connection,(clear,input,layer,output))=self.connections.get_key_value(label)?;
		let clear=*clear>0;
		let graph=self;

		Some(ConnectionInfo{clear,connection,graph,input,layer,output})
	}
	/// gets the layer information by label
	pub fn get_layer(&self,label:&Label)->Option<&C>{self.layers.get(label)}
	/// another graphs into this one
	pub fn merge<A:Into<C>>(&mut self,graph:Graph<A>){
		self.connections.extend(graph.connections);
		self.layers.extend(graph.layers.into_iter().map(|(l,a)|(l,a.into())));
		self.order.extend(graph.order);
	}
	/// gets the connection order
	pub fn order(&self)->&[Label]{&self.order}
	/// splits the graph according to the predicate(clear, connectionlabel, inputlabel, layer, layerlabel, outputlabel). true will be sent to the returned graph. the resulting graphs will only have the layers they use
	pub fn split<F:FnMut(bool,&Label,&Label,&C,&Label,&Label)->bool>(&mut self,mut predicate:F)->Self where C:Clone{
		let (connections,layers,order)=(&mut self.connections,&mut self.layers,&mut self.order);

		let newconnections:HashMap<_,_,H>=connections.extract_if(|connectionlabel,(clear,inputlabel,layerlabel,outputlabel)|if let Some(layer)=layers.get_mut(layerlabel){
			predicate(*clear>0,connectionlabel,inputlabel,layer,layerlabel,outputlabel)
		}else{
			false
		}).collect();

		let mut oldlayers=mem::take(layers);
		let newlayers=newconnections.iter().filter_map(|(_connectionlabel,(_clear,_inputlabel,layerlabel,_outputlabel))|if let Some(layer)=oldlayers.get(layerlabel){
			Some((layerlabel.clone(),layer.clone()))
		}else{
			None
		}).collect();
		connections.iter().for_each(|(_connectionlabel,(_clear,_inputlabel,layerlabel,_outputlabel))|if let Some(layer)=oldlayers.remove(layerlabel){
			layers.insert(layerlabel.clone(),layer);
		});

		let neworder=order.extract_if(..,|label|newconnections.contains_key(label)).collect();
		order.retain(|label|connections.contains_key(label));

		Self{connections:newconnections,layers:newlayers,order:neworder}
	}
	/// topologically sorts the graph. Inputs to the same node will retain their relative order.
	pub fn sort(&mut self){
		let connections=&mut self.connections;
		let mut dedup=HashSet::with_capacity(connections.len());
		let mut nodes:HashMap<Label,(Vec<Label>,usize)>=HashMap::with_capacity(connections.len());
		let order=&mut self.order;
		order.drain(..).for_each(|label|if let Some((_clear,input,_layer,output))=connections.get(&label){
			let (_inputinputs,inputoutputs)=nodes.entry(input.clone()).or_default();
			*inputoutputs+=1;
			let (outputinputs,_outputoutputs)=nodes.entry(output.clone()).or_default();
			outputinputs.push(label);
		});

		while nodes.len()>0{
			let mut cycle=true;
			let mut n=order.len();
			nodes.retain(|_node,(inputs,outputs)|{
				if *outputs==0{
					cycle=false;
					order.extend(inputs.drain(..).filter(|i|dedup.insert(i.clone())).rev());
				}
				inputs.len()>0
			});
			if cycle{order.extend(nodes.iter_mut().filter_map(|(_node,(inputs,_outputs))|inputs.pop()).filter(|i|dedup.insert(i.clone())))}
			while n<order.len(){
				if let Some((inputs,outputs))=nodes.get_mut(&connections.get(&order[n]).unwrap().1){
					*outputs-=1;
					if inputs.len()==1&&*outputs==0{order.push(inputs.pop().unwrap())}
				}
				n+=1;
			}
		}
		order.iter().for_each(|label|if let Some((clear,input,_layer,_output))=connections.get_mut(label){*clear=set_bit(*clear,1,!nodes.contains_key(input))});
		order.reverse();
	}
}
impl<C:Decompose> Decompose for Graph<C>{// TODO ideally this would preserve unconnected layers too
	fn compose((connections,layers):Self::Decomposition)->Self{
		let mut order=Vec::with_capacity(connections.len());
		let connections=connections.into_iter().map(|decomposed|{
			let (label,(clear,input,layer,output)):(Label,(u64,Label,Label,Label))=Decompose::compose(decomposed);
			order.push(label.clone());
			(label,(clear,input,layer,output))
		}).collect();
		let layers=layers.into_iter().map(Decompose::compose).collect();
		Self{connections,layers,order}
	}
	fn decompose(self)->Self::Decomposition{
		let (mut connections,mut layers,order)=(self.connections,self.layers,self.order);

		let (mut decomposedconnections,mut decomposedlayers)=(Vec::new(),Vec::new());
		order.iter().filter_map(|label|connections.remove_entry(label)).for_each(|(label,(clear,input,layer,output))|{
			decomposedlayers.extend(layers.remove_entry(&layer).map(|(label,layer)|(label.decompose(),layer.decompose())));
			decomposedconnections.push((label.decompose(),(clear,input.decompose(),layer.decompose(),output.decompose())));
		});
		(decomposedconnections,decomposedlayers)
	}
	fn decompose_cloned(&self)->Self::Decomposition{
		let (connections,layers,order)=(&self.connections,&self.layers,&self.order);
		let mut layersfound=HashSet::new();

		let (mut decomposedconnections,mut decomposedlayers)=(Vec::new(),Vec::new());
		order.iter().filter_map(|label|connections.get_key_value(label)).for_each(|(label,(clear,input,layer,output))|{
			if layersfound.insert(layer.clone()){decomposedlayers.extend(layers.get_key_value(layer).map(|(label,layer)|(label.decompose_cloned(),layer.decompose_cloned())))}
			decomposedconnections.push((label.decompose_cloned(),(*clear,input.decompose_cloned(),layer.decompose_cloned(),output.decompose_cloned())));
		});
		(decomposedconnections,decomposedlayers)
	}
	type Decomposition=(Vec<(String,(u64,String,String,String))>,Vec<(String,C::Decomposition)>);
}
impl<C:Op> Op for Graph<C>{
	type Output=Vec<C::Output>;
}
impl<E,V:Extend<E>+IntoIterator<Item=E>> Merge for Extendable<V>{
	fn merge(&mut self,other:Self){self.0.extend(other.0)}
}
impl<E:Merge> Merge for Option<E>{
	fn merge(&mut self,other:Self){
		match (self,other){(Some(this),Some(other))=>this.merge(other),(this,Some(other))=>*this=Some(other),_=>()}
	}
}
impl<E> Merge for Vec<E>{
	fn merge(&mut self,other:Self){self.extend(other)}
}
impl<S:?Sized+AsRef<str>> From<&S> for Label{
	fn from(value:&S)->Self{Self::from_str(value.as_ref()).unwrap()}
}
#[cfg(test)]
mod tests{
	#[test]
	fn sort_digons(){
		let mut graph:Graph<Append<u64>>=Graph::new();
		let mut order=[0,1,2,3,4];

		order.shuffle(&mut rand::rng());
		for &n in order.iter(){
			graph.connect(n,n+1).with_clear(true).with(Append(n+100));
			graph.connect(n,n+1).with_clear(true).with(Append(n+200));
		}
		let unsorted:Vec<u64>=Unvec(&graph).forward(Vec::new());

		graph.sort();
		let sorted:Vec<u64>=Unvec(&graph).forward(Vec::new());
		assert_eq!(sorted,[100,200,101,201,102,202,103,203,104,204]);
		assert_ne!(sorted,unsorted);
	}
	#[test]
	fn sort_line(){
		let mut graph:Graph<Append<u64>>=Graph::new();
		let mut order=[0,1,2,3,4,5,6,7,8,9];

		order.shuffle(&mut rand::rng());
		for &n in order.iter(){
			graph.connect(n,n+1).with(Append(n));
		}
		let unsorted:Vec<u64>=Unvec(&graph).forward(Vec::new());

		graph.sort();
		let sorted:Vec<u64>=Unvec(&graph).forward(Vec::new());
		assert_eq!(sorted,[0,1,2,3,4,5,6,7,8,9]);
		assert_ne!(sorted,unsorted);
	}
	impl<E:Clone> AI<Vec<E>,Vec<E>> for Append<E>{
		fn forward(&self,mut input:Vec<E>)->Vec<E>{
			input.push(self.0.clone());
			input
		}
	}
	impl<E:Clone> Op for Append<E>{
		type Output=Vec<E>;
	}
	#[derive(Clone,Debug)]
	/// test ai module that appends a number to a vec
	struct Append<E:Clone>(E);
	use rand::seq::SliceRandom;
	use super::*;
}
#[derive(Clone,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// wraps collections so that merging concatenates them using the Extend trait
pub struct Extendable<V>(pub V);
/*#[derive(Clone,Debug)]
/// allows configuring a connection to add to the graph
pub struct ConnectionConfig{clear:bool,connection:Label,index:usize,input:Label,layer:Label,output:Label}*/
#[derive(Debug)]// TODO split connection config and connection editor
/// allows configuring a connection to add to the graph, or manipulating an existing connection
pub struct ConnectionEditor<'a,C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge>{clear:bool,connection:Label,graph:Option<&'a mut Graph<C>>,index:usize,input:Label,layer:Label,output:Label}
#[derive(Clone,Debug)]
/// information about a connection
pub struct ConnectionInfo<'a,C:AI<V,V>+Op<Output=V>,V:Clone+Default+Merge>{clear:bool,connection:&'a Label,graph:&'a Graph<C>,input:&'a Label,layer:&'a Label,output:&'a Label}
#[derive(Clone,Debug,Deserialize,Eq,PartialEq,Serialize)]
/// graph like ai operation structure
pub struct Graph<C>{connections:HashMap<Label,(u64,Label,Label,Label),H>,layers:HashMap<Label,C,H>,order:Vec<Label>}
#[derive(Clone,Debug,Deserialize,Eq,Hash,PartialEq,Serialize)]
/// label for graph connections or layers or nodes. format is id: name where id is a hex number or simply id if there is no name. name without a number will be parse as a name with a 0 id
pub struct Label{id:u64,name:Option<Arc<str>>}
#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
#[repr(transparent)]
/// wraps the graph so it can take singular io
pub struct Unvec<A>(pub A);
/// trait to allow merging multiple outputs into one graph node
pub trait Merge{
	/// merges the other into self, taking out of other if convenient
	fn merge(&mut self,other:Self);
}
#[derive(Clone,Copy,Debug,Default)]
#[repr(transparent)]
struct H(u64);
use crate::{AI,Decompose,Op};
use serde::{Deserialize,Serialize};
use std::{
	collections::{HashMap,HashSet},fmt::{Display,Formatter,UpperHex,Result as FmtResult},hash::{BuildHasher,Hasher,Hash},iter::{FromIterator,Extend},mem,str::FromStr,sync::Arc
};
