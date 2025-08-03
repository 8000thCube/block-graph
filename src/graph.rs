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

		order.iter().filter_map(|c|connections.get(c)).for_each(|(clear,input,layer,output)|if let Some(f)=layers.get(layer){
			let x=if *clear>0{map.remove(input)}else{map.get(input).cloned()}.unwrap_or_default();
			let y=f.forward(x);
			map.entry(output.clone()).or_default().merge(y);
		});
		map
	}
	fn forward_mut(&mut self,mut map:HashMap<Label,V,S>)->HashMap<Label,V,S>{
		let (connections,order)=(&self.connections,&self.order);
		let layers=&mut self.layers;

		order.iter().filter_map(|c|connections.get(c)).for_each(|(clear,input,layer,output)|if let Some(f)=layers.get_mut(layer){
			//dbg!(input);
			let x=if *clear>0{map.remove(input)}else{map.get(input).cloned()}.unwrap_or_default();
			let y=f.forward_mut(x);
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
	/// adds a connection between two vertices reusing a layer
	pub fn add_connection<X:Into<Label>,I:Into<Label>,L:Into<Label>,O:Into<Label>>(&mut self,clear:bool,connection:X,input:I,layer:L,output:O){
		let (connections,order)=(&mut self.connections,&mut self.order);
		let (connection,input,layer,output)=(connection.into(),input.into(),layer.into(),output.into());

		connections.insert(connection.clone(),(clear as u64,input,layer,output));
		order.push(connection);
	}
	/// adds a layer without connecting it
	pub fn add_layer<X:Into<C>,L:Into<Label>>(&mut self,label:L,layer:X){
		self.layers.insert(label.into(),layer.into());
	}
	/// adds a connection between vertices, returning the connection and layer indices//TODO with clear or somethign because clear is only going to be needed if someone knows better than the sort
	pub fn connect<I:Into<Label>,L:Into<C>,O:Into<Label>>(&mut self,clear:bool,input:I,layer:L,output:O)->(Label,Label){// TODO more helpful return types with chain opportunities. possibly include clear and labels there
		let (connectionlabel,layerlabel)=(Label::new(),Label::new());
		self.add_connection(clear,connectionlabel.clone(),input,layerlabel.clone(),output);
		self.add_layer(layerlabel.clone(),layer);
		(connectionlabel,layerlabel)
	}
	/// gets connection information by label. (flags, input, layer, output)
	pub fn get_connection(&self,label:&Label)->Option<(bool,&Label,&Label,&Label)>{
		let (clear,input,layer,output)=self.connections.get(label)?;
		Some((*clear>0,input,layer,output))
	}
	/// gets the layer information by label
	pub fn get_layer(&self,label:&Label)->Option<&C>{self.layers.get(label)}
	/// another graphs into this one
	pub fn merge<A:Into<C>>(&mut self,graph:Graph<A>){
		self.connections.extend(graph.connections);
		self.layers.extend(graph.layers.into_iter().map(|(l,a)|(l,a.into())));
		self.order.extend(graph.order);
	}
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
	/// topologically sorts the graph. Inputs to the same node will retain their relative order. // TODO a output splitter might be helpful if output order must be maintained somewhere
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
		order.iter().for_each(|label|if let Some((clear,input,_layer,_output))=connections.get_mut(label){*clear=set_bit(*clear,1,nodes.insert(input.clone(),(Vec::new(),0)).is_none())});
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
			graph.connect(true,n,Append(n+100),n+1);
			graph.connect(true,n,Append(n+200),n+1);
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
			graph.connect(false,n,Append(n),n+1);
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
#[derive(Clone,Debug,Eq,PartialEq)]
/// graph like ai operation structure
pub struct Graph<C>{connections:HashMap<Label,(u64,Label,Label,Label),H>,layers:HashMap<Label,C,H>,order:Vec<Label>}
#[derive(Clone,Debug,Eq,Hash,PartialEq)]
/// label for graph connections or layers or nodes. format is id: name where id is a hex number or simply id if there is no name. name without a number will be parse as a name with a 0 id
pub struct Label{id:u64,name:Option<Arc<str>>}
#[derive(Clone,Copy,Debug,Default)]
/// wraps the graph so it can take singular io
pub struct Unvec<A>(pub A);
/// trait to allow merging multiple outputs into one graph node
pub trait Merge{// TODO wrapper to implement in terms of intoiterator and from iterator might be useful
	/// merges the other into self, taking out of other if convenient
	fn merge(&mut self,other:Self);
}
#[derive(Clone,Copy,Debug,Default)]
struct H(u64);
use crate::{AI,Decompose,Op};
use std::{
	collections::{HashMap,HashSet},fmt::{Display,Formatter,UpperHex,Result as FmtResult},hash::{BuildHasher,Hasher,Hash},iter::{FromIterator,Extend},mem,str::FromStr,sync::Arc
};
