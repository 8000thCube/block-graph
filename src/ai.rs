impl Decompose for (){
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl Decompose for Range<usize>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.0..decomposition.1}
	fn decompose(self)->Self::Decomposition{(self.start,self.end)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.start,self.end)}
	type Decomposition=(usize,usize);
}
impl Decompose for usize{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl WhichDims for Range<usize>{
	fn which_dims(&self)->Self::Iter<'_>{self.clone()}
	type Iter<'a>=Self where Self:'a;
}
impl WhichDims for Vec<usize>{
	fn which_dims(&self)->Self::Iter<'_>{self.iter()}
	type Iter<'a>=SliceIter<'a,usize> where Self:'a;
}
impl WhichDims for usize{
	fn which_dims(&self)->Self::Iter<'_>{iter::once(*self)}
	type Iter<'a>=Once<usize> where Self:'a;
}
impl<A:AI<R,S>,B:AI<T,U>,C:AI<V,W>,D:AI<X,Y>,R,S,T,U,V,W,X,Y> AI<(R,T,V,X),(S,U,W,Y)> for Zip<(A,B,C,D)>{
	fn forward(&self,(r,t,v,x):(R,T,V,X))->(S,U,W,Y){
		let (a,b,c,d)=self.inner();
		(a.forward(r),b.forward(t),c.forward(v),d.forward(x))
	}
	fn forward_mut(&mut self,(r,t,v,x):(R,T,V,X))->(S,U,W,Y){
		let (a,b,c,d)=self.inner_mut();
		(a.forward_mut(r),b.forward_mut(t),c.forward_mut(v),d.forward_mut(x))
	}
}impl<A:AI<T,U>,B:AI<V,W>,C:AI<X,Y>,T,U,V,W,X,Y> AI<(T,V,X),(U,W,Y)> for Zip<(A,B,C)>{
	fn forward(&self,(t,v,x):(T,V,X))->(U,W,Y){
		let (a,b,c)=self.inner();
		(a.forward(t),b.forward(v),c.forward(x))
	}
	fn forward_mut(&mut self,(t,v,x):(T,V,X))->(U,W,Y){
		let (a,b,c)=self.inner_mut();
		(a.forward_mut(t),b.forward_mut(v),c.forward_mut(x))
	}
}impl<A:AI<V,W>+Op<Output=W>,B:AI<W,X>+Op<Output=X>,C:AI<X,Y>+Op<Output=Y>,D:AI<Y,Z>,V,W,X,Y,Z> AI<V,Z> for Sequential<(A,B,C,D)>{
	fn forward(&self,input:V)->Z{
		let (a,b,c,d)=self.inner();
		d.forward(c.forward(b.forward(a.forward(input))))
	}
	fn forward_mut(&mut self,input:V)->Z{
		let (a,b,c,d)=self.inner_mut();
		d.forward_mut(c.forward_mut(b.forward_mut(a.forward_mut(input))))
	}
}
impl<A:AI<V,W>,B:AI<X,Y>,V,W,X,Y> AI<(V,X),(W,Y)> for Zip<(A,B)>{
	fn forward(&self,(v,x):(V,X))->(W,Y){
		let (a,b)=self.inner();
		(a.forward(v),b.forward(x))
	}
	fn forward_mut(&mut self,(v,x):(V,X))->(W,Y){
		let (a,b)=self.inner_mut();
		(a.forward_mut(v),b.forward_mut(x))
	}
}
impl<A:AI<W,X>+Op<Output=X>,B:AI<X,Y>+Op<Output=Y>,C:AI<Y,Z>,W,X,Y,Z> AI<W,Z> for Sequential<(A,B,C)>{
	fn forward(&self,input:W)->Z{
		let (a,b,c)=self.inner();
		c.forward(b.forward(a.forward(input)))
	}
	fn forward_mut(&mut self,input:W)->Z{
		let (a,b,c)=self.inner_mut();
		c.forward_mut(b.forward_mut(a.forward_mut(input)))
	}
}
impl<A:AI<W,Z>+AI<X,Y>,W,X,Y,Z> AI<W,Z> for SetType<A,X,Y>{
	fn forward(&self,input:W)->Z{self.inner().forward(input)}
	fn forward_mut(&mut self,input:W)->Z{self.inner_mut().forward_mut(input)}
}
impl<A:AI<X,X>+Op<Output=X>,X> Op for Option<A>{
	type Output=X;
}
impl<A:AI<X,X>+Op<Output=X>,X> Op for Sequential<Vec<A>>{
	type Output=X;
}
impl<A:AI<X,Y>+Op<Output=Y>,I:IntoIterator<Item=X>,J:FromIterator<Y>,X,Y> AI<I,J> for ToEach<A>{
	fn forward(&self,input:I)->J{
		let a=self.inner();
		input.into_iter().map(|x|a.forward(x)).collect()
	}
	fn forward_mut(&mut self,input:I)->J{
		let a=self.inner_mut();
		input.into_iter().map(|x|a.forward_mut(x)).collect()
	}
}
impl<A:AI<X,X>,X:Clone> Iterator for Autoregression<A,X>{
	fn next(&mut self)->Option<Self::Item>{
		let output=self.state.clone();
		self.state=Some(self.ai.forward_mut(self.state.take().unwrap()));
		output
	}
	fn size_hint(&self)->(usize,Option<usize>){(usize::MAX,None)}
	type Item=X;
}
impl<A:AI<X,X>,X> AI<X,X> for Option<A>{
	fn forward(&self,x:X)->X{
		if let Some(a)=self{a.forward(x)}else{x}
	}
	fn forward_mut(&mut self,x:X)->X{
		if let Some(a)=self{a.forward_mut(x)}else{x}
	}
}
impl<A:AI<X,X>,X> AI<X,X> for Sequential<Vec<A>>{
	fn forward(&self,input:X)->X{self.inner().iter().fold(input,|x,a|a.forward(x))}
	fn forward_mut(&mut self,input:X)->X{self.inner_mut().iter_mut().fold(input,|x,a|a.forward_mut(x))}
}
impl<A:AI<X,Y>+Decompose,X,Y> Decompose for SetType<A,X,Y>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition),phantom:PhantomData}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:AI<X,Y>+Op<Output=Y>,B:AI<Y,Z>,X,Y,Z> AI<X,Z> for Sequential<(A,B)>{
	fn forward(&self,input:X)->Z{
		let (a,b)=self.inner();
		b.forward(a.forward(input))
	}
	fn forward_mut(&mut self,input:X)->Z{
		let (a,b)=self.inner_mut();
		b.forward_mut(a.forward_mut(input))
	}
}
impl<A:AI<X,Y>,X:Clone,Y> AI<X,Vec<Y>> for Branch<Vec<A>>{
	fn forward(&self,input:X)->Vec<Y>{
		let a=self.inner();
		let mut y:Vec<Y>=a.iter().take(a.len().saturating_sub(1)).map(|a|a.forward(input.clone())).collect();
		if let Some(a)=a.last(){y.push(a.forward(input))}
		y
	}
	fn forward_mut(&mut self,input:X)->Vec<Y>{
		let a=self.inner_mut();
		let l=a.len().saturating_sub(1);
		let mut y:Vec<Y>=a.iter_mut().take(l).map(|a|a.forward_mut(input.clone())).collect();
		if let Some(a)=a.last_mut(){y.push(a.forward_mut(input))}
		y
	}
}
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y){
		let y=self.inner().forward(input);
		(y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y){
		let y=self.inner_mut().forward_mut(input);
		(y.clone(),y)
	}
}
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y,Y){
		let y=self.inner().forward(input);
		(y.clone(),y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y,Y){
		let y=self.inner_mut().forward_mut(input);
		(y.clone(),y.clone(),y)
	}
}
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y,Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y,Y,Y){
		let y=self.inner().forward(input);
		(y.clone(),y.clone(),y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y,Y,Y){
		let y=self.inner_mut().forward_mut(input);
		(y.clone(),y.clone(),y.clone(),y)
	}
}
impl<A:AI<X,Y>,X,Y> AI<X,Y> for &A{
	fn forward(&self,input:X)->Y{(**self).forward(input)}
}
impl<A:AI<X,Y>,X,Y> AI<X,Y> for &mut A{
	fn forward(&self,input:X)->Y{(**self).forward(input)}
	fn forward_mut(&mut self,input:X)->Y{(**self).forward_mut(input)}
}
impl<A:AI<X,Y>,X,Y> Op for SetType<A,X,Y>{
	type Output=Y;
}
impl<A:Decompose,B:Decompose,C:Decompose,D:Decompose> Decompose for (A,B,C,D){
	fn compose(decomposition:Self::Decomposition)->Self{(A::compose(decomposition.0),B::compose(decomposition.1),C::compose(decomposition.2),D::compose(decomposition.3))}
	fn decompose(self)->Self::Decomposition{(self.0.decompose(),self.1.decompose(),self.2.decompose(),self.3.decompose())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.0.decompose_cloned(),self.1.decompose_cloned(),self.2.decompose_cloned(),self.3.decompose_cloned())}
	type Decomposition=(A::Decomposition,B::Decomposition,C::Decomposition,D::Decomposition);
}
impl<A:Decompose,B:Decompose,C:Decompose> Decompose for (A,B,C){
	fn compose(decomposition:Self::Decomposition)->Self{(A::compose(decomposition.0),B::compose(decomposition.1),C::compose(decomposition.2))}
	fn decompose(self)->Self::Decomposition{(self.0.decompose(),self.1.decompose(),self.2.decompose())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.0.decompose_cloned(),self.1.decompose_cloned(),self.2.decompose_cloned())}
	type Decomposition=(A::Decomposition,B::Decomposition,C::Decomposition);
}
impl<A:Decompose,B:Decompose> Decompose for (A,B){
	fn compose(decomposition:Self::Decomposition)->Self{(A::compose(decomposition.0),B::compose(decomposition.1))}
	fn decompose(self)->Self::Decomposition{(self.0.decompose(),self.1.decompose())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.0.decompose_cloned(),self.1.decompose_cloned())}
	type Decomposition=(A::Decomposition,B::Decomposition);
}
impl<A:Decompose,X:Decompose> Decompose for Autoregression<A,X>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{ai:A::compose(decomposition.0),state:Some(X::compose(decomposition.1))}
	}
	fn decompose(self)->Self::Decomposition{(self.ai.decompose(),self.state.unwrap().decompose())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.ai.decompose_cloned(),self.state.as_ref().unwrap().decompose_cloned())}
	type Decomposition=(A::Decomposition,X::Decomposition);
}
impl<A:Decompose> Decompose for AccQ<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition.0),gamma:decomposition.1}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.gamma)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.gamma)}
	type Decomposition=(A::Decomposition,f32);
}
impl<A:Decompose> Decompose for Branch<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for MSE<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Cat<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition.0),dim:decomposition.1}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.dim)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.dim)}
	type Decomposition=(A::Decomposition,usize);
}
impl<A:Decompose> Decompose for Duplicate<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Option<A>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.map(A::compose)}
	fn decompose(self)->Self::Decomposition{self.map(A::decompose)}
	fn decompose_cloned(&self)->Self::Decomposition{self.as_ref().map(A::decompose_cloned)}
	type Decomposition=Option<A::Decomposition>;
}
impl<A:Decompose> Decompose for Sequential<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for SoftChoose<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition.0),temperature:decomposition.1}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.temperature)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.temperature)}
	type Decomposition=(A::Decomposition,f32);
}
impl<A:Decompose> Decompose for ToEach<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for TruncateToMatch<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{alignment:decomposition.1,inner:A::compose(decomposition.0)}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.alignment)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.alignment)}
	type Decomposition=(A::Decomposition,usize);
}
impl<A:Decompose> Decompose for Vec<A>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.into_iter().map(A::compose).collect()}
	fn decompose(self)->Self::Decomposition{self.into_iter().map(A::decompose).collect()}
	fn decompose_cloned(&self)->Self::Decomposition{self.iter().map(A::decompose_cloned).collect()}
	type Decomposition=Vec<A::Decomposition>;
}
impl<A:Decompose> Decompose for Zip<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Op<Output=W>,B:AI<W,X>+Op<Output=X>,C:AI<X,Y>+Op<Output=Y>,D:AI<Y,Z>+Op<Output=Z>,W,X,Y,Z> Op for Sequential<(A,B,C,D)>{
	type Output=Z;
}
impl<A:Op<Output=W>,B:Op<Output=X>,C:Op<Output=Y>,D:Op<Output=Z>,W,X,Y,Z> Op for Zip<(A,B,C,D)>{
	type Output=(W,X,Y,Z);
}
impl<A:Op<Output=X>,B:AI<X,Y>+Op<Output=Y>,C:AI<Y,Z>+Op<Output=Z>,X,Y,Z> Op for Sequential<(A,B,C)>{
	type Output=Z;
}
impl<A:Op<Output=X>,B:Op<Output=Y>,C:Op<Output=Z>,X,Y,Z> Op for Zip<(A,B,C)>{
	type Output=(X,Y,Z);
}
impl<A:Op<Output=X>,B:Op<Output=Y>,X,Y> Op for Zip<(A,B)>{
	type Output=(X,Y);
}
impl<A:Op<Output=Y>,B:AI<Y,Z>+Op<Output=Z>,Y,Z> Op for Sequential<(A,B)>{
	type Output=Z;
}
impl<A:Op<Output=Y>,Y> Op for Branch<Vec<A>>{
	type Output=Vec<Y>;
}
impl<A:Op<Output=Y>,Y> Op for Duplicate<A>{
	type Output=(Y,Y);
}
impl<A:Op<Output=Y>,Y> Op for ToEach<A>{
	type Output=Vec<Y>;
}
impl<A:Op> Op for &A{
	type Output=A::Output;
}
impl<A:Op> Op for &mut A{
	type Output=A::Output;
}
impl<A:Op<Output=Y>,Y> Op for AccQ<A> where AccQ<()>:AI<Y,Y>{
	type Output=Y;
}
impl<A:Op<Output=Y>,I,Y:IntoIterator<Item=I>> Op for Cat<A> where Cat<()>:AI<I,I>{
	type Output=I;
}
impl<A:Op<Output=Y>,Y> Op for MSE<A> where MSE<()>:AI<Y,f32>{
	type Output=f32;
}
impl<A:Op<Output=Y>,Y> Op for SoftChoose<A> where SoftChoose<()>:AI<Y,u32>{
	type Output=u32;
}
impl<A:Op<Output=Y>,Y> Op for TruncateToMatch<A> where TruncateToMatch<()>:AI<Y,Y>{
	type Output=Y;
}
impl<A,B> Op for (A,B){
	type Output=();
}
impl<A,B,C> Op for (A,B,C){
	type Output=();
}
impl<A,B,C,D> Op for (A,B,C,D){
	type Output=();
}
impl<A> Op for Vec<A>{
	type Output=();
}
impl<A> TruncateToMatch<A>{
	/// references the inner value
	pub fn inner(&self)->&A{&self.inner}
	/// references the inner value
	pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
	/// returns the inner value
	pub fn into_inner(self)->A{self.inner}
}
impl<X> AI<X,X> for (){
	fn forward(&self,input:X)->X{input}
}
/// creates accessor functions for the inner value
macro_rules! accessible_inner{
	($field:ident:$type:ident)=>(
		/// references the inner value
		pub fn inner(&self)->&$type{&self.$field}
		/// references the inner value
		pub fn inner_mut(&mut self)->&mut $type{&mut self.$field}
		/// returns the inner value
		pub fn into_inner(self)->$type{self.$field}
	);
}

impl<A> AccQ<A>{
	accessible_inner!(inner:A);
}
impl<A> Branch<A>{
	accessible_inner!(inner:A);
}
impl<A> Cat<A>{
	accessible_inner!(inner:A);
}
impl<A> Duplicate<A>{
	accessible_inner!(inner:A);
}
impl<A> MSE<A>{
	accessible_inner!(inner:A);
}
impl<A> Sequential<A>{
	accessible_inner!(inner:A);
}
impl<A:AI<X,Y>,X,Y> SetType<A,X,Y>{
	accessible_inner!(inner:A);
}
impl<A> SoftChoose<A>{
	accessible_inner!(inner:A);
}
impl<A> ToEach<A>{
	accessible_inner!(inner:A);
}
impl<A> Zip<A>{
	accessible_inner!(inner:A);
}


#[derive(Clone,Copy,Debug,Default,PartialEq)]
/// accumulates cumulative
pub struct AccQ<A>{gamma:f32,inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// autoregressive inference
pub struct Autoregression<A,X>{ai:A,state:Option<X>}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wrapper for applying ai modules to the same input
pub struct Branch<A>{inner:A}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wrapper for applying mean squared error loss
pub struct MSE<A>{inner:A}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wrapper for concatenating tensors in the output
pub struct Cat<A>{dim:usize,inner:A}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// module for cloning things
pub struct Duplicate<A>{inner:A}//TODO replicate that has a number and makes a vec
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wrapper for applying ai modules sequentially
pub struct Sequential<A>{inner:A}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// fixes the output type of a layer for a particular input type.
pub struct SetType<A:AI<X,Y>,X,Y>{inner:A,phantom:PhantomData<fn(X)->Y>}
#[derive(Clone,Copy,Debug,Default)]
/// chooses from the softmax
pub struct SoftChoose<A>{inner:A,temperature:f32}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wraps to apply to every element of a vector
pub struct ToEach<A>{inner:A}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// truncates each tensor dimension in the list to the minimum so that they match
pub struct TruncateToMatch<A>{alignment:usize,inner:A}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wraps to apply each function
pub struct Zip<A>{inner:A}
/// general ai trait
pub trait AI<X,Y>{
	/// applies to the input
	fn forward(&self,input:X)->Y;
	/// applies to the input, possibly updating internal caches
	fn forward_mut(&mut self,input:X)->Y{self.forward(input)}
}
/// trait to decompose AI modules into components that implement other libraries' traits
pub trait Decompose{
	/// recreates from the decomposition
	fn compose(decomposition:Self::Decomposition)->Self where Self:Sized;
	/// owned decomposition
	fn decompose(self)->Self::Decomposition where Self:Sized;
	/// decomposition that copies data
	fn decompose_cloned(&self)->Self::Decomposition;
	/// the decomposed type
	type Decomposition;
}
/// composition trait
pub trait Op{
	/// wraps with a accq operation
	fn acc_q(self,gamma:f32)->AccQ<Self> where AccQ<Self>:Op,Self:Sized{
		AccQ{inner:self,gamma}
	}
	/// wraps with a branch operation
	fn branch(self)->Branch<Self> where Branch<Self>:Op,Self:Sized{
		Branch{inner:self}
	}
	/// wraps with a cat operation
	fn cat(self,dim:usize)->Cat<Self> where Cat<Self>:Op,Self:Sized{
		Cat{inner:self,dim}
	}
	/// wraps with a duplicate operation
	fn duplicate(self)->Duplicate<Self> where Duplicate<Self>:Op,Self:Sized{
		Duplicate{inner:self}
	}
	/// set type but with the same input and output
	fn fix_type<Z>(self)->SetType<Self,Z,Z> where Self:AI<Z,Z>+Sized{self.set_type()}
	/// creates an autoregressive inference
	fn infer_autoregressive<X,Y>(self,input:X)->Autoregression<Self,Y> where Self:AI<X,Y>+AI<Y,Y>+Sized,Y:Clone{
		let mut ai=self;
		let state=Some(ai.forward_mut(input));
		Autoregression{ai,state}
	}
	/// wraps with a mse operation
	fn mse(self)->MSE<Self> where MSE<Self>:Op,Self:Sized{
		MSE{inner:self}
	}
	/// creates an optional operation
	fn optional(self)->Option<Self> where Self:Sized{Some(self)}
	/// produces a sequential module
	fn sequential(self)->Sequential<Self> where Sequential<Self>:Op,Self:Sized{
		Sequential{inner:self}
	}
	/// sets the input output types
	fn set_type<W,Z>(self)->SetType<Self,W,Z> where Self:AI<W,Z>+Sized{
		SetType{inner:self,phantom:PhantomData}
	}
	/// wraps with a choose operation
	fn soft_choose(self,temperature:f32)->SoftChoose<Self> where Self:Sized,SoftChoose<Self>:Op{
		SoftChoose{inner:self,temperature}
	}
	/// wraps with a truncate to match operation. alignment=0 for left alignment. will have other alignment settings in the future
	fn truncate_to_match(self,alignment:usize)->TruncateToMatch<Self> where Self:Sized,TruncateToMatch<Self>:Op{// TODO center/left/right alignment
		assert!(alignment==0,"non left alignment not yet supported");
		let _todo=alignment;
		TruncateToMatch{alignment:0,inner:self}
	}
	/// produces a zip module
	fn zip(self)->Zip<Self> where Self:Sized,Zip<Self>:Op{
		Zip{inner:self}
	}
	/// suggested output type to help with composition coherence. Ideally, Self should implement AI<X,Self::Output> for some X
	type Output;
}
/// tells which dims to apply an operation
pub trait WhichDims{
	/// iterates over the dims
	fn which_dims(&self)->Self::Iter<'_>;
	/// the type of dimension iterator
	type Iter<'a> where Self:'a;
}
use std::{
	iter::{FromIterator,Once,self},marker::PhantomData,ops::Range,slice::Iter as SliceIter
};
