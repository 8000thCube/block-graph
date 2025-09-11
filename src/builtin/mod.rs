impl Decompose for Alignment{
	fn compose(decomposition:Self::Decomposition)->Self{
		match decomposition{0=>Self::Center,1=>Self::Left,2=>Self::Right,_=>panic!("unknown alignment number")}
	}
	fn decompose(self)->Self::Decomposition{
		match self{Self::Center=>0,Self::Left=>1,Self::Right=>2}
	}
	fn decompose_cloned(&self)->Self::Decomposition{
		match self{Self::Center=>0,Self::Left=>1,Self::Right=>2}
	}
	type Decomposition=usize;
}
impl Decompose for OnMismatch{// TODO with explicit u64 decomposition this could fit a f32 in the pad
	fn compose(decomposition:Self::Decomposition)->Self{
		match decomposition%10{0=>Self::Error,1=>Self::Pad(Alignment::compose(decomposition/10)),2=>Self::Truncate(Alignment::compose(decomposition/10)),_=>panic!("unknown mismatch number")}
	}
	fn decompose(self)->Self::Decomposition{
		match self{Self::Error=>0,Self::Pad(a)=>a.decompose()*10+1,Self::Truncate(a)=>a.decompose()*10+2}
	}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone().decompose()}
	type Decomposition=usize;
}
impl Decompose for ReductionMode{
	fn compose(decomposition:u64)->Self{
		match decomposition>>32{0=>Self::Component,1=>Self::Dim(decomposition as i32),2=>Self::Tensor,_=>panic!("unknown reduction mode number")}
	}
	fn decompose(self)->Self::Decomposition{
		match self{Self::Component=>0,Self::Dim(x)=>(1_u64<<32)|(x as u32 as u64),Self::Tensor=>2<<32}
	}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone().decompose()}
	type Decomposition=u64;
}
impl Default for Alignment{
	fn default()->Self{Self::Left}
}
impl Default for OnMismatch{
	fn default()->Self{Self::Error}
}
impl Default for ReductionMode{
	fn default()->Self{Self::Component}
}
impl<M:AI<M::Output,M::Output>+Op> Sequential<Vec<M>>{
	/// appends the module to the sequence, then returns the sequence
	pub fn with_next<A:Into<M>>(mut self,m:A)->Self{
		self.inner_mut().push(m.into());
		self
	}
}
impl<A:AI<R,S>+Op<Output=S>,B:AI<S,T>+Op<Output=T>,C:AI<T,U>+Op<Output=U>,D:AI<U,V>+Op<Output=V>,E:AI<V,W>+Op<Output=W>,F:AI<W,X>+Op<Output=X>,G:AI<X,Y>+Op<Output=Y>,H:AI<Y,Z>,R,S,T,U,V,W,X,Y,Z> AI<R,Z> for Sequential<(A,B,C,D,E,F,G,H)>{
	fn forward(&self,input:R)->Z{
		let (a,b,c,d,e,f,g,h)=self.inner();
		h.forward(g.forward(f.forward(e.forward(d.forward(c.forward(b.forward(a.forward(input))))))))
	}
	fn forward_mut(&mut self,input:R)->Z{
		let (a,b,c,d,e,f,g,h)=self.inner_mut();
		h.forward(g.forward_mut(f.forward_mut(e.forward_mut(d.forward_mut(c.forward_mut(b.forward_mut(a.forward_mut(input))))))))
	}
}
impl<A:AI<S,T>+Op<Output=T>,B:AI<T,U>+Op<Output=U>,C:AI<U,V>+Op<Output=V>,D:AI<V,W>+Op<Output=W>,E:AI<W,X>+Op<Output=X>,F:AI<X,Y>+Op<Output=Y>,G:AI<Y,Z>,S,T,U,V,W,X,Y,Z> AI<S,Z> for Sequential<(A,B,C,D,E,F,G)>{
	fn forward(&self,input:S)->Z{
		let (a,b,c,d,e,f,g)=self.inner();
		g.forward(f.forward(e.forward(d.forward(c.forward(b.forward(a.forward(input)))))))
	}
	fn forward_mut(&mut self,input:S)->Z{
		let (a,b,c,d,e,f,g)=self.inner_mut();
		g.forward_mut(f.forward_mut(e.forward_mut(d.forward_mut(c.forward_mut(b.forward_mut(a.forward_mut(input)))))))
	}
}
impl<A:AI<T,U>+Op<Output=U>,B:AI<U,V>+Op<Output=V>,C:AI<V,W>+Op<Output=W>,D:AI<W,X>+Op<Output=X>,E:AI<X,Y>+Op<Output=Y>,F:AI<Y,Z>,T,U,V,W,X,Y,Z> AI<T,Z> for Sequential<(A,B,C,D,E,F)>{
	fn forward(&self,input:T)->Z{
		let (a,b,c,d,e,f)=self.inner();
		f.forward(e.forward(d.forward(c.forward(b.forward(a.forward(input))))))
	}
	fn forward_mut(&mut self,input:T)->Z{
		let (a,b,c,d,e,f)=self.inner_mut();
		f.forward_mut(e.forward_mut(d.forward_mut(c.forward_mut(b.forward_mut(a.forward_mut(input))))))
	}
}
impl<A:AI<U,V>+Op<Output=V>,B:AI<V,W>+Op<Output=W>,C:AI<W,X>+Op<Output=X>,D:AI<X,Y>+Op<Output=Y>,E:AI<Y,Z>,U,V,W,X,Y,Z> AI<U,Z> for Sequential<(A,B,C,D,E)>{
	fn forward(&self,input:U)->Z{
		let (a,b,c,d,e)=self.inner();
		e.forward(d.forward(c.forward(b.forward(a.forward(input)))))
	}
	fn forward_mut(&mut self,input:U)->Z{
		let (a,b,c,d,e)=self.inner_mut();
		e.forward_mut(d.forward_mut(c.forward_mut(b.forward_mut(a.forward_mut(input)))))
	}
}
impl<A:AI<V,W>+Op<Output=W>,B:AI<W,X>+Op<Output=X>,C:AI<X,Y>+Op<Output=Y>,D:AI<Y,Z>,V,W,X,Y,Z> AI<V,Z> for Sequential<(A,B,C,D)>{
	fn forward(&self,input:V)->Z{
		let (a,b,c,d)=self.inner();
		d.forward(c.forward(b.forward(a.forward(input))))
	}
	fn forward_mut(&mut self,input:V)->Z{
		let (a,b,c,d)=self.inner_mut();
		d.forward_mut(c.forward_mut(b.forward_mut(a.forward_mut(input))))
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
impl<A:AI<X,X>+Op<Output=X>,X> Op for Sequential<&[A]>{
	type Output=X;
}
impl<A:AI<X,X>+Op<Output=X>,X> Op for Sequential<&mut [A]>{
	type Output=X;
}
impl<A:AI<X,X>+Op<Output=X>,X> Op for Sequential<Vec<A>>{
	type Output=X;
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
impl<A:AI<X,X>,X> AI<X,X> for Sequential<&[A]>{
	fn forward(&self,input:X)->X{self.inner().iter().fold(input,|x,a|a.forward(x))}
}
impl<A:AI<X,X>,X> AI<X,X> for Sequential<&mut [A]>{
	fn forward(&self,input:X)->X{self.inner().iter().fold(input,|x,a|a.forward(x))}
	fn forward_mut(&mut self,input:X)->X{self.inner_mut().iter_mut().fold(input,|x,a|a.forward_mut(x))}
}
impl<A:AI<X,X>,X> AI<X,X> for Sequential<Vec<A>>{
	fn forward(&self,input:X)->X{self.inner().iter().fold(input,|x,a|a.forward(x))}
	fn forward_mut(&mut self,input:X)->X{self.inner_mut().iter_mut().fold(input,|x,a|a.forward_mut(x))}
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
impl<A:AI<X,Y>+Op<Output=Y>,I:IntoIterator<Item=X>,J:FromIterator<Y>,X,Y> AI<I,J> for Map<A>{
	fn forward(&self,input:I)->J{
		let a=self.inner();
		input.into_iter().map(|x|a.forward(x)).collect()
	}
	fn forward_mut(&mut self,input:I)->J{
		let a=self.inner_mut();
		input.into_iter().map(|x|a.forward_mut(x)).collect()
	}
}
impl<A:AI<X,Y>,X,Y:Clone,const N:usize> AI<X,[Y;N]> for Duplicate<A>{
	fn forward(&self,input:X)->[Y;N]{
		let y=self.inner().forward(input);
		[0;N].map(|_|y.clone())
	}
	fn forward_mut(&mut self,input:X)->[Y;N]{
		let y=self.inner_mut().forward_mut(input);
		[0;N].map(|_|y.clone())
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
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y,Y,Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y,Y,Y,Y){
		let y=self.inner().forward(input);
		(y.clone(),y.clone(),y.clone(),y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y,Y,Y,Y){
		let y=self.inner_mut().forward_mut(input);
		(y.clone(),y.clone(),y.clone(),y.clone(),y)
	}
}
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y,Y,Y,Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y,Y,Y,Y,Y){
		let y=self.inner().forward(input);
		(y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y,Y,Y,Y,Y){
		let y=self.inner_mut().forward_mut(input);
		(y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y)
	}
}
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y,Y,Y,Y,Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y,Y,Y,Y,Y,Y){
		let y=self.inner().forward(input);
		(y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y,Y,Y,Y,Y,Y){
		let y=self.inner_mut().forward_mut(input);
		(y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y)
	}
}
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y,Y,Y,Y,Y,Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y,Y,Y,Y,Y,Y,Y){
		let y=self.inner().forward(input);
		(y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y,Y,Y,Y,Y,Y,Y){
		let y=self.inner_mut().forward_mut(input);
		(y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y.clone(),y)
	}
}
impl<A:AI<X,Y>,X,Y> Op for SetType<A,X,Y>{
	type Output=Y;
}
impl<A:AI<X,Y>,X,Y> SetType<A,X,Y>{
	accessible_inner!(inner:A);
	pub fn new(inner:A)->Self{
		Self{inner,phantom:PhantomData}
	}
}
impl<A:Decompose,X:Decompose> Decompose for Autoregression<A,X>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{ai:A::compose(decomposition.0),state:Some(X::compose(decomposition.1))}
	}
	fn decompose(self)->Self::Decomposition{(self.ai.decompose(),self.state.unwrap().decompose())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.ai.decompose_cloned(),self.state.as_ref().unwrap().decompose_cloned())}
	type Decomposition=(A::Decomposition,X::Decomposition);
}
impl<A:Decompose,X,Y> Decompose for SetType<A,X,Y>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition),phantom:PhantomData}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Duplicate<A>{
	fn compose((inner,times):Self::Decomposition)->Self{
		Self{inner:A::compose(inner),times}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.times)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.times)}
	type Decomposition=(A::Decomposition,usize);
}
impl<A:Decompose> Decompose for Sequential<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Map<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Residual<A>{
	fn compose((decomposition,apply):Self::Decomposition)->Self{
		Self{apply,inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.apply)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.apply)}
	type Decomposition=(A::Decomposition,bool);
}
impl<A:Op<Output=Y>,Y> Op for Residual<A>{
	type Output=Y;
}
impl<A:Decompose> Decompose for Zip<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for Duplicate<A> where Duplicate<M>:Into<M>{
	fn into_sequence(self)->Sequential<Vec<M>>{
		let mut s=self.inner.into_sequence();
		if let Some(l)=s.inner_mut().pop(){
			s.inner_mut().push(Duplicate{inner:l,times:self.times}.into())
		}
		s
	}
}
impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for Map<A> where Map<M>:Into<M>{
	fn into_sequence(self)->Sequential<Vec<M>>{
		Sequential::new(self.inner.into_sequence().into_inner().into_iter().map(|inner|Map{inner}.into()).collect())
	}
}
impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for Sequential<Vec<A>>{//TODO into sequence for tuple
	fn into_sequence(self)->Sequential<Vec<M>>{
		Sequential{inner:self.into_inner().into_iter().flat_map(|a|a.into_sequence().into_inner()).collect()}
	}
}
impl<A:Op<Output=S>,B:AI<S,T>+Op<Output=T>,C:AI<T,U>+Op<Output=U>,D:AI<U,V>+Op<Output=V>,E:AI<V,W>+Op<Output=W>,F:AI<W,X>+Op<Output=X>,G:AI<Y,Z>+Op<Output=Y>,H:AI<Y,Z>+Op<Output=Z>,S,T,U,V,W,X,Y,Z> Op for Sequential<(A,B,C,D,E,F,G,H)>{
	type Output=Z;
}
impl<A:Op<Output=T>,B:AI<T,U>+Op<Output=U>,C:AI<U,V>+Op<Output=V>,D:AI<V,W>+Op<Output=W>,E:AI<W,X>+Op<Output=X>,F:AI<Y,Z>+Op<Output=Y>,G:AI<Y,Z>+Op<Output=Z>,T,U,V,W,X,Y,Z> Op for Sequential<(A,B,C,D,E,F,G)>{
	type Output=Z;
}
impl<A:Op<Output=U>,B:AI<U,V>+Op<Output=V>,C:AI<V,W>+Op<Output=W>,D:AI<W,X>+Op<Output=X>,E:AI<Y,Z>+Op<Output=Y>,F:AI<Y,Z>+Op<Output=Z>,U,V,W,X,Y,Z> Op for Sequential<(A,B,C,D,E,F)>{
	type Output=Z;
}
impl<A:Op<Output=V>,B:AI<V,W>+Op<Output=W>,C:AI<W,X>+Op<Output=X>,D:AI<Y,Z>+Op<Output=Y>,E:AI<Y,Z>+Op<Output=Z>,V,W,X,Y,Z> Op for Sequential<(A,B,C,D,E)>{
	type Output=Z;
}
impl<A:Op<Output=W>,B:AI<W,X>+Op<Output=X>,C:AI<Y,Z>+Op<Output=Y>,D:AI<Y,Z>+Op<Output=Z>,W,X,Y,Z> Op for Sequential<(A,B,C,D)>{
	type Output=Z;
}
impl<A:Op<Output=X>,B:AI<Y,Z>+Op<Output=Y>,C:AI<Y,Z>+Op<Output=Z>,X,Y,Z> Op for Sequential<(A,B,C)>{
	type Output=Z;
}
impl<A:Op<Output=Y>,B:AI<Y,Z>+Op<Output=Z>,Y,Z> Op for Sequential<(A,B)>{
	type Output=Z;
}
impl<A:Op<Output=Y>,Y> Op for Duplicate<A>{
	type Output=(Y,Y);
}
impl<A:Op<Output=Y>,Y> Op for Map<A>{
	type Output=Vec<Y>;
}
impl<A:UnwrapInner> UnwrapInner for Duplicate<A>{
	fn unwrap_inner(self)->Self::Inner{self.inner.unwrap_inner()}
	type Inner=A::Inner;
}
impl<A:UnwrapInner> UnwrapInner for Map<A>{
	fn unwrap_inner(self)->Self::Inner{self.inner.unwrap_inner()}
	type Inner=A::Inner;
}
impl<A,X> Autoregression<A,X>{
	accessible_inner!(ai:A);
	pub fn new<W>(mut ai:A,input:W)->Self where A:AI<W,X>+AI<X,X>,X:Clone{
		let state=Some(ai.forward_mut(input));
		Self{ai,state}
	}
}
impl<A> Duplicate<A>{
	accessible_inner!(inner:A);
	/// creates a new duplicate module from the inner value
	pub fn from_inner(inner:A)->Self{
		Duplicate{inner,times:2}
	}
	/// creates a new duplicate module from the inner value
	pub fn new(inner:A)->Self{
		Duplicate{inner,times:2}
	}
	/// returns the suggested number of times to duplicate
	pub fn times(&self)->usize{self.times}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->Duplicate<B>{Duplicate::from_inner(inner).with_times(self.times)}
	/// sets the suggested number of times to duplicate for a variable sized output like a vec. fixed array outputs of other lengths will still be allowed
	pub fn with_times(mut self,times:usize)->Self{
		self.times=times;
		self
	}
}
impl<A> Sequential<A>{
	accessible_inner!(inner:A);
	pub fn new(inner:A)->Self{
		Self{inner}
	}
}
impl<A> Map<A>{
	accessible_inner!(inner:A);
	pub fn new(inner:A)->Self{
		Self{inner}
	}
}
impl<A> Zip<A>{
	accessible_inner!(inner:A);
	pub fn new(inner:A)->Self{
		Self{inner}
	}
}
impl<F:Fn(X)->Y,M:AI<M::Output,M::Output>+Op,X,Y> IntoSequence<M> for Apply<F,X,Y> where Self:Into<M>{
	fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
}
impl<F:Fn(X)->Y,X,Y> AI<X,Y> for Apply<F,X,Y>{
	fn forward(&self,input:X)->Y{(&self.inner)(input)}
}
impl<F:Fn(X)->Y,X,Y> Op for Apply<F,X,Y>{
	type Output=Y;
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
macro_rules! zip_tuple{
	($(($($type:ident),+):($($input:ident),+)->($($output:ident),+)),*)=>($(
		impl<$($type:AI<$input,$output>,$input,$output),+> AI<($($input),+),($($output),+)> for Zip<($($type),+)>{
			#[allow(non_snake_case)]
			fn forward(&self,($($input),+):($($input),+))->($($output),+){
				let ($($type),+)=self.inner();
				($($type.forward($input)),+)
			}
			#[allow(non_snake_case)]
			fn forward_mut(&mut self,($($input),+):($($input),+))->($($output),+){
				let ($($type),+)=self.inner_mut();
				($($type.forward_mut($input)),+)
			}
		}
		impl<$($type:Op<Output=$output>,$output),+> Op for Zip<($($type),+)>{
			type Output=($($output),+);
		}
	)*);
}
#[cfg(test)]
mod tests{
	#[test]
	fn acc_q_vec(){
		let op=().fix_type::<Vec<f32>>().acc_q(0.5);
		let x:Vec<f32>=vec![1.0,1.0,1.0,1.0,1.0];
		let y:Vec<f32>=op.forward(x);
		assert_eq!(y,[1.9375_f32,1.875,1.75,1.5,1.0]);
	}
	#[test]
	fn mse_vec(){
		let op=().fix_type::<Vec<f32>>().squared_error().mean();
		let x:(Vec<f32>,Vec<f32>)=(vec![0.0,0.5,1.5],vec![-2.0,1.5,5.5]);
		let y:f32=op.forward(x);
		assert_eq!(y,7.0);
	}
	use super::*;
}
#[derive(Clone,Copy,Debug,Deserialize,Eq,Hash,PartialEq,Serialize)]
/// alignment
pub enum Alignment{Center,Left,Right}
#[derive(Clone,Copy,Debug,Deserialize,Eq,Hash,PartialEq,Serialize)]
/// shape mismatch handling
pub enum OnMismatch{Error,Pad(Alignment),Truncate(Alignment)}
#[derive(Clone,Copy,Debug,Deserialize,Eq,Hash,PartialEq,Serialize)]
/// reduction mode
pub enum ReductionMode{Component,Dim(i32),Tensor}
/// creates an operation that applies the closure
pub fn apply<F:Fn(X)->Y,X,Y>(f:F)->Apply<F,X,Y>{
	Apply{inner:f,phantom:PhantomData}
}
/// layers representing important mathematical operations
pub mod math;
/// layers relating to reinforcement learning
pub mod reinforcement;
/// layers relating to cross entropy or softmax
pub mod soft;
/// layers like cat or swap dims that change the organization or structure of tensors
pub mod structural;

impl<A:AI<X,Y>+IntoSequence<M>,M:AI<M::Output,M::Output>+Op,X,Y> IntoSequence<M> for SetType<A,X,Y>{
	fn into_sequence(self)->Sequential<Vec<M>>{self.into_inner().into_sequence()}
}
impl<A:AI<X,Y>+UnwrapInner,X,Y> UnwrapInner for SetType<A,X,Y>{
	fn unwrap_inner(self)->Self::Inner{self.into_inner().unwrap_inner()}
	type Inner=A::Inner;
}

#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// applies a closure to the input// TODO more closure layers maybe
pub struct Apply<F:Fn(X)->Y,X,Y>{inner:F,phantom:PhantomData<fn(X)->Y>}
#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// autoregressive inference
pub struct Autoregression<A,X>{ai:A,state:Option<X>}
#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// module for cloning things
pub struct Duplicate<A>{inner:A,times:usize}
#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// wraps to apply to every element of a vector
pub struct Map<A>{inner:A}

impl<A:AI<X,Y>+Op<Output=Y>,X:Clone+OpsAdd<Y,Output=Z>,Y:Into<Z>,Z> AI<X,Z> for Residual<A>{
	fn forward(&self,x:X)->Z{
		let apply=self.apply;
		let f=&self.inner;

		if apply{x.clone()+f.forward(x)}else{f.forward(x).into()}
	}
	fn forward_mut(&mut self,x:X)->Z{
		let apply=self.apply;
		let f=&mut self.inner;

		if apply{x.clone()+f.forward_mut(x)}else{f.forward_mut(x).into()}
	}
}

#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// layer to add an optional residual connection
pub struct Residual<A>{apply:bool,inner:A}
#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// layer for removing dimensions with size 1
pub struct SqueezeLayer{dim:i32}
#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// wrapper for applying ai modules sequentially
pub struct Sequential<A>{inner:A}
#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// fixes the output type of a layer for a particular input type.
pub struct SetType<A,X,Y>{inner:A,phantom:PhantomData<fn(X)->Y>}
#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// wraps to apply each function
pub struct Zip<A>{inner:A}
use {accessible_inner,zip_tuple};
use crate::{
	AI,Decompose,IntoSequence,Op,UnwrapInner
};
use serde::{Deserialize,Serialize};
use std::{
	iter::FromIterator,marker::PhantomData,ops::Add as OpsAdd
};
zip_tuple!((A,B):(W,X)->(Y,Z),(A,B,C):(U,V,W)->(X,Y,Z),(A,B,C,D):(S,T,U,V)->(W,X,Y,Z),(A,B,C,D,E):(Q,R,S,T,U)->(V,W,X,Y,Z),(A,B,C,D,E,F):(O,P,Q,R,S,T)->(U,V,W,X,Y,Z),(A,B,C,D,E,F,G):(M,N,O,P,Q,R,S)->(T,U,V,W,X,Y,Z),(A,B,C,D,E,F,G,H):(K,L,M,N,O,P,Q,R)->(S,T,U,V,W,X,Y,Z));
