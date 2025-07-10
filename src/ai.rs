decompose_primitive!((),bool,char,f32,f64,i128,i16,i32,i64,i8,isize,u128,u16,u32,u64,u8,usize);
impl Decompose for All{
	fn compose(_decomposition:Self::Decomposition)->Self{All}
	fn decompose(self)->Self::Decomposition{}
	fn decompose_cloned(&self)->Self::Decomposition{}
	type Decomposition=();
}
impl Decompose for Range<usize>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.0..decomposition.1}
	fn decompose(self)->Self::Decomposition{(self.start,self.end)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.start,self.end)}
	type Decomposition=(usize,usize);
}
impl WhichDims for All{
	fn strict(&self)->bool{false}
	fn which_dims(&self)->Self::Iter<'_>{0..usize::MAX}
	type Iter<'a>=Range<usize> where Self:'a;
}
impl WhichDims for Range<usize>{
	fn which_dims(&self)->Self::Iter<'_>{self.clone()}
	type Iter<'a>=Self where Self:'a;
}
impl WhichDims for Vec<usize>{
	fn which_dims(&self)->Self::Iter<'_>{self.iter().copied()}
	type Iter<'a>=Copied<SliceIter<'a,usize>> where Self:'a;
}
impl WhichDims for usize{
	fn which_dims(&self)->Self::Iter<'_>{iter::once(*self)}
	type Iter<'a>=Once<usize> where Self:'a;
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
impl<const N:usize> WhichDims for [usize;N]{
	fn which_dims(&self)->Self::Iter<'_>{self.iter().copied()}
	type Iter<'a>=Copied<SliceIter<'a,usize>> where Self:'a;
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
impl<A:Op<Output=Y>,I,Y:IntoIterator<Item=I>> Op for Cat<A> where Cat<()>:AI<Y,I>{
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
/// implements decompose for primitive types
macro_rules! decompose_primitive{
	($($type:ty),*)=>($(impl Decompose for $type{
		fn compose(decomposition:Self::Decomposition)->Self{decomposition}
		fn decompose(self)->Self::Decomposition{self}
		fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
		type Decomposition=Self;
	})*);
}
macro_rules! decompose_tuple{
	($(($($type:ident),+)),*)=>($(impl<$($type:Decompose),+> Decompose for ($($type),+){
		#[allow(non_snake_case)]
		fn compose(($($type),+):Self::Decomposition)->Self{($(Decompose::compose($type)),+)}
		#[allow(non_snake_case)]
		fn decompose(self)->Self::Decomposition{
			let ($($type),+)=self;
			($($type.decompose()),+)
		}
		#[allow(non_snake_case)]
		fn decompose_cloned(&self)->Self::Decomposition{
			let ($($type),+)=self;
			($($type.decompose_cloned()),+)
		}
		type Decomposition=($($type::Decomposition),+);
	})*);
}
/// implements op for tuples
macro_rules! op_tuple{
	($(($($type:ident),+)),*)=>($(impl<$($type:Op),+> Op for ($($type),+){
		type Output=();
	})*);
}
macro_rules! branch_tuple{
	($(($($type:ident),+):$input:ident->($($output:ident),+)),*)=>($(
		impl<$($type:AI<$input,$output>,$output),+,$input:Clone> AI<$input,($($output),+)> for Branch<($($type),+)>{
			#[allow(non_snake_case)]
			fn forward(&self,input:$input)->($($output),+){
				let ($($type),+)=self.inner();
				($($type.forward(input.clone())),+)
			}
			#[allow(non_snake_case)]
			fn forward_mut(&mut self,input:$input)->($($output),+){
				let ($($type),+)=self.inner_mut();
				($($type.forward_mut(input.clone())),+)
			}
		}
		impl<$($type:Op<Output=$output>,$output),+> Op for Branch<($($type),+)>{
			type Output=($($output),+);
		}
	)*);
}
branch_tuple!((A,B):X->(Y,Z),(A,B,C):W->(X,Y,Z),(A,B,C,D):V->(W,X,Y,Z),(A,B,C,D,E):U->(V,W,X,Y,Z),(A,B,C,D,E,F):T->(U,V,W,X,Y,Z),(A,B,C,D,E,F,G):S->(T,U,V,W,X,Y,Z),(A,B,C,D,E,F,G,H):R->(S,T,U,V,W,X,Y,Z));

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
zip_tuple!((A,B):(W,X)->(Y,Z),(A,B,C):(U,V,W)->(X,Y,Z),(A,B,C,D):(S,T,U,V)->(W,X,Y,Z),(A,B,C,D,E):(Q,R,S,T,U)->(V,W,X,Y,Z),(A,B,C,D,E,F):(O,P,Q,R,S,T)->(U,V,W,X,Y,Z),(A,B,C,D,E,F,G):(M,N,O,P,Q,R,S)->(T,U,V,W,X,Y,Z),(A,B,C,D,E,F,G,H):(K,L,M,N,O,P,Q,R)->(S,T,U,V,W,X,Y,Z));



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

decompose_tuple!((A,B),(A,B,C),(A,B,C,D),(A,B,C,D,E),(A,B,C,D,E,F),(A,B,C,D,E,F,G),(A,B,C,D,E,F,G,H));
op_tuple!((A,B),(A,B,C),(A,B,C,D),(A,B,C,D,E),(A,B,C,D,E,F),(A,B,C,D,E,F,G),(A,B,C,D,E,F,G,H));

impl<A> AccQ<A>{
	accessible_inner!(inner:A);
}
impl<A,X> Autoregression<A,X>{
	accessible_inner!(ai:A);
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
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
/// dimension specifier for all dimensions
pub struct All;
#[derive(Clone,Copy,Debug,Default)]
/// autoregressive inference
pub struct Autoregression<A,X>{ai:A,state:Option<X>}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wrapper for applying ai modules to the same input
pub struct Branch<A>{inner:A}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wrapper for concatenating tensors in the output
pub struct Cat<A>{dim:usize,inner:A}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wrapper for applying mean squared error loss
pub struct MSE<A>{inner:A}
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
/// tells which dimensions to apply an operation
pub trait WhichDims{
	/// returns true if specifying more dims than the tensor has should be an error
	fn strict(&self)->bool{true}
	/// iterates over the dims
	fn which_dims(&self)->Self::Iter<'_>;
	/// the type of dimension iterator
	type Iter<'a>:Iterator<Item=usize> where Self:'a;
}
use {accessible_inner,branch_tuple,op_tuple,decompose_primitive,decompose_tuple,zip_tuple};
use std::{
	iter::{Copied,FromIterator,Once,self},marker::PhantomData,ops::Range,slice::Iter as SliceIter
};
