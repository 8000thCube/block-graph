branch_tuple!((A,B):X->(Y,Z),(A,B,C):W->(X,Y,Z),(A,B,C,D):V->(W,X,Y,Z),(A,B,C,D,E):U->(V,W,X,Y,Z),(A,B,C,D,E,F):T->(U,V,W,X,Y,Z),(A,B,C,D,E,F,G):S->(T,U,V,W,X,Y,Z),(A,B,C,D,E,F,G,H):R->(S,T,U,V,W,X,Y,Z));
decompose_primitive!((),bool,char,f32,f64,i128,i16,i32,i64,i8,isize,u128,u16,u32,u64,u8,usize);
decompose_tuple!((A,B),(A,B,C),(A,B,C,D),(A,B,C,D,E),(A,B,C,D,E,F),(A,B,C,D,E,F,G),(A,B,C,D,E,F,G,H));
impl AI<(Vec<f32>,Vec<f32>),Vec<f32>> for SquaredError<()>{
	fn forward(&self,(output,target):(Vec<f32>,Vec<f32>))->Vec<f32>{
		let (ol,tl)=(output.len(),target.len());
		assert!(ol==tl,"output len {ol} should match target len {tl}");

		output.into_iter().zip(target).map(|(o,t)|o-t).map(|x|x*x).collect()
	}
}
impl AI<(Vec<f32>,Vec<f32>),f32> for CrossEntropy<()>{
	fn forward(&self,(output,target):(Vec<f32>,Vec<f32>))->f32{-output.iter().zip(target.iter()).map(|(o,t)|o.ln()*t).fold(0.0,|acc,x|acc+x)}
}
impl AI<(Vec<f32>,Vec<f32>),f32> for SquaredError<()>{
	fn forward(&self,(output,target):(Vec<f32>,Vec<f32>))->f32{
		let (ol,tl)=(output.len(),target.len());
		assert!(ol==tl,"output len {ol} should match target len {tl}");

		output.into_iter().zip(target).map(|(o,t)|o-t).map(|x|x*x).sum::<f32>()/ol as f32
	}
}
impl AI<(Vec<f32>,Vec<f32>),f32> for SoftEntropy<()>{
	fn forward(&self,(output,target):(Vec<f32>,Vec<f32>))->f32{-new().fix_type::<Vec<f32>>().log_softmax().forward_fixed(output).iter().zip(target.iter()).map(|(o,t)|o*t).fold(0.0,|acc,x|acc+x)}
}
impl AI<(Vec<f32>,u32),f32> for CrossEntropy<()>{
	fn forward(&self,(output,target):(Vec<f32>,u32))->f32{-output[target as usize].ln()}
}
impl AI<(Vec<f32>,u32),f32> for SoftEntropy<()>{
	fn forward(&self,(output,target):(Vec<f32>,u32))->f32{-new().fix_type::<Vec<f32>>().log_softmax().forward_fixed(output)[target as usize]}
}
impl AI<Vec<f32>,Vec<f32>> for AbnormalSoftmax<()>{
	fn forward(&self,input:Vec<f32>)->Vec<f32>{
		let max=input.iter().fold(f32::NEG_INFINITY,|x,&y|if x<y{y}else{x});
		input.into_iter().map(|x|(x-max).exp()).collect()
	}
}
impl AI<Vec<f32>,Vec<f32>> for AccQ<()>{
	fn forward(&self,mut input:Vec<f32>)->Vec<f32>{
		let (dim,gamma)=(self.dim,self.gamma);
		assert!(dim==0,"Dimension index was {dim} but a vec only has one tensor dimension");

		input.iter_mut().rev().fold(0.0,|future,present|{
			*present+=future*gamma;
			*present
		});
		input
	}
}
impl AI<Vec<f32>,Vec<f32>> for LogSoftmax<()>{
	fn forward(&self,input:Vec<f32>)->Vec<f32>{
		let mut sum=0.0;
		input.iter().for_each(|x|sum+=x.exp());
		let r=sum.ln();
		let output:Vec<f32>=input.into_iter().map(|x|x-r).collect();
		output
	}
}
impl AI<Vec<f32>,Vec<f32>> for Softmax<()>{
	fn forward(&self,input:Vec<f32>)->Vec<f32>{
		let max=input.iter().fold(f32::NEG_INFINITY,|x,&y|if x<y{y}else{x});
		let mut sum=0.0;
		let intermediate:Vec<f32>=input.into_iter().map(|x|(x-max).exp()).inspect(|y|sum+=y).collect();
		let r=sum.recip();
		let output:Vec<f32>=intermediate.into_iter().map(|x|r*x).collect();
		output
	}
}
impl AI<Vec<f32>,f32> for Mean<()>{
	fn forward(&self,input:Vec<f32>)->f32{
		let sum:f32=input.iter().sum();

		sum/input.len() as f32
	}
}
impl AI<f32,f32> for Mean<()>{
	fn forward(&self,input:f32)->f32{input}
}
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
impl Decompose for All{
	fn compose(_decomposition:Self::Decomposition)->Self{All}
	fn decompose(self)->Self::Decomposition{}
	fn decompose_cloned(&self)->Self::Decomposition{}
	type Decomposition=();
}
impl Decompose for Identity{
	fn compose(_decomposition:Self::Decomposition)->Self{Self}
	fn decompose(self){}
	fn decompose_cloned(&self){}
	type Decomposition=();
}
impl Decompose for OnMismatch{
	fn compose(decomposition:Self::Decomposition)->Self{
		match decomposition%10{0=>Self::Error,1=>Self::Pad(Alignment::compose(decomposition/10)),2=>Self::Truncate(Alignment::compose(decomposition/10)),_=>panic!("unknown mismatch number")}
	}
	fn decompose(self)->Self::Decomposition{
		match self{Self::Error=>0,Self::Pad(a)=>a.decompose()*10+1,Self::Truncate(a)=>a.decompose()*10+2}
	}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone().decompose()}
	type Decomposition=usize;
}
impl Decompose for Range<usize>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.0..decomposition.1}
	fn decompose(self)->Self::Decomposition{(self.start,self.end)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.start,self.end)}
	type Decomposition=(usize,usize);
}
impl Decompose for ReductionMode{
	fn compose(decomposition:usize)->Self{
		const C:usize=usize::MAX-1;
		const T:usize=usize::MAX;
		match decomposition{C=>Self::Component,T=>Self::Tensor,x=>Self::Dim(x)}
	}
	fn decompose(self)->Self::Decomposition{
		match self{Self::Component=>usize::MAX-1,Self::Dim(x)=>x,Self::Tensor=>usize::MAX}
	}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone().decompose()}
	type Decomposition=usize;
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
impl Op for AccQ<()>{
	type Output=Vec<f32>;
}
impl Op for AbnormalSoftmax<()>{
	type Output=Vec<f32>;
}
impl Op for CrossEntropy<()>{// TODO these should have vec outputs. explicit reduction would compose more intuitively
	type Output=f32;
}
impl Op for SoftEntropy<()>{
	type Output=f32;
}
impl Op for Identity{
	type Output=();
}
impl Op for LogSoftmax<()>{
	type Output=Vec<f32>;
}
impl Op for Mean<()>{
	type Output=f32;
}
impl Op for SquaredError<()>{
	type Output=f32;
}
impl Op for SoftChoose<()>{
	type Output=u32;
}
impl Op for Softmax<()>{
	type Output=Vec<f32>;
}
impl WhichDims for All{
	fn is_strict(&self)->bool{false}
	fn which_dims(&self,rank:usize)->Self::Iter<'_>{0..rank}
	type Iter<'a>=Range<usize> where Self:'a;
}
impl WhichDims for Range<usize>{
	fn which_dims(&self,rank:usize)->Self::Iter<'_>{rank.min(self.start)..rank.min(self.end)}
	type Iter<'a>=Self where Self:'a;
}
impl WhichDims for Vec<usize>{
	fn which_dims(&self,_rank:usize)->Self::Iter<'_>{self.iter().copied()}
	type Iter<'a>=Copied<SliceIter<'a,usize>> where Self:'a;
}
impl WhichDims for usize{
	fn which_dims(&self,_rank:usize)->Self::Iter<'_>{iter::once(*self)}
	type Iter<'a>=Once<usize> where Self:'a;
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
impl<A:AI<X,X>+Op<Output=X>,X> Op for Option<A>{
	type Output=X;
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
impl<A:AI<X,X>+Op<Output=X>,X> Op for Sequential<Vec<A>>{
	type Output=X;
}
impl<A:AI<X,Y>+Op<Output=Y>,T,X,Y,Z> AI<(X,T),Z> for CrossEntropy<A> where CrossEntropy<()>:AI<(Y,T),Z>{
	fn forward(&self,(input,target):(X,T))->Z{self.with_inner(()).forward((self.inner().forward(input),target))}
	fn forward_mut(&mut self,(input,target):(X,T))->Z{self.with_inner(()).forward((self.inner_mut().forward_mut(input),target))}
}
impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for Mean<A> where Mean<()>:AI<Y,Z>{
	fn forward(&self,input:X)->Z{self.with_inner(()).forward(self.inner().forward(input))}
	fn forward_mut(&mut self,input:X)->Z{self.with_inner(()).forward(self.inner_mut().forward_mut(input))}
}
impl<A:AI<X,Y>+Op<Output=Y>,T,X,Y,Z> AI<(X,T),Z> for SquaredError<A> where SquaredError<()>:AI<(Y,T),Z>{
	fn forward(&self,(input,target):(X,T))->Z{self.with_inner(()).forward((self.inner().forward(input),target))}
	fn forward_mut(&mut self,(input,target):(X,T))->Z{self.with_inner(()).forward((self.inner_mut().forward_mut(input),target))}
}
impl<A:AI<X,Y>+Op<Output=Y>,T,X,Y,Z> AI<(X,T),Z> for SoftEntropy<A> where SoftEntropy<()>:AI<(Y,T),Z>{
	fn forward(&self,(input,target):(X,T))->Z{self.with_inner(()).forward((self.inner().forward(input),target))}
	fn forward_mut(&mut self,(input,target):(X,T))->Z{self.with_inner(()).forward((self.inner_mut().forward_mut(input),target))}
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
impl<A:AI<X,Y>+Op<Output=Y>,D,X,Y,Z> AI<X,Z> for TruncateToMatch<A,D> where for<'a>TruncateToMatch<(),&'a D>:AI<Y,Z>{
	fn forward(&self,input:X)->Z{self.with_inner(&self.dims,()).forward(self.inner().forward(input))}
	fn forward_mut(&mut self,input:X)->Z{
		let input=self.inner_mut().forward_mut(input);
		self.with_inner(&self.dims,()).forward(input)
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
impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for SoftChoose<A> where SoftChoose<()>:AI<Y,Z>{
	fn forward(&self,input:X)->Z{self.with_inner(()).forward(self.inner().forward(input))}
	fn forward_mut(&mut self,input:X)->Z{self.with_inner(()).forward(self.inner_mut().forward_mut(input))}
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
impl<A:AI<X,Y>,X,Y> AI<X,Y> for &A{
	fn forward(&self,input:X)->Y{(**self).forward(input)}
}
impl<A:AI<X,Y>,X,Y> AI<X,Y> for &mut A{
	fn forward(&self,input:X)->Y{(**self).forward(input)}
	fn forward_mut(&mut self,input:X)->Y{(**self).forward_mut(input)}
}
impl<A:AI<X,Y>,X,Y> AI<X,Y> for AbnormalSoftmax<A> where AbnormalSoftmax<()>:AI<Y,Y>{
	fn forward(&self,input:X)->Y{self.with_inner(()).forward(self.inner.forward(input))}
	fn forward_mut(&mut self,input:X)->Y{self.with_inner(()).forward(self.inner.forward_mut(input))}
}
impl<A:AI<X,Y>,X,Y> AI<X,Y> for AccQ<A> where AccQ<()>:AI<Y,Y>{
	fn forward(&self,input:X)->Y{self.with_inner(()).forward(self.inner.forward(input))}
	fn forward_mut(&mut self,input:X)->Y{self.with_inner(()).forward(self.inner.forward_mut(input))}
}
impl<A:AI<X,Y>,X,Y> AI<X,Y> for LogSoftmax<A> where LogSoftmax<()>:AI<Y,Y>{
	fn forward(&self,input:X)->Y{self.with_inner(()).forward(self.inner.forward(input))}
	fn forward_mut(&mut self,input:X)->Y{self.with_inner(()).forward(self.inner.forward_mut(input))}
}
impl<A:AI<X,Y>,X,Y> AI<X,Y> for Softmax<A> where Softmax<()>:AI<Y,Y>{
	fn forward(&self,input:X)->Y{self.with_inner(()).forward(self.inner.forward(input))}
	fn forward_mut(&mut self,input:X)->Y{self.with_inner(()).forward(self.inner.forward_mut(input))}
}
impl<A:AI<X,Y>,X,Y> Op for SetType<A,X,Y>{
	type Output=Y;
}
impl<A:AI<X,Y>,X,Y> SetType<A,X,Y>{
	accessible_inner!(inner:A);
}
impl<A:Decompose,D:Decompose> Decompose for TruncateToMatch<A,D>{
	fn compose((inner,alignment,dims):Self::Decomposition)->Self{
		Self{alignment:Alignment::compose(alignment),dims:D::compose(dims),inner:A::compose(inner)}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.alignment.decompose(),self.dims.decompose())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.alignment.decompose_cloned(),self.dims.decompose_cloned())}
	type Decomposition=(A::Decomposition,<Alignment as Decompose>::Decomposition,D::Decomposition);
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
impl<A:Decompose> Decompose for AbnormalSoftmax<A>{
	fn compose((inner,_temperature,_dim):Self::Decomposition)->Self{
		Self{inner:A::compose(inner)}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),1.0,-1)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),1.0,-1)}
	type Decomposition=(A::Decomposition,f32,isize);
}
impl<A:Decompose> Decompose for AccQ<A>{
	fn compose((inner,gamma,dim):Self::Decomposition)->Self{
		Self{dim,gamma,inner:A::compose(inner)}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.gamma,self.dim)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.gamma,self.dim)}
	type Decomposition=(A::Decomposition,f32,usize);
}
impl<A:Decompose> Decompose for Branch<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for CrossEntropy<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for SoftEntropy<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
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
impl<A:Decompose> Decompose for LogSoftmax<A>{
	fn compose((inner,_temperature,_dim):Self::Decomposition)->Self{
		Self{inner:A::compose(inner)}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),1.0,-1)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),1.0,-1)}
	type Decomposition=(A::Decomposition,f32,isize);
}
impl<A:Decompose> Decompose for Mean<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for SquaredError<A>{
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
impl<A:Decompose> Decompose for SoftChoose<A>{// TODO dim should really be isize
	fn compose((inner,temperature,dim):Self::Decomposition)->Self{
		Self{dim,inner:A::compose(inner),temperature}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.temperature,self.dim)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.temperature,self.dim)}
	type Decomposition=(A::Decomposition,f32,usize);
}
impl<A:Decompose> Decompose for Softmax<A>{
	fn compose((inner,_temperature,_dim):Self::Decomposition)->Self{
		Self{inner:A::compose(inner)}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),1.0,-1)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),1.0,-1)}
	type Decomposition=(A::Decomposition,f32,isize);
}
impl<A:Decompose> Decompose for Map<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
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
impl<A:Op<Output=Y>,D,Y> Op for TruncateToMatch<A,D> where TruncateToMatch<(),D>:AI<Y,Y>{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for &A{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for &mut A{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for AbnormalSoftmax<A> where AbnormalSoftmax<()>:AI<Y,Y>{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for AccQ<A> where AccQ<()>:AI<Y,Y>{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for Branch<Vec<A>>{
	type Output=Vec<Y>;
}
impl<A:Op<Output=Y>,Y> Op for CrossEntropy<A> where CrossEntropy<()>:AI<(Y,Y),f32>{
	type Output=f32;
}
impl<A:Op<Output=Y>,Y> Op for Duplicate<A>{
	type Output=(Y,Y);
}
impl<A:Op<Output=Y>,Y> Op for Mean<A> where Mean<()>:AI<Y,f32>{
	type Output=f32;
}
impl<A:Op<Output=Y>,Y> Op for SquaredError<A> where SquaredError<()>:AI<(Y,Y),f32>{
	type Output=f32;
}
impl<A:Op<Output=Y>,Y> Op for SoftChoose<A> where SoftChoose<()>:AI<Y,u32>{
	type Output=u32;
}
impl<A:Op<Output=Y>,Y> Op for SoftEntropy<A> where SoftEntropy<()>:AI<(Y,Y),f32>{
	type Output=f32;
}
impl<A:Op<Output=Y>,Y> Op for LogSoftmax<A> where LogSoftmax<()>:AI<Y,Y>{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for Map<A>{
	type Output=Vec<Y>;
}
impl<A:Op<Output=Y>,Y> Op for Softmax<A> where Softmax<()>:AI<Y,Y>{
	type Output=Y;
}
impl<A,D> TruncateToMatch<A,D>{
	accessible_inner!(inner:A);
	/// gets the alignment
	pub fn alignment(&self)->Alignment{self.alignment}
	/// gets the dims
	pub fn dims(&self)->&D{&self.dims}
	/// creates from the inner value
	pub fn from_inner(alignment:Alignment,dims:D,inner:A)->Self{
		TruncateToMatch{alignment,dims,inner}
	}
	/// replaces the inner value
	pub fn with_inner<B,E>(&self,dims:E,inner:B)->TruncateToMatch<B,E> where TruncateToMatch<B,E>:Op{TruncateToMatch::from_inner(self.alignment,dims,inner)}
}
impl<A,X> Autoregression<A,X>{
	accessible_inner!(ai:A);
}
impl<A> Op for Vec<A>{
	type Output=();
}
impl<A> AccQ<A>{
	accessible_inner!(inner:A);
	/// gets the dimension
	pub fn dim(&self)->usize{self.dim}
	/// creates from the inner value
	pub fn from_inner(dim:usize,gamma:f32,inner:A)->Self{
		AccQ{dim,inner,gamma}
	}
	/// gets the gamma
	pub fn gamma(&self)->f32{self.gamma}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->AccQ<B> where AccQ<B>:Op{AccQ::from_inner(self.dim,self.gamma,inner)}
}
impl<A> Branch<A>{
	accessible_inner!(inner:A);
}
impl<A> CrossEntropy<A>{
	accessible_inner!(inner:A);
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self{
		CrossEntropy{inner}
	}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->CrossEntropy<B> where CrossEntropy<B>:Op{CrossEntropy::from_inner(inner)}
}
impl<A> Duplicate<A>{
	accessible_inner!(inner:A);
	/// creates a new duplicate module from the inner value
	pub fn from_inner(inner:A)->Self{
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
impl<A> Mean<A>{
	accessible_inner!(inner:A);
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self{
		Mean{inner}
	}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->Mean<B> where Mean<B>:Op{Mean::from_inner(inner)}
}
impl<A> SquaredError<A>{
	accessible_inner!(inner:A);
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self{
		SquaredError{inner}
	}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->SquaredError<B> where SquaredError<B>:Op{SquaredError::from_inner(inner)}
}
impl<A> Sequential<A>{
	accessible_inner!(inner:A);
}
impl<A> SoftChoose<A>{
	accessible_inner!(inner:A);
	/// gets the dimension to choose along
	pub fn dim(&self)->usize{self.dim}
	/// creates from the inner value
	pub fn from_inner(dim:usize,inner:A,temperature:f32)->Self{
		SoftChoose{dim,inner,temperature}
	}
	/// gets the temperature to soft choose with
	pub fn temperature(&self)->f32{self.temperature}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->SoftChoose<B> where SoftChoose<B>:Op{SoftChoose::from_inner(self.dim,inner,self.temperature)}
}
impl<A> SoftEntropy<A>{
	accessible_inner!(inner:A);
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self{
		SoftEntropy{inner}
	}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->SoftEntropy<B> where SoftEntropy<B>:Op{SoftEntropy::from_inner(inner)}
}
impl<A> AbnormalSoftmax<A>{
	accessible_inner!(inner:A);
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self{
		AbnormalSoftmax{inner}
	}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->AbnormalSoftmax<B> where AbnormalSoftmax<B>:Op{AbnormalSoftmax::from_inner(inner)}
}
impl<A> LogSoftmax<A>{
	accessible_inner!(inner:A);
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self{
		LogSoftmax{inner}
	}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->LogSoftmax<B> where LogSoftmax<B>:Op{LogSoftmax::from_inner(inner)}
}
impl<A> Softmax<A>{
	accessible_inner!(inner:A);
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self{
		Softmax{inner}
	}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->Softmax<B> where Softmax<B>:Op{Softmax::from_inner(inner)}
}
impl<A> Map<A>{
	accessible_inner!(inner:A);
}
impl<A> Zip<A>{
	accessible_inner!(inner:A);
}
impl<D:WhichDims,X> AI<Vec<Vec<X>>,Vec<Vec<X>>> for TruncateToMatch<(),D>{
	fn forward(&self,mut input:Vec<Vec<X>>)->Vec<Vec<X>>{
		let mut dims=self.dims.which_dims(1);
		if self.dims.is_strict(){
			let dim=if let Some(d)=dims.next(){d}else{return input};
			let count=dims.count()+1;
			assert!((count,dim)==(1,0),"Dimension index was {dim} but a vec only has one tensor dimension");
		}else{
			if dims.all(|x|x!=0){return input}
		}
		let l=input.iter().map(|x|x.len()).min().unwrap_or(0);
		input.iter_mut().for_each(|x|x.truncate(l));
		input
	}
}
impl<D:WhichDims> WhichDims for &D{
	fn is_strict(&self)->bool{(**self).is_strict()}
	fn which_dims(&self,rank:usize)->Self::Iter<'_>{(**self).which_dims(rank)}
	type Iter<'a>=D::Iter<'a> where Self:'a;
}
impl<D> Op for TruncateToMatch<(),D>{
	type Output=Vec<Vec<()>>;
}
impl<F:Fn(X)->Y,X,Y> AI<X,Y> for Apply<F,X,Y>{
	fn forward(&self,input:X)->Y{(&self.inner)(input)}
}
impl<F:Fn(X)->Y,X,Y> Op for Apply<F,X,Y>{
	type Output=Y;
}
impl<K:Decompose+Eq+Hash,V:Decompose,S:Default+BuildHasher> Decompose for HashMap<K,V,S>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.into_iter().map(Decompose::compose).collect()}
	fn decompose(self)->Self::Decomposition{self.into_iter().map(Decompose::decompose).collect()}
	fn decompose_cloned(&self)->Self::Decomposition{self.iter().map(|(k,v)|(k.decompose_cloned(),v.decompose_cloned())).collect()}
	type Decomposition=Vec<(K::Decomposition,V::Decomposition)>;
}
impl<X:Into<Y>,Y> AI<X,Y> for Identity{
	fn forward(&self,input:X)->Y{input.into()}
}
impl<X> AI<Vec<Vec<X>>,Vec<X>> for CatLayer{
	fn forward(&self,input:Vec<Vec<X>>)->Vec<X>{
		let dim=self.dim;
		assert!(dim==0,"Dimension index was {dim} but a vec only has one tensor dimension");

		let acc=Vec::with_capacity(input.iter().map(|x|x.len()).sum());
		input.into_iter().fold(acc,|mut acc,x|{
			acc.extend(x);
			acc
		})
	}
}
impl<const N:usize> WhichDims for [usize;N]{
	fn which_dims(&self,_rank:usize)->Self::Iter<'_>{self.iter().copied()}
	type Iter<'a>=Copied<SliceIter<'a,usize>> where Self:'a;
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
		let op=new().fix_type::<Vec<f32>>().acc_q(0,0.5);
		let x:Vec<f32>=vec![1.0,1.0,1.0,1.0,1.0];
		let y:Vec<f32>=op.forward(x);
		assert_eq!(y,[1.9375_f32,1.875,1.75,1.5,1.0]);
	}
	#[test]
	fn cat_vec(){
		let op=new().fix_type::<Vec<Vec<f32>>>().cat(0);
		let x:Vec<Vec<f32>>=vec![vec![1.0,1.0,1.0,1.0,1.0],vec![2.0,2.0,2.0]];
		let y=op.forward(x);
		assert_eq!(y,[1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0]);
	}
	#[test]
	fn mse_vec(){
		let op=new().fix_type::<Vec<f32>>().squared_error().mean();
		let x:(Vec<f32>,Vec<f32>)=(vec![0.0,0.5,1.5],vec![-2.0,1.5,5.5]);
		let y:f32=op.forward(x);
		assert_eq!(y,7.0);
	}
	/*#[test]
	fn sum_vec(){
		let op=new().fix_type::<Vec<f32>>().sum()
	}*/
	use super::*;
}
op_tuple!((A,B),(A,B,C),(A,B,C,D),(A,B,C,D,E),(A,B,C,D,E,F),(A,B,C,D,E,F,G),(A,B,C,D,E,F,G,H));
#[derive(Clone,Copy,Debug,Eq,Hash,PartialEq)]
/// alignment
pub enum Alignment{Center,Left,Right}
#[derive(Clone,Copy,Debug,Eq,Hash,PartialEq)]
/// shape mismatch handling
pub enum OnMismatch{Error,Pad(Alignment),Truncate(Alignment)}
#[derive(Clone,Copy,Debug,Eq,Hash,PartialEq)]
/// reduction mode
pub enum ReductionMode{Component,Dim(usize),Tensor}
/// creates an operation that applies the closure
pub fn apply<F:Fn(X)->Y,X,Y>(f:F)->Apply<F,X,Y>{
	Apply{inner:f,phantom:PhantomData}
}
/// starts the building of an ai structure in chained method style from an identity operation
pub fn new()->Identity{Identity}
/// undivided softmax
pub struct AbnormalSoftmax<A>{inner:A}// TODO macros: cat_like map_like reduce_like soft_like

/*
impl AI<Vec<f32>,f32> for SumLayer{
	fn forward(&self,input:Vec<f32>)->f32{input.into_iter().sum()}//TODO check dim
}*/

impl<E> AI<Vec<Vec<E>>,Vec<E>> for StackLayer{// TODO squeeze unsqueeze so we can properly implement this
	fn forward(&self,input:Vec<Vec<E>>)->Vec<E>{todo!()}
}
impl<E> AI<Vec<Vec<E>>,Vec<Vec<E>>> for StackLayer{
	fn forward(&self,input:Vec<Vec<E>>)->Vec<Vec<E>>{todo!()}
}

/// declares layer and wrapper structs and implements accessor functions, decompose and op for reduction operations that have dim and mismatch behavior as configuration fields. ai will still have to be externally implemented for the layer stuct
macro_rules! cat_like{
	($layer:ident,$wrap:ident)=>{
		impl $layer{
			/// gets the dimension
			pub fn dim(&self)->&usize{&self.dim}
			/// gets the dimension
			pub fn dim_mut(&mut self)->&mut usize{&mut self.dim}
			/// gets the dimension
			pub fn get_dim(&self)->usize{self.dim}
			/// gets the mismatch behavior
			pub fn get_mismatch_behavior(&self)->OnMismatch{self.mismatchbehavior}
			/// gets the mismatch behavior
			pub fn mismatch_behavior(&self)->&OnMismatch{&self.mismatchbehavior}
			/// gets the mismatch behavior
			pub fn mismatch_behavior_mut(&mut self)->&mut OnMismatch{&mut self.mismatchbehavior}
			/// creates a new layer
			pub fn new(dim:usize)->Self{Self::default().with_dim(dim)}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:usize){self.dim=dim}
			/// sets the mismatch behavior
			pub fn set_mismatch_behavior(&mut self,behavior:OnMismatch){self.mismatchbehavior=behavior}
			/// sets the dimension
			pub fn with_dim(mut self,dim:usize)->Self{
				self.dim=dim;
				self
			}
			/// sets the mismatch behavior
			pub fn with_mismatch_behavior(mut self,behavior:OnMismatch)->Self{
				self.mismatchbehavior=behavior;
				self
			}
		}
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose((inner,layer):Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:$layer::compose(layer)}
			}
			fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.layer.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.layer.decompose_cloned())}
			type Decomposition=(A::Decomposition,<$layer as Decompose>::Decomposition);
		}
		impl<A:Op<Output=Y>,Y:IntoIterator<Item=Z>,Z> Op for $wrap<A> where $layer:AI<Y,Z>{
			type Output=Z;
		}
		impl<A> $wrap<A>{
			/// gets the dimension
			pub fn dim(&self)->&usize{&self.layer.dim}
			/// gets the dimension
			pub fn dim_mut(&mut self)->&mut usize{&mut self.layer.dim}
			/// gets the dimension
			pub fn get_dim(&self)->usize{self.layer.dim}
			/// gets the inner layer
			pub fn get_inner(&self)->A where A:Copy{self.inner}
			/// gets the mismatch behavior
			pub fn get_mismatch_behavior(&self)->OnMismatch{self.layer.mismatchbehavior}
			/// gets the inner layer
			pub fn inner(&self)->&A{&self.inner}
			/// gets the inner layer
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// gets the inner layer
			pub fn into_inner(self)->A{self.inner}
			/// gets the mismatch behavior
			pub fn mismatch_behavior(&self)->&OnMismatch{&self.layer.mismatchbehavior}
			/// gets the mismatch behavior
			pub fn mismatch_behavior_mut(&mut self)->&mut OnMismatch{&mut self.layer.mismatchbehavior}
			/// creates a new layer
			pub fn new(dim:usize,inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new(dim)}
			}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:usize){self.layer.dim=dim}
			/// sets the inner module
			pub fn set_inner(&mut self,inner:A){self.inner=inner}
			/// sets the mismatch behavior
			pub fn set_mismatch_behavior(&mut self,behavior:OnMismatch){self.layer.mismatchbehavior=behavior}
			/// sets the dimension
			pub fn with_dim(mut self,dim:usize)->Self{
				self.layer.dim=dim;
				self
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
			/// sets the mismatch behavior
			pub fn with_mismatch_behavior(mut self,behavior:OnMismatch)->Self{
				self.layer.mismatchbehavior=behavior;
				self
			}
		}
		impl Decompose for $layer{
			fn compose((dim,mismatchbehavior):Self::Decomposition)->Self{
				Self{dim,mismatchbehavior:OnMismatch::compose(mismatchbehavior)}
			}
			fn decompose(self)->Self::Decomposition{(self.dim,self.mismatchbehavior.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.dim,self.mismatchbehavior.decompose_cloned())}
			type Decomposition=(usize,usize);
		}
		impl Op for $layer{
			type Output=Vec<()>;
		}
		#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
		/// layer to apply an operation
		pub struct $layer{dim:usize,mismatchbehavior:OnMismatch}
		#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	}
}

/// declares layer and wrapper structs and implements accessor functions, decompose and op for reduction operations that have dim and reduction mode as configuration fields. ai will still have to be externally implemented for the layer stuct
macro_rules! reduce_like{
	($layer:ident,$wrap:ident)=>{
		impl $layer{
			/// gets the dimension
			pub fn dim(&self)->&usize{&self.dim}
			/// gets the dimension
			pub fn dim_mut(&mut self)->&mut usize{&mut self.dim}
			/// gets the dimension
			pub fn get_dim(&self)->usize{self.dim}
			/// gets the reduction mode
			pub fn get_reduction_mode(&self)->ReductionMode{self.reductionmode}
			/// gets the reduction mode
			pub fn reduction_mode(&self)->&ReductionMode{&self.reductionmode}
			/// gets the reduction mode
			pub fn reduction_mode_mut(&mut self)->&mut ReductionMode{&mut self.reductionmode}
			/// creates a new layer
			pub fn new(dim:usize)->Self{Self::default().with_dim(dim)}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:usize){self.dim=dim}
			/// sets the reduction mode
			pub fn set_reduction_mode(&mut self,mode:ReductionMode){self.reductionmode=mode}
			/// sets the dimension
			pub fn with_dim(mut self,dim:usize)->Self{
				self.dim=dim;
				self
			}
			/// sets the reduction mode
			pub fn with_reduction_mode(mut self,mode:ReductionMode)->Self{
				self.reductionmode=mode;
				self
			}
		}
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose((inner,layer):Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:$layer::compose(layer)}
			}
			fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.layer.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.layer.decompose_cloned())}
			type Decomposition=(A::Decomposition,<$layer as Decompose>::Decomposition);
		}
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<Y,f32>{
			type Output=f32;
		}
		impl<A> $wrap<A>{
			/// gets the dimension
			pub fn dim(&self)->&usize{&self.layer.dim}
			/// gets the dimension
			pub fn dim_mut(&mut self)->&mut usize{&mut self.layer.dim}
			/// gets the dimension
			pub fn get_dim(&self)->usize{self.layer.dim}
			/// gets the inner layer
			pub fn get_inner(&self)->A where A:Copy{self.inner}
			/// gets the reduction mode
			pub fn get_reduction_mode(&self)->ReductionMode{self.layer.reductionmode}
			/// gets the inner layer
			pub fn inner(&self)->&A{&self.inner}
			/// gets the inner layer
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// gets the inner layer
			pub fn into_inner(self)->A{self.inner}
			/// gets the reduction mode
			pub fn reduction_mode(&self)->&ReductionMode{&self.layer.reductionmode}
			/// gets the reduction mode
			pub fn reduction_mode_mut(&mut self)->&mut ReductionMode{&mut self.layer.reductionmode}
			/// creates a new layer
			pub fn new(dim:usize,inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new(dim)}
			}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:usize){self.layer.dim=dim}
			/// sets the inner module
			pub fn set_inner(&mut self,inner:A){self.inner=inner}
			/// sets the reduction mode
			pub fn set_reduction_mode(&mut self,mode:ReductionMode){self.layer.reductionmode=mode}
			/// sets the dimension
			pub fn with_dim(mut self,dim:usize)->Self{
				self.layer.dim=dim;
				self
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
			/// sets the reduction mode
			pub fn with_reduction_mode(mut self,mode:ReductionMode)->Self{
				self.layer.reductionmode=mode;
				self
			}
		}
		impl Decompose for $layer{
			fn compose((dim,reductionmode):Self::Decomposition)->Self{
				Self{dim,reductionmode:ReductionMode::compose(reductionmode)}
			}
			fn decompose(self)->Self::Decomposition{(self.dim,self.reductionmode.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.dim,self.reductionmode.decompose_cloned())}
			type Decomposition=(usize,usize);
		}
		impl Op for $layer{
			type Output=f32;
		}
		#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
		/// layer to apply an operation
		pub struct $layer{dim:usize,reductionmode:ReductionMode}
		#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	}
}
cat_like!(CatLayer,Cat);
//cat_like!(MeanLayer,Mean);
cat_like!(StackLayer,Stack);
reduce_like!(SumLayer,Sum);

#[derive(Clone,Copy,Debug,Default)]
/// accumulates cumulative
pub struct AccQ<A>{dim:usize,gamma:f32,inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// dimension specifier for all dimensions
pub struct All;
#[derive(Clone,Copy,Debug,Default)]
/// applies a closure to the input
pub struct Apply<F:Fn(X)->Y,X,Y>{inner:F,phantom:PhantomData<fn(X)->Y>}
#[derive(Clone,Copy,Debug,Default)]
/// autoregressive inference
pub struct Autoregression<A,X>{ai:A,state:Option<X>}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wrapper for applying ai modules to the same input
pub struct Branch<A>{inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for applying cross entropy loss
pub struct CrossEntropy<A>{inner:A}//TODO dim
#[derive(Clone,Copy,Debug,Default)]
/// module for cloning things
pub struct Duplicate<A>{inner:A,times:usize}
#[derive(Clone,Copy,Debug,Default)]
/// ai module for returning the input
pub struct Identity;
#[derive(Clone,Copy,Debug,Default)]
/// log softmax
pub struct LogSoftmax<A>{inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for applying mean squared error loss
pub struct SquaredError<A>{inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// wraps to apply to every element of a vector
pub struct Map<A>{inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for applying mean
pub struct Mean<A>{inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for applying ai modules sequentially
pub struct Sequential<A>{inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// fixes the output type of a layer for a particular input type.
pub struct SetType<A,X,Y>{inner:A,phantom:PhantomData<fn(X)->Y>}
#[derive(Clone,Copy,Debug,Default)]
/// chooses from the softmax
pub struct SoftChoose<A>{dim:usize,inner:A,temperature:f32}//TODO with operations //TODO hard choose
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for applying cross entropy loss
pub struct SoftEntropy<A>{inner:A}//TODO dim
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for applying softmax
pub struct Softmax<A>{inner:A}//TODO dim, temperature
#[derive(Clone,Copy,Debug,Default)]
/// truncates each tensor dimension in the list to the minimum so that they match
pub struct TruncateToMatch<A,D>{alignment:Alignment,dims:D,inner:A}
#[derive(Clone,Copy,Debug,Default)]
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
	/// wraps with a softmax operation
	fn abnormal_softmax(self)->AbnormalSoftmax<Self> where Self:Sized,AbnormalSoftmax<Self>:Op{
		AbnormalSoftmax{inner:self}
	}
	/// wraps with a accq operation
	fn acc_q(self,dim:usize,gamma:f32)->AccQ<Self> where AccQ<Self>:Op,Self:Sized{
		AccQ{dim,gamma,inner:self}
	}
	/// wraps with a branch operation
	fn branch(self)->Branch<Self> where Branch<Self>:Op,Self:Sized{
		Branch{inner:self}
	}
	/// wraps with a cat operation
	fn cat(self,dim:usize)->Cat<Self> where Cat<Self>:Op,Self:Sized{Cat::new(dim,self)}
	/// sequences with another ai operation
	fn chain<B>(self,b:B)->Sequential<(Self,B)> where Self:Sized,Sequential<(Self,B)>:Op{
		Sequential{inner:(self,b)}
	}
	/// wraps with a mse operation
	fn cross_entropy(self)->CrossEntropy<Self> where CrossEntropy<Self>:Op,Self:Sized{
		CrossEntropy{inner:self}
	}
	/// wraps with a duplicate operation
	fn duplicate(self)->Duplicate<Self> where Duplicate<Self>:Op,Self:Sized{
		Duplicate{inner:self,times:2}
	}
	/// set type but with the same input and output
	fn fix_type<Z>(self)->SetType<Self,Z,Z> where Self:AI<Z,Z>+Sized{self.set_type()}
	/// applies to the input
	fn forward_fixed<Z>(&self,input:Z)->Z where Self:AI<Z,Z>+Sized{self.forward(input)}
	/// applies to the input
	fn forward_fixed_mut<Z>(&mut self,input:Z)->Z where Self:AI<Z,Z>+Sized{self.forward(input)}
	/// applies to the input
	fn forward_typed<W,Z>(&self,input:W)->Z where Self:AI<W,Z>+Sized{self.forward(input)}
	/// applies to the input, possibly updating internal caches
	fn forward_typed_mut<W,Z>(&mut self,input:W)->Z where Self:AI<W,Z>+Sized{self.forward(input)}
	/// creates an autoregressive inference
	fn infer_autoregressive<X,Y>(self,input:X)->Autoregression<Self,Y> where Self:AI<X,Y>+AI<Y,Y>+Sized,Y:Clone{
		let mut ai=self;
		let state=Some(ai.forward_mut(input));
		Autoregression{ai,state}
	}
	/// wraps with a softmax operation
	fn log_softmax(self)->LogSoftmax<Self> where Self:Sized,LogSoftmax<Self>:Op{
		LogSoftmax{inner:self}
	}
	/// applies the operation to every output
	fn map<B>(self,b:B)->Map<Sequential<(Self,B)>> where Map<Sequential<(Self,B)>>:Op,Self:Sized,Sequential<(Self,B)>:Op{self.chain(b).to_each()}
	/// wraps with a mean operation
	fn mean(self)->Mean<Self> where Mean<Self>:Op,Self:Sized{
		Mean{inner:self}
	}
	/// creates an optional operation
	fn optional(self)->Option<Self> where Self:Sized{Some(self)}
	/// produces a zip module
	fn separately(self)->Zip<Self> where Self:Sized,Zip<Self>:Op{
		Zip{inner:self}
	}
	/// produces a sequential module
	fn sequential(self)->Sequential<Self> where Self:Sized,Sequential<Self>:Op{
		Sequential{inner:self}
	}
	/// sets the input output types
	fn set_type<W,Z>(self)->SetType<Self,W,Z> where Self:AI<W,Z>+Sized{
		SetType{inner:self,phantom:PhantomData}
	}
	/// wraps with a choose operation
	fn soft_choose(self,dim:usize,temperature:f32)->SoftChoose<Self> where Self:Sized,SoftChoose<Self>:Op{
		SoftChoose{dim,inner:self,temperature}
	}
	/// wraps with a mse operation
	fn soft_entropy(self)->SoftEntropy<Self> where SoftEntropy<Self>:Op,Self:Sized{
		SoftEntropy{inner:self}
	}
	/// wraps with a softmax operation
	fn softmax(self)->Softmax<Self> where Self:Sized,Softmax<Self>:Op{
		Softmax{inner:self}
	}
	/// wraps with a mse operation
	fn squared_error(self)->SquaredError<Self> where SquaredError<Self>:Op,Self:Sized{
		SquaredError{inner:self}
	}
	/// wraps with a map operation
	fn to_each(self)->Map<Self> where Map<Self>:Op,Self:Sized{
		Map{inner:self}
	}
	/// wraps with a truncate to match operation. alignment=0 for left alignment. will have other alignment settings in the future
	fn truncate_to_match<D>(self,dims:D)->TruncateToMatch<Self,D> where Self:Sized,TruncateToMatch<Self,D>:Op{
		TruncateToMatch{alignment:Alignment::Left,dims,inner:self}
	}
	/// zips with another ai operation
	fn zip<B>(self,b:B)->Zip<(Self,B)> where Self:Sized,Zip<(Self,B)>:Op{
		Zip{inner:(self,b)}
	}
	/// suggested output type to help with composition coherence. Ideally, Self should implement AI<X,Self::Output> for some X
	type Output;
}
/// trait for unwrapping nested wrapped values
pub trait UnwrapInner<T>{
	/// unwraps the inner value
	fn unwrap_inner(self)->T;
}
/// tells which dimensions to apply an operation
pub trait WhichDims:Clone{
	/// returns true if specifying more dims than the tensor has should be an error
	fn is_strict(&self)->bool{true}
	/// iterates over the dims. tensor rank is provided to prevent eternal or excessively long loops due to iteration, but isn't necessarily a limitation on what dims are returned
	fn which_dims(&self,rank:usize)->Self::Iter<'_>;
	/// the type of dimension iterator
	type Iter<'a>:Iterator<Item=usize> where Self:'a;
}
use {accessible_inner,branch_tuple,op_tuple,decompose_primitive,decompose_tuple,zip_tuple};
use std::{
	collections::HashMap,hash::{BuildHasher,Hash},iter::{Copied,FromIterator,Once,self},marker::PhantomData,ops::Range,slice::Iter as SliceIter
};
zip_tuple!((A,B):(W,X)->(Y,Z),(A,B,C):(U,V,W)->(X,Y,Z),(A,B,C,D):(S,T,U,V)->(W,X,Y,Z),(A,B,C,D,E):(Q,R,S,T,U)->(V,W,X,Y,Z),(A,B,C,D,E,F):(O,P,Q,R,S,T)->(U,V,W,X,Y,Z),(A,B,C,D,E,F,G):(M,N,O,P,Q,R,S)->(T,U,V,W,X,Y,Z),(A,B,C,D,E,F,G,H):(K,L,M,N,O,P,Q,R)->(S,T,U,V,W,X,Y,Z));
