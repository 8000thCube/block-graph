bicop_like!(AddLayer,Add);
bicop_like!(MulLayer,Mul);
bicop_like!(SquaredErrorLayer,SquaredError);
cat_like!(CatLayer,Cat);
cat_like!(StackLayer,Stack);
impl AI<(Vec<f32>,Vec<f32>),Vec<f32>> for SquaredErrorLayer{// TODO some kind of namable map iter might make some of these compose better
	fn forward(&self,(output,target):(Vec<f32>,Vec<f32>))->Vec<f32>{
		let (ol,tl)=(output.len(),target.len());
		assert!(ol==tl,"output len {ol} should match target len {tl}");

		output.into_iter().zip(target).map(|(o,t)|o-t).map(|x|x*x).collect()
	}
}
impl AI<(Vec<f32>,Vec<f32>),f32> for CrossEntropyLayer{
	fn forward(&self,(_output,_target):(Vec<f32>,Vec<f32>))->f32{
		//let t=self.temperature;
		//-if t.i

		//-new().fix_type::<Vec<f32>>().log_softmax().forward_fixed(output).iter().zip(target.iter()).map(|(o,t)|o*t).fold(0.0,|acc,x|acc+x)
		todo!()
	}
}
impl AI<(Vec<f32>,Vec<f32>),f32> for SquaredErrorLayer{
	fn forward(&self,(output,target):(Vec<f32>,Vec<f32>))->f32{
		let (ol,tl)=(output.len(),target.len());
		assert!(ol==tl,"output len {ol} should match target len {tl}");

		output.into_iter().zip(target).map(|(o,t)|o-t).map(|x|x*x).sum::<f32>()/ol as f32
	}
}
impl AI<(Vec<f32>,u32),f32> for CrossEntropyLayer{
	fn forward(&self,(output,target):(Vec<f32>,u32))->f32{
		let t=self.temperature;
		-if t.is_nan(){output[target as usize].ln()}else{LogSoftmaxLayer::new(t).forward_fixed(output)[target as usize]}
	}
}
impl AI<Vec<f32>,Vec<f32>> for AbnormalSoftmaxLayer{
	fn forward(&self,input:Vec<f32>)->Vec<f32>{
		let max=input.iter().fold(f32::NEG_INFINITY,|x,&y|if x<y{y}else{x});
		input.into_iter().map(|x|if x==max{1.0}else{(x-max).exp()}).collect()
	}
}
impl AI<Vec<f32>,Vec<f32>> for AccQLayer{
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
impl AI<Vec<f32>,Vec<f32>> for SoftmaxLayer{
	fn forward(&self,input:Vec<f32>)->Vec<f32>{
		let t=self.temperature.recip();
		if t.is_nan(){
			let mut count=0;
			let max=input.iter().fold(f32::NEG_INFINITY,|x,&y|if x<y{
				count=0;
				y
			}else{
				if x==y{count+=1}
				x
			});
			let r=(count as f32).recip();
			return input.into_iter().map(|x|if x==max{r}else{0.0}).collect();
		}
		let max=input.iter().fold(f32::NEG_INFINITY,|x,&y|if x<y{y}else{x});
		let mut sum=0.0;
		let intermediate:Vec<f32>=input.into_iter().map(|x|if x==max{1.0}else{((x-max)*t).exp()}).inspect(|y|sum+=y).collect();
		let r=sum.recip();
		let output:Vec<f32>=intermediate.into_iter().map(|y|r*y).collect();
		output
	}
}
impl AI<Vec<f32>,Vec<f32>> for LogSoftmaxLayer{
	fn forward(&self,input:Vec<f32>)->Vec<f32>{
		let t=self.temperature.recip();
		let mut sum=0.0;
		input.iter().for_each(|x|sum+=(t*x).exp());
		let r=sum.ln();
		let output:Vec<f32>=input.into_iter().map(|x|t*x-r).collect();
		output
	}
}
impl AI<Vec<f32>,f32> for MeanLayer{
	fn forward(&self,input:Vec<f32>)->f32{
		let sum:f32=input.iter().sum();

		sum/input.len() as f32
	}
}
impl AI<Vec<f32>,f32> for SumLayer{
	fn forward(&self,input:Vec<f32>)->f32{input.into_iter().sum()}//TODO check dim
}
impl AI<f32,f32> for MeanLayer{
	fn forward(&self,input:f32)->f32{input}
}
impl AI<f32,f32> for SumLayer{
	fn forward(&self,input:f32)->f32{input}
}
impl Decompose for AccQLayer{
	fn compose((dim,gamma):Self::Decomposition)->Self{
		Self{dim,gamma}
	}
	fn decompose(self)->Self::Decomposition{(self.dim,self.gamma)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.dim,self.gamma)}
	type Decomposition=(i32,f32);
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
impl Op for AccQLayer{
	type Output=Vec<f32>;
}
impl Op for ChooseLayer{
	type Output=u32;
}
impl Op for CrossEntropyLayer{
	type Output=Vec<f32>;
}
impl<M:AI<M::Output,M::Output>+Op> IntoSequence<M> for AccQLayer where AccQLayer:Into<M>{
	fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
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
impl<A:AI<X,Y>+Op<Output=Y>,T,X,Y,Z> AI<(X,T),Z> for CrossEntropy<A> where CrossEntropyLayer:AI<(Y,T),Z>{
	fn forward(&self,(input,target):(X,T))->Z{self.layer.forward((self.inner.forward(input),target))}
	fn forward_mut(&mut self,(input,target):(X,T))->Z{self.layer.forward_mut((self.inner.forward_mut(input),target))}
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
impl<A:AI<X,Y>,X,Y> AI<X,Y> for AccQ<A> where AccQLayer:AI<Y,Y>{
	fn forward(&self,input:X)->Y{self.layer.forward(self.inner.forward(input))}
	fn forward_mut(&mut self,input:X)->Y{self.layer.forward(self.inner.forward_mut(input))}
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
impl<A:Decompose> Decompose for AccQ<A>{
	fn compose((inner,(dim,gamma)):Self::Decomposition)->Self{
		Self{inner:A::compose(inner),layer:AccQLayer::compose((dim,gamma))}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.layer.decompose())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.layer.decompose_cloned())}
	type Decomposition=(A::Decomposition,(i32,f32));
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
impl<A:Decompose> Decompose for Zip<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for AccQ<A> where AccQLayer:Into<M>{
	fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
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
impl<A:Op<Output=Y>,Y> Op for AccQ<A> where AccQLayer:AI<Y,Y>{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for Choose<A> where ChooseLayer:AI<Y,u32>{
	type Output=u32;
}
impl<A:Op<Output=Y>,Y> Op for CrossEntropy<A> where CrossEntropyLayer:AI<(Y,Y),Vec<f32>>{
	type Output=Vec<f32>;
}
impl<A:Op<Output=Y>,Y> Op for Duplicate<A>{
	type Output=(Y,Y);
}
impl<A:Op<Output=Y>,Y> Op for Map<A>{
	type Output=Vec<Y>;
}
impl<A:UnwrapInner> UnwrapInner for AccQ<A>{
	fn unwrap_inner(self)->Self::Inner{self.into_inner().unwrap_inner()}
	type Inner=A::Inner;
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
impl AccQLayer{
	/// creates from the inner value
	pub fn new(dim:i32,gamma:f32)->Self{
		AccQLayer{dim,gamma}
	}
	/// gets the dimension
	pub fn get_dim(&self)->i32{self.dim}
	/// gets the gamma
	pub fn get_gamma(&self)->f32{self.gamma}
}
impl<A> AccQ<A>{
	accessible_inner!(inner:A);
	/// gets the dimension
	pub fn get_dim(&self)->i32{self.layer.dim}
	/// creates from the inner value
	pub fn new(dim:i32,gamma:f32,inner:A)->Self{
		AccQ{inner,layer:AccQLayer::new(dim,gamma)}
	}
	/// gets the gamma
	pub fn get_gamma(&self)->f32{self.layer.gamma}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->AccQ<B> where AccQ<B>:Op{AccQ::new(self.get_dim(),self.get_gamma(),inner)}
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
impl<E> AI<Vec<Vec<E>>,Vec<E>> for StackLayer{// TODO squeeze unsqueeze so we can properly implement this
	fn forward(&self,_input:Vec<Vec<E>>)->Vec<E>{todo!()}
}
impl<E> AI<Vec<Vec<E>>,Vec<Vec<E>>> for StackLayer{
	fn forward(&self,_input:Vec<Vec<E>>)->Vec<Vec<E>>{todo!()}
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
impl<L:OpsAdd<R,Output=Y>,R,Y> AI<(L,R),Y> for AddLayer{
	fn forward(&self,(left,right):(L,R))->Y{left+right}
}
impl<L:OpsMul<R,Output=Y>,R,Y> AI<(L,R),Y> for MulLayer{
	fn forward(&self,(left,right):(L,R))->Y{left*right}
}
impl<X:OpsAbs<Output=Y>,Y> AI<X,Y> for AbsLayer{
	fn forward(&self,input:X)->Y{input.abs()}
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
/// declares layer and wrapper structs and implements accessor functions, decompose and op for binary componentwise operations. ai will still have to be externally implemented for the layer stuct
macro_rules! bicop_like{// TODO separate parts of this like in one of the other likes and make squared error specifically have vec output
	($layer:ident,$wrap:ident)=>{
		impl $layer{
			/// creates a new layer
			pub fn new()->Self{Self::default()}
		}
		impl<A:AI<X,L>+Op<Output=L>,L,R,X,Y> AI<(X,R),Y> for $wrap<A> where $layer:AI<(L,R),Y>{
			fn forward(&self,(input,right):(X,R))->Y{self.layer.forward((self.inner.forward(input),right))}// TODO swap operation
			fn forward_mut(&mut self,(input,right):(X,R))->Y{self.layer.forward_mut((self.inner.forward_mut(input),right))}
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose(inner:Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:Default::default()}
			}
			fn decompose(self)->Self::Decomposition{self.inner.decompose()}
			fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
			type Decomposition=A::Decomposition;
		}
		impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $wrap<A> where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
		}
		impl<A:UnwrapInner> UnwrapInner for $wrap<A>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<(Y,Y),Y>{
			type Output=Y;
		}
		impl<A> $wrap<A>{
			accessible_inner!(inner:A);
			/// creates a new layer
			pub fn new(inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new()}
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
		}
		impl<M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $layer where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
		}
		impl Decompose for $layer{
			fn compose(_decomposition:Self::Decomposition)->Self{Self::new()}
			fn decompose(self){}
			fn decompose_cloned(&self){}
			type Decomposition=();
		}
		impl Op for $layer{
			type Output=Vec<f32>;
		}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{seal:PhantomData<()>}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	}
}
/// declares layer and wrapper structs and implements accessor functions, decompose and op for reduction operations that have dim and mismatch behavior as configuration fields. ai will still have to be externally implemented for the layer stuct
macro_rules! cat_like{
	(@aiwrap $layer:ident,$wrap:ident)=>{
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
	};
	(@declare $layer:ident,$wrap:ident)=>{
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{dim:i32,mismatchbehavior:OnMismatch}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	};
	(@decompose $layer:ident,$wrap:ident)=>{
		impl Decompose for $layer{
			fn compose((dim,mismatchbehavior):Self::Decomposition)->Self{
				Self{dim,mismatchbehavior:OnMismatch::compose(mismatchbehavior)}
			}
			fn decompose(self)->Self::Decomposition{(self.dim,self.mismatchbehavior.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.dim,self.mismatchbehavior.decompose_cloned())}
			type Decomposition=(i32,usize);
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose((inner,layer):Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:$layer::compose(layer)}
			}
			fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.layer.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.layer.decompose_cloned())}
			type Decomposition=(A::Decomposition,<$layer as Decompose>::Decomposition);
		}
	};
	(@impl $layer:ident,$wrap:ident)=>{
		impl $layer{
			/// gets the dimension
			pub fn get_dim(&self)->i32{self.dim}
			/// gets the mismatch behavior
			pub fn get_mismatch_behavior(&self)->OnMismatch{self.mismatchbehavior}
			/// creates a new layer
			pub fn new(dim:i32)->Self{Self::default().with_dim(dim)}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:i32){self.dim=dim}
			/// sets the mismatch behavior
			pub fn set_mismatch_behavior(&mut self,behavior:OnMismatch){self.mismatchbehavior=behavior}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
				self.dim=dim;
				self
			}
			/// sets the mismatch behavior
			pub fn with_mismatch_behavior(mut self,behavior:OnMismatch)->Self{
				self.mismatchbehavior=behavior;
				self
			}
		}
		impl<A> $wrap<A>{
			pub fn get_dim(&self)->i32{self.layer.dim}
			/// gets the mismatch behavior
			pub fn get_mismatch_behavior(&self)->OnMismatch{self.layer.mismatchbehavior}
			accessible_inner!(inner:A);
			/// creates a new layer
			pub fn new(dim:i32,inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new(dim)}
			}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:i32){self.layer.dim=dim}
			/// sets the mismatch behavior
			pub fn set_mismatch_behavior(&mut self,behavior:OnMismatch){self.layer.mismatchbehavior=behavior}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
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
	};
	(@op $layer:ident,$wrap:ident)=>{
		impl Op for $layer{
			type Output=Vec<()>;
		}
		impl<A:Op<Output=Y>,Y:IntoIterator<Item=Z>,Z> Op for $wrap<A> where $layer:AI<Y,Z>{
			type Output=Z;
		}
	};
	($layer:ident,$wrap:ident)=>{
		cat_like!(@aiwrap @declare @decompose @impl @op $layer,$wrap);
	};
	($(@$command:tt)* $layer:ident,$wrap:ident)=>{
		$(cat_like!(@$command $layer,$wrap);)*
	};
}
/// declares layer and wrapper structs and implements accessor functions, decompose and op for reduction operations that have dim and temperature as configuration fields. ai will still have to be externally implemented for the layer stuct
macro_rules! soft_like{
	(@aiwrap $layer:ident,$wrap:ident)=>{
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
	};
	(@declare $layer:ident,$wrap:ident)=>{
		impl Default for $layer{
			fn default()->Self{
				Self{dim:-1,temperature:1.0}
			}
		}
		#[derive(Clone,Copy,Debug,Deserialize,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{dim:i32,temperature:f32}
		#[derive(Clone,Copy,Debug,Default,Deserialize,PartialEq,Serialize)]// TODO eq and hash that do something about the float
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	};
	(@decompose $layer:ident,$wrap:ident)=>{
		impl Decompose for $layer{
			fn compose((dim,temperature):Self::Decomposition)->Self{
				Self{dim,temperature}
			}
			fn decompose(self)->Self::Decomposition{(self.dim,self.temperature)}
			fn decompose_cloned(&self)->Self::Decomposition{(self.dim,self.temperature)}
			type Decomposition=(i32,f32);
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose((inner,layer):Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:$layer::compose(layer)}
			}
			fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.layer.decompose())}
			fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.layer.decompose_cloned())}
			type Decomposition=(A::Decomposition,<$layer as Decompose>::Decomposition);
		}
	};
	(@impl $layer:ident,$wrap:ident)=>{
		impl $layer{
			/// gets the dimension
			pub fn get_dim(&self)->i32{self.dim}
			/// gets the temperature
			pub fn get_temperature(&self)->f32{self.temperature}
			/// creates a new layer
			pub fn new(temperature:f32)->Self{
				Self{dim:-1,temperature}
			}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:i32){self.dim=dim}
			/// sets the mismatch behavior. A temperature of NaN will make the non soft version if possible. A finite temperature will make the soft version
			pub fn set_temperature(&mut self,temperature:f32){self.temperature=temperature}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
				self.dim=dim;
				self
			}
			/// sets the temperature. A temperature of NaN will make the non soft version if possible. A finite temperature will make the soft version
			pub fn with_temperature(mut self,temperature:f32)->Self{
				self.temperature=temperature;
				self
			}
		}
		impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $wrap<A> where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
		}
		impl<A:UnwrapInner> UnwrapInner for $wrap<A>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A> $wrap<A>{
			pub fn get_dim(&self)->i32{self.layer.dim}
			/// gets the temperature
			pub fn get_temperature(&self)->f32{self.layer.temperature}
			accessible_inner!(inner:A);
			/// creates a new layer
			pub fn new(inner:A,temperature:f32)->Self where Self:Op{
				Self{inner,layer:$layer::new(temperature)}
			}
			/// sets the dimension
			pub fn set_dim(&mut self,dim:i32){self.layer.dim=dim}
			/// sets the temperature
			pub fn set_temperature(&mut self,temperature:f32){self.layer.temperature=temperature}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
				self.layer.dim=dim;
				self
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
			/// sets the mismatch behavior
			pub fn with_temperature(mut self,temperature:f32)->Self{
				self.layer.temperature=temperature;
				self
			}
		}
	};
	(@op $layer:ident,$wrap:ident)=>{
		impl Op for $layer{
			type Output=Vec<f32>;
		}
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<Y,Vec<f32>>{
			type Output=Vec<f32>;
		}
	};
	($layer:ident,$wrap:ident)=>{
		soft_like!(@aiwrap @declare @decompose @impl @op $layer,$wrap);
	};
	($(@$command:tt)* $layer:ident,$wrap:ident)=>{
		$(soft_like!(@$command $layer,$wrap);)*
	};
}
/// declares layer and wrapper structs and implements accessor functions, decompose and op for reduction operations that have a reduction mode as configuration fields. ai will still have to be externally implemented for the layer stuct
macro_rules! sum_like{
	($layer:ident,$wrap:ident)=>{
		impl $layer{
			/// gets the reduction mode
			pub fn get_reduction_mode(&self)->ReductionMode{self.reductionmode}
			/// creates a new layer
			pub fn new()->Self{Self::default()}
			/// sets the reduction mode
			pub fn set_reduction_mode(&mut self,mode:ReductionMode){self.reductionmode=mode}
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
		impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $wrap<A> where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
		}
		impl<A:UnwrapInner> UnwrapInner for $wrap<A>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<Y,f32>{
			type Output=f32;
		}
		impl<A> $wrap<A>{
			/// gets the reduction mode
			pub fn get_reduction_mode(&self)->ReductionMode{self.layer.reductionmode}
			accessible_inner!(inner:A);
			/// creates a new layer
			pub fn new(inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new()}
			}
			/// sets the reduction mode
			pub fn set_reduction_mode(&mut self,mode:ReductionMode){self.layer.reductionmode=mode}
			/// sets the dimension
			pub fn with_dim(mut self,dim:i32)->Self{
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
			type Decomposition=(i32,u64);
		}
		impl Op for $layer{
			type Output=f32;
		}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{dim:i32,reductionmode:ReductionMode}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	}
}
/// declares layer and wrapper structs and implements accessor functions, decompose and op for unary componentwise operations. ai will still have to be externally implemented for the layer stuct
macro_rules! uncop_like{// TODO using op traits with output may better allow type shifting
	($layer:ident,$wrap:ident)=>{
		impl $layer{
			/// creates a new layer
			pub fn new()->Self{Self::default()}
		}
		impl<A:AI<X,Y>+Op<Output=Y>,X,Y,Z> AI<X,Z> for $wrap<A> where $layer:AI<Y,Z>{
			fn forward(&self,input:X)->Z{self.layer.forward(self.inner.forward(input))}// TODO swap operation
			fn forward_mut(&mut self,input:X)->Z{self.layer.forward_mut(self.inner.forward_mut(input))}
		}
		impl<A:Decompose> Decompose for $wrap<A>{
			fn compose(inner:Self::Decomposition)->Self{
				Self{inner:A::compose(inner),layer:Default::default()}
			}
			fn decompose(self)->Self::Decomposition{self.inner.decompose()}
			fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
			type Decomposition=A::Decomposition;
		}
		impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $wrap<A> where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{self.inner.into_sequence().with_next(self.layer)}
		}
		impl<A:UnwrapInner> UnwrapInner for $wrap<A>{
			fn unwrap_inner(self)->A::Inner{self.into_inner().unwrap_inner()}
			type Inner=A::Inner;
		}
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<Y,Y>{
			type Output=Y;
		}
		impl<A> $wrap<A>{
			accessible_inner!(inner:A);
			/// creates a new layer
			pub fn new(inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new()}
			}
			/// sets the inner module
			pub fn with_inner<B>(self,inner:B)->$wrap<B> where $wrap<B>:Op{
				$wrap{inner,layer:self.layer}
			}
		}
		impl<M:AI<M::Output,M::Output>+Op> IntoSequence<M> for $layer where $layer:Into<M>{
			fn into_sequence(self)->Sequential<Vec<M>>{vec![self.into()].sequential()}
		}
		impl Decompose for $layer{
			fn compose(_decomposition:Self::Decomposition)->Self{Self::new()}
			fn decompose(self){}
			fn decompose_cloned(&self){}
			type Decomposition=();
		}
		impl Op for $layer{
			type Output=Vec<f32>;
		}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// layer to apply an operation
		pub struct $layer{seal:PhantomData<()>}
		#[derive(Clone,Copy,Debug,Default,Deserialize,Eq,Hash,PartialEq,Serialize)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	}
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
		let op=().fix_type::<Vec<f32>>().acc_q(0,0.5);
		let x:Vec<f32>=vec![1.0,1.0,1.0,1.0,1.0];
		let y:Vec<f32>=op.forward(x);
		assert_eq!(y,[1.9375_f32,1.875,1.75,1.5,1.0]);
	}
	#[test]
	fn cat_vec(){
		let op=().fix_type::<Vec<Vec<f32>>>().cat(0);
		let x:Vec<Vec<f32>>=vec![vec![1.0,1.0,1.0,1.0,1.0],vec![2.0,2.0,2.0]];
		let y=op.forward(x);
		assert_eq!(y,[1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0]);
	}
	#[test]
	fn mse_vec(){
		let op=().fix_type::<Vec<f32>>().squared_error().mean();
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
impl<A:IntoSequence<M>,M:AI<M::Output,M::Output>+Op> IntoSequence<M> for Sequential<Vec<A>>{//TODO into sequence for tuple
	fn into_sequence(self)->Sequential<Vec<M>>{
		Sequential{inner:self.into_inner().into_iter().flat_map(|a|a.into_sequence().into_inner()).collect()}
	}
}
impl<A:AI<X,Y>+IntoSequence<M>,M:AI<M::Output,M::Output>+Op,X,Y> IntoSequence<M> for SetType<A,X,Y>{
	fn into_sequence(self)->Sequential<Vec<M>>{self.into_inner().into_sequence()}
}
impl<A:AI<X,Y>+UnwrapInner,X,Y> UnwrapInner for SetType<A,X,Y>{
	fn unwrap_inner(self)->Self::Inner{self.into_inner().unwrap_inner()}
	type Inner=A::Inner;
}

#[derive(Clone,Copy,Debug,Deserialize,Default,PartialEq,Serialize)]
/// accumulates cumulative
pub struct AccQ<A>{inner:A,layer:AccQLayer}
#[derive(Clone,Copy,Debug,Default,Deserialize,PartialEq,Serialize)]
/// accumulates cumulative
pub struct AccQLayer{dim:i32,gamma:f32}
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
soft_like!(@aiwrap @declare @decompose @impl ChooseLayer,Choose);
soft_like!(AbnormalSoftmaxLayer,AbnormalSoftmax);
soft_like!(SoftmaxLayer,Softmax);
soft_like!(@declare @decompose @impl CrossEntropyLayer,CrossEntropy);
soft_like!(LogSoftmaxLayer,LogSoftmax);
sum_like!(MeanLayer,Mean);
sum_like!(SumLayer,Sum);
uncop_like!(AbsLayer,Abs);
use {accessible_inner,bicop_like,cat_like,soft_like,sum_like,zip_tuple};
use crate::{
	AI,Decompose,IntoSequence,Op,UnwrapInner,ops::Abs as OpsAbs
};
use serde::{Deserialize,Serialize};
use std::{
	iter::FromIterator,marker::PhantomData,ops::{Add as OpsAdd,Mul as OpsMul}
};
zip_tuple!((A,B):(W,X)->(Y,Z),(A,B,C):(U,V,W)->(X,Y,Z),(A,B,C,D):(S,T,U,V)->(W,X,Y,Z),(A,B,C,D,E):(Q,R,S,T,U)->(V,W,X,Y,Z),(A,B,C,D,E,F):(O,P,Q,R,S,T)->(U,V,W,X,Y,Z),(A,B,C,D,E,F,G):(M,N,O,P,Q,R,S)->(T,U,V,W,X,Y,Z),(A,B,C,D,E,F,G,H):(K,L,M,N,O,P,Q,R)->(S,T,U,V,W,X,Y,Z));
