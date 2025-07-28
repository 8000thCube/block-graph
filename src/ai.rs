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
impl AI<(Vec<f32>,Vec<f32>),f32> for SquaredError<()>{
	fn forward(&self,(output,target):(Vec<f32>,Vec<f32>))->f32{
		let (ol,tl)=(output.len(),target.len());
		assert!(ol==tl,"output len {ol} should match target len {tl}");

		output.into_iter().zip(target).map(|(o,t)|o-t).map(|x|x*x).sum::<f32>()/ol as f32
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
impl AI<(Vec<f32>,u32),f32> for CrossEntropyLayer{
	fn forward(&self,(output,target):(Vec<f32>,u32))->f32{
		let t=self.temperature;
		-if t.is_nan(){output[target as usize].ln()}else{LogSoftmaxLayer::new(0).with_temperature(t).forward_fixed(output)[target as usize]}
	}
}
impl AI<Vec<f32>,Vec<f32>> for AbnormalSoftmaxLayer{
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
impl AI<Vec<f32>,Vec<f32>> for ArgmaxLayer{
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
		let intermediate:Vec<f32>=input.into_iter().map(|x|((x-max)*t).exp()).inspect(|y|sum+=y).collect();
		let r=sum.recip();
		let output:Vec<f32>=intermediate.into_iter().map(|y|r*y).collect();
		output
	}
}
impl AI<Vec<f32>,f32> for MeanLayer{
	fn forward(&self,input:Vec<f32>)->f32{
		let sum:f32=input.iter().sum();

		sum/input.len() as f32
	}
}
impl AI<f32,f32> for MeanLayer{
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
impl Op for Identity{
	type Output=();
}
impl Op for SquaredError<()>{
	type Output=f32;
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
impl<A:AI<X,Y>+Op<Output=Y>,T,X,Y,Z> AI<(X,T),Z> for SquaredError<A> where SquaredError<()>:AI<(Y,T),Z>{
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
impl<A:AI<X,Y>,X,Y> AI<X,Y> for AccQ<A> where AccQ<()>:AI<Y,Y>{
	fn forward(&self,input:X)->Y{self.with_inner(()).forward(self.inner.forward(input))}
	fn forward_mut(&mut self,input:X)->Y{self.with_inner(()).forward(self.inner.forward_mut(input))}
}
impl<A:AI<X,Y>,X,Y> Op for SetType<A,X,Y>{
	type Output=Y;
}
impl<A:AI<X,Y>,X,Y> SetType<A,X,Y>{
	accessible_inner!(inner:A);
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
	fn compose((inner,gamma,dim):Self::Decomposition)->Self{
		Self{dim,gamma,inner:A::compose(inner)}
	}
	fn decompose(self)->Self::Decomposition{(self.inner.decompose(),self.gamma,self.dim)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.inner.decompose_cloned(),self.gamma,self.dim)}
	type Decomposition=(A::Decomposition,f32,i32);
}
impl<A:Decompose> Decompose for Branch<A>{
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
impl<A:Op<Output=Y>,Y> Op for &A{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for &mut A{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for AccQ<A> where AccQ<()>:AI<Y,Y>{
	type Output=Y;
}
impl<A:Op<Output=Y>,Y> Op for Branch<Vec<A>>{
	type Output=Vec<Y>;
}
impl<A:Op<Output=Y>,Y> Op for Duplicate<A>{
	type Output=(Y,Y);
}
impl<A:Op<Output=Y>,Y> Op for SquaredError<A> where SquaredError<()>:AI<(Y,Y),f32>{
	type Output=f32;
}
impl<A:Op<Output=Y>,Y> Op for Map<A>{
	type Output=Vec<Y>;
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
	pub fn dim(&self)->i32{self.dim}
	/// creates from the inner value
	pub fn from_inner(dim:i32,gamma:f32,inner:A)->Self{
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
impl<A> Map<A>{
	accessible_inner!(inner:A);
}
impl<A> Zip<A>{
	accessible_inner!(inner:A);
}
impl<F:Fn(X)->Y,X,Y> AI<X,Y> for Apply<F,X,Y>{
	fn forward(&self,input:X)->Y{(&self.inner)(input)}
}
impl<F:Fn(X)->Y,X,Y> Op for Apply<F,X,Y>{
	type Output=Y;
}
impl<K:Decompose+Eq+Hash,V:Decompose,S:Default+BuildHasher> Decompose for HashMap<K,V,S> where K::Decomposition:Ord{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.into_iter().map(Decompose::compose).collect()}
	fn decompose(self)->Self::Decomposition{
		let mut v:Vec<_>=self.into_iter().map(Decompose::decompose).collect();
		v.sort_unstable_by(|(k,_v),(k2,_v2)|k.cmp(k2));
		v
	}
	fn decompose_cloned(&self)->Self::Decomposition{
		let mut v:Vec<_>=self.iter().map(|(k,v)|(k.decompose_cloned(),v.decompose_cloned())).collect();
		v.sort_unstable_by(|(k,_v),(k2,_v2)|k.cmp(k2));
		v
	}
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


impl AI<Vec<f32>,f32> for SumLayer{
	fn forward(&self,input:Vec<f32>)->f32{input.into_iter().sum()}//TODO check dim
}

impl<E> AI<Vec<Vec<E>>,Vec<E>> for StackLayer{// TODO squeeze unsqueeze so we can properly implement this
	fn forward(&self,_input:Vec<Vec<E>>)->Vec<E>{todo!()}
}
impl<E> AI<Vec<Vec<E>>,Vec<Vec<E>>> for StackLayer{
	fn forward(&self,_input:Vec<Vec<E>>)->Vec<Vec<E>>{todo!()}
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
		#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
		/// layer to apply an operation
		pub struct $layer{dim:i32,mismatchbehavior:OnMismatch}
		#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
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
			/// gets the inner layer
			pub fn inner(&self)->&A{&self.inner}
			/// gets the inner layer
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// gets the inner layer
			pub fn into_inner(self)->A{self.inner}
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
				Self{dim:0,temperature:1.0}
			}
		}
		#[derive(Clone,Copy,Debug,PartialEq)]
		/// layer to apply an operation
		pub struct $layer{dim:i32,temperature:f32}
		#[derive(Clone,Copy,Debug,Default,PartialEq)]// TODO eq and hash that do something about the float
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
			pub fn new(dim:i32)->Self{Self::default().with_dim(dim)}
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
		impl<A> $wrap<A>{
			pub fn get_dim(&self)->i32{self.layer.dim}
			/// gets the temperature
			pub fn get_temperature(&self)->f32{self.layer.temperature}
			/// gets the inner layer
			pub fn inner(&self)->&A{&self.inner}
			/// gets the inner layer
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// gets the inner layer
			pub fn into_inner(self)->A{self.inner}
			/// creates a new layer
			pub fn new(dim:i32,inner:A)->Self where Self:Op{
				Self{inner,layer:$layer::new(dim)}
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
		impl<A:Op<Output=Y>,Y> Op for $wrap<A> where $layer:AI<Y,f32>{
			type Output=f32;
		}
		impl<A> $wrap<A>{
			/// gets the reduction mode
			pub fn get_reduction_mode(&self)->ReductionMode{self.layer.reductionmode}
			/// gets the inner layer
			pub fn inner(&self)->&A{&self.inner}
			/// gets the inner layer
			pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
			/// gets the inner layer
			pub fn into_inner(self)->A{self.inner}
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
			type Decomposition=(i32,usize);
		}
		impl Op for $layer{
			type Output=f32;
		}
		#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
		/// layer to apply an operation
		pub struct $layer{dim:i32,reductionmode:ReductionMode}
		#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
		/// wrapper to apply an operation
		pub struct $wrap<A>{inner:A,layer:$layer}
	}
}


impl Op for ChooseLayer{
	type Output=u32;
}
impl<A:Op<Output=Y>,Y> Op for Choose<A> where ChooseLayer:AI<Y,u32>{
	type Output=u32;
}
impl Op for CrossEntropyLayer{
	type Output=Vec<f32>;
}
impl<A:Op<Output=Y>,Y> Op for CrossEntropy<A> where CrossEntropyLayer:AI<(Y,Y),Vec<f32>>{
	type Output=Vec<f32>;
}
impl<A:AI<X,Y>+Op<Output=Y>,T,X,Y,Z> AI<(X,T),Z> for CrossEntropy<A> where CrossEntropyLayer:AI<(Y,T),Z>{
	fn forward(&self,(input,target):(X,T))->Z{self.layer.forward((self.inner.forward(input),target))}
	fn forward_mut(&mut self,(input,target):(X,T))->Z{self.layer.forward_mut((self.inner.forward_mut(input),target))}
}

cat_like!(CatLayer,Cat);
cat_like!(StackLayer,Stack);
soft_like!(@aiwrap @declare @decompose @impl ChooseLayer,Choose);
soft_like!(AbnormalSoftmaxLayer,AbnormalSoftmax);
soft_like!(ArgmaxLayer,Argmax);
soft_like!(@declare @decompose @impl CrossEntropyLayer,CrossEntropy);
soft_like!(LogSoftmaxLayer,LogSoftmax);
sum_like!(MeanLayer,Mean);
sum_like!(SumLayer,Sum);

// TODO dim should probably be i32 to match the general sizing of things and allow reverse indexing

#[derive(Clone,Copy,Debug,Default)]
/// accumulates cumulative
pub struct AccQ<A>{dim:i32,gamma:f32,inner:A}
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
/// module for cloning things
pub struct Duplicate<A>{inner:A,times:usize}
#[derive(Clone,Copy,Debug,Default)]
/// ai module for returning the input
pub struct Identity;
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for applying mean squared error loss
pub struct SquaredError<A>{inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// wraps to apply to every element of a vector
pub struct Map<A>{inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for applying ai modules sequentially
pub struct Sequential<A>{inner:A}
#[derive(Clone,Copy,Debug,Default)]
/// fixes the output type of a layer for a particular input type.
pub struct SetType<A,X,Y>{inner:A,phantom:PhantomData<fn(X)->Y>}
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
	fn abnormal_softmax(self,dim:i32)->AbnormalSoftmax<Self> where Self:Sized,AbnormalSoftmax<Self>:Op{AbnormalSoftmax::new(dim,self)}
	/// wraps with a accq operation
	fn acc_q(self,dim:i32,gamma:f32)->AccQ<Self> where AccQ<Self>:Op,Self:Sized{
		AccQ{dim,gamma,inner:self}
	}
	/// wraps with a branch operation
	fn branch(self)->Branch<Self> where Branch<Self>:Op,Self:Sized{
		Branch{inner:self}
	}
	/// wraps with a cat operation
	fn cat(self,dim:i32)->Cat<Self> where Cat<Self>:Op,Self:Sized{Cat::new(dim,self)}
	/// sequences with another ai operation
	fn chain<B>(self,b:B)->Sequential<(Self,B)> where Self:Sized,Sequential<(Self,B)>:Op{
		Sequential{inner:(self,b)}
	}
	/// wraps with a cross entropy operation
	fn cross_entropy(self,dim:i32)->CrossEntropy<Self> where CrossEntropy<Self>:Op,Self:Sized{CrossEntropy::new(dim,self)}
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
	fn log_softmax(self,dim:i32)->LogSoftmax<Self> where Self:Sized,LogSoftmax<Self>:Op{LogSoftmax::new(dim,self)}
	/// applies the operation to every output
	fn map<B>(self,b:B)->Map<Sequential<(Self,B)>> where Map<Sequential<(Self,B)>>:Op,Self:Sized,Sequential<(Self,B)>:Op{self.chain(b).to_each()}
	/// wraps with a mean operation
	fn mean(self)->Mean<Self> where Mean<Self>:Op,Self:Sized{Mean::new(self)}
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
	fn soft_choose(self,dim:i32)->Choose<Self> where Self:Sized,Choose<Self>:Op{Choose::new(dim,self)}
	/// wraps with a softmax operation
	fn softmax(self,dim:i32)->Argmax<Self> where Self:Sized,Argmax<Self>:Op{Argmax::new(dim,self)}
	/// wraps with a mse operation
	fn squared_error(self)->SquaredError<Self> where SquaredError<Self>:Op,Self:Sized{
		SquaredError{inner:self}
	}
	/// wraps with a map operation
	fn to_each(self)->Map<Self> where Map<Self>:Op,Self:Sized{
		Map{inner:self}
	}
	/// wraps with a sum operation
	fn sum(self)->Sum<Self> where Sum<Self>:Op,Self:Sized{Sum::new(self)}
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
use {accessible_inner,branch_tuple,cat_like,op_tuple,decompose_primitive,decompose_tuple,soft_like,sum_like,zip_tuple};
use std::{
	collections::HashMap,cmp::Ord,hash::{BuildHasher,Hash},iter::FromIterator,marker::PhantomData,ops::Range
};
zip_tuple!((A,B):(W,X)->(Y,Z),(A,B,C):(U,V,W)->(X,Y,Z),(A,B,C,D):(S,T,U,V)->(W,X,Y,Z),(A,B,C,D,E):(Q,R,S,T,U)->(V,W,X,Y,Z),(A,B,C,D,E,F):(O,P,Q,R,S,T)->(U,V,W,X,Y,Z),(A,B,C,D,E,F,G):(M,N,O,P,Q,R,S)->(T,U,V,W,X,Y,Z),(A,B,C,D,E,F,G,H):(K,L,M,N,O,P,Q,R)->(S,T,U,V,W,X,Y,Z));
