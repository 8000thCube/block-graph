fn accumulate_q_burn<B:Backend,const N:usize>(gamma:f32,i:Tensor<B,N>)->Tensor<B,N>{
	let i=i.split(1,N-1);
	let q=i.into_iter().rev().scan(None,|acc,i|{
		let q=Some(if let Some(a)=acc.take(){a*gamma+i}else{i});
		*acc=q.clone();
		q
	}).collect();
	Tensor::cat(q,N-1)
}
fn soft_choose_burn_1<B:Backend,const N:usize>(logits:Tensor<B,N>,temperature:f32)->u32{
	let distribution=softmax(logits/temperature,N-1).into_data();
	distribution.iter().scan(random(),|choice:&mut f32,weight:f32|Some(*choice-=weight).filter(|_|*choice>=0.0)).count() as u32
}
fn soft_choose_burn_multi<B:Backend,const N:usize>(logits:Tensor<B,N>,temperature:f32)->Vec<u32>{
	let chunk=logits.dims()[N-1];
	let distribution=softmax(logits/temperature,N-1).into_data().to_vec().unwrap();
	distribution.chunks_exact(chunk).map(|d|d.iter().scan(random(),|choice:&mut f32,weight:&f32|Some(*choice-=weight).filter(|_|*choice>=0.0)).count() as u32).collect()
}
fn soft_choose_burn_tensor<B:Backend,const N:usize>(logits:Tensor<B,N>,temperature:f32)->Tensor<B,N,Int>{
	let device=logits.device();
	let mut dims=logits.dims();

	dims[N-1]=1;
	Tensor::from_data(TensorData::new(soft_choose_burn_multi(logits,temperature),dims),&device)
}
impl Decompose for (){
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl Decompose for Dropout{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl Decompose for Relu{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl Op for (){
	type Output=();
}
impl Op for Dropout{
	type Output=();
}
impl Op for MseLoss{
	type Output=();
}
impl Op for Relu{
	type Output=();
}
impl<A:AI<V,W>+Op<Output=W>,B:AI<W,X>+Op<Output=X>,C:AI<X,Y>+Op<Output=Y>,D:AI<Y,Z>,V,W,X,Y,Z> AI<V,Z> for Sequential<(A,B,C,D)>{
	fn forward(&self,input:V)->Z{
		let Self((a,b,c,d))=self;
		d.forward(c.forward(b.forward(a.forward(input))))
	}
	fn forward_mut(&mut self,input:V)->Z{
		let Self((a,b,c,d))=self;
		d.forward(c.forward(b.forward_mut(a.forward_mut(input))))
	}
}
impl<A:AI<W,X>+Op<Output=X>,B:AI<X,Y>+Op<Output=Y>,C:AI<Y,Z>,W,X,Y,Z> AI<W,Z> for Sequential<(A,B,C)>{
	fn forward(&self,input:W)->Z{
		let Self((a,b,c))=self;
		c.forward(b.forward(a.forward(input)))
	}
	fn forward_mut(&mut self,input:W)->Z{
		let Self((a,b,c))=self;
		c.forward(b.forward_mut(a.forward_mut(input)))
	}
}
impl<A:AI<W,Z>+AI<X,Y>,W,X,Y,Z> AI<W,Z> for SetType<A,X,Y>{
	fn forward(&self,input:W)->Z{self.inner.forward(input)}
	fn forward_mut(&mut self,input:W)->Z{self.inner.forward_mut(input)}
}
impl<A:AI<X,Tensor<B,N>>+Op<Output=Tensor<B,N>>,B:Backend,X,const N:usize> AI<X,Tensor<B,N,Int>> for SoftChoose<A>{
	fn forward(&self,input:X)->Tensor<B,N,Int>{soft_choose_burn_tensor(self.layer.forward(input),self.temperature)}
	fn forward_mut(&mut self,input:X)->Tensor<B,N,Int>{soft_choose_burn_tensor(self.layer.forward_mut(input),self.temperature)}
}
impl<A:AI<X,Tensor<B,N>>+Op<Output=Tensor<B,N>>,B:Backend,X,const N:usize> AI<X,Tensor<B,N>> for AccQ<A>{
	fn forward(&self,input:X)->Tensor<B,N>{accumulate_q_burn(self.gamma,self.layer.forward(input))}
	fn forward_mut(&mut self,input:X)->Tensor<B,N>{accumulate_q_burn(self.gamma,self.layer.forward_mut(input))}
}
impl<A:AI<X,Tensor<B,N>>+Op<Output=Tensor<B,N>>,B:Backend,X,const N:usize> AI<X,Vec<u32>> for SoftChoose<A>{
	fn forward(&self,input:X)->Vec<u32>{soft_choose_burn_multi(self.layer.forward(input),self.temperature)}
	fn forward_mut(&mut self,input:X)->Vec<u32>{soft_choose_burn_multi(self.layer.forward_mut(input),self.temperature)}
}
impl<A:AI<X,Tensor<B,N>>+Op<Output=Tensor<B,N>>,B:Backend,X,const N:usize> AI<X,u32> for SoftChoose<A>{
	fn forward(&self,input:X)->u32{soft_choose_burn_1(self.layer.forward(input),self.temperature)}
	fn forward_mut(&mut self,input:X)->u32{soft_choose_burn_1(self.layer.forward_mut(input),self.temperature)}
}
impl<A:AI<X,Vec<Tensor<B,N,K>>>,B:Backend,K:BasicOps<B>+TensorKind<B>,X,const N:usize> AI<X,Tensor<B,N,K>> for Cat<A>{
	fn forward(&self,input:X)->Tensor<B,N,K>{Tensor::cat(self.layer.forward(input),self.dim)}
	fn forward_mut(&mut self,input:X)->Tensor<B,N,K>{Tensor::cat(self.layer.forward_mut(input),self.dim)}
}
impl<A:AI<X,X>+Op<Output=X>,X> Op for Sequential<Vec<A>>{
	type Output=X;
}
impl<A:AI<X,Y>+Op<Output=Y>,I:IntoIterator<Item=X>,J:FromIterator<Y>,X,Y> AI<I,J> for ToEach<A>{
	fn forward(&self,input:I)->J{input.into_iter().map(|x|self.0.forward(x)).collect()}
	fn forward_mut(&mut self,input:I)->J{input.into_iter().map(|x|self.0.forward_mut(x)).collect()}
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
	fn forward(&self,input:X)->X{self.0.iter().fold(input,|x,a|a.forward(x))}
	fn forward_mut(&mut self,input:X)->X{self.0.iter_mut().fold(input,|x,a|a.forward_mut(x))}
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
		let Self((a,b))=self;
		b.forward(a.forward(input))
	}
	fn forward_mut(&mut self,input:X)->Z{
		let Self((a,b))=self;
		b.forward_mut(a.forward_mut(input))
	}
}
impl<A:AI<X,Y>,X:Clone,Y> AI<X,Vec<Y>> for Branch<Vec<A>>{
	fn forward(&self,input:X)->Vec<Y>{
		let Self(a)=self;
		let mut y:Vec<Y>=a.iter().take(a.len().saturating_sub(1)).map(|a|a.forward(input.clone())).collect();
		if let Some(a)=a.last(){y.push(a.forward(input))}
		y
	}
	fn forward_mut(&mut self,input:X)->Vec<Y>{
		let Self(a)=self;
		let l=a.len().saturating_sub(1);
		let mut y:Vec<Y>=a.iter_mut().take(l).map(|a|a.forward_mut(input.clone())).collect();
		if let Some(a)=a.last_mut(){y.push(a.forward_mut(input))}
		y
	}
}
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y){
		let y=self.0.forward(input);
		(y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y){
		let y=self.0.forward_mut(input);
		(y.clone(),y)
	}
}
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y,Y){
		let y=self.0.forward(input);
		(y.clone(),y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y,Y){
		let y=self.0.forward_mut(input);
		(y.clone(),y.clone(),y)
	}
}
impl<A:AI<X,Y>,X,Y:Clone> AI<X,(Y,Y,Y,Y)> for Duplicate<A>{
	fn forward(&self,input:X)->(Y,Y,Y,Y){
		let y=self.0.forward(input);
		(y.clone(),y.clone(),y.clone(),y)
	}
	fn forward_mut(&mut self,input:X)->(Y,Y,Y,Y){
		let y=self.0.forward_mut(input);
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
impl<A:AI<X,Y>,X,Y> AI<X,Y> for Composite<A>{
	fn forward(&self,input:X)->Y{self.0.forward(input)}
	fn forward_mut(&mut self,input:X)->Y{self.0.forward_mut(input)}
}
impl<A:AI<X,Y>,X,Y> Op for SetType<A,X,Y>{
	type Output=Y;
}
impl<A:AutodiffBackend,B:Decompose+Send+Sync> AutodiffModule<A> for Composite<B> where B::Decomposition:AutodiffModule<A>{
	fn valid(&self)->Self::InnerModule{self.0.decompose_cloned().valid()}
	type InnerModule=<B::Decomposition as AutodiffModule<A>>::InnerModule;
}
impl<A:Copy+Decompose> Copy for Composite<A> where A::Decomposition:Clone{}
impl<A:Decompose+Send+Sync,B:Backend> Module<B> for Composite<A> where A::Decomposition:Module<B>{
	fn collect_devices(&self,devices:Vec<<B as Backend>::Device>)->Vec<<B as Backend>::Device>{self.0.decompose_cloned().collect_devices(devices)}
	fn devices(&self)->Vec<<B as Backend>::Device>{self.0.decompose_cloned().devices()}
	fn fork(self,device:&<B as Backend>::Device)->Self{Self(A::compose(self.0.decompose().fork(device)))}
	fn into_record(self)->Self::Record{self.0.decompose().into_record()}
	fn load_file<F:FileRecorder<B>,P:Into<PathBuf>>(self,filepath:P,recorder:&F,device:&<B as Backend>::Device)->Result<Self,RecorderError>{self.0.decompose().load_file(filepath,recorder,device).map(|a|Self(A::compose(a)))}
	fn load_record(self,record:Self::Record)->Self{Self(A::compose(self.0.decompose().load_record(record)))}
	fn map<Mapper:ModuleMapper<B>>(self,mapper:&mut Mapper)->Self{Self(A::compose(self.0.decompose().map(mapper)))}
	fn num_params(&self)->usize{self.0.decompose_cloned().num_params()}
	fn quantize_weights<C:Calibration>(self,quantizer:&mut Quantizer<C>)->Self{Self(A::compose(self.0.decompose().quantize_weights(quantizer)))}
	fn save_file<F:FileRecorder<B>,P:Into<PathBuf>>(self,filepath:P,recorder:&F)->Result<(),RecorderError>{self.0.decompose().save_file(filepath,recorder)}
	fn to_device(self,device:&<B as Backend>::Device)->Self{Self(A::compose(self.0.decompose().to_device(device)))}
	fn visit<Visitor:ModuleVisitor<B>>(&self,visitor:&mut Visitor){self.0.decompose_cloned().visit(visitor)}
	type Record=<A::Decomposition as Module<B>>::Record;
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
impl<A:Decompose> Clone for Composite<A> where A::Decomposition:Clone{
	fn clone(&self)->Self{Self(A::compose(self.0.decompose_cloned()))}
}
impl<A:Decompose> Debug for Composite<A> where A::Decomposition:Debug{
	fn fmt(&self,f:&mut Formatter<'_>)->Result<(),FmtError>{self.0.decompose_cloned().fmt(f)}
}
impl<A:Decompose> Decompose for AccQ<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{layer:A::compose(decomposition.0),gamma:decomposition.1}
	}
	fn decompose(self)->Self::Decomposition{(self.layer.decompose(),self.gamma)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.layer.decompose_cloned(),self.gamma)}
	type Decomposition=(A::Decomposition,f32);
}
impl<A:Decompose> Decompose for Branch<A>{
	fn compose(decomposition:Self::Decomposition)->Self{Self(A::compose(decomposition))}
	fn decompose(self)->Self::Decomposition{self.0.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.0.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Cat<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{layer:A::compose(decomposition.0),dim:decomposition.1}
	}
	fn decompose(self)->Self::Decomposition{(self.layer.decompose(),self.dim)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.layer.decompose_cloned(),self.dim)}
	type Decomposition=(A::Decomposition,usize);
}
impl<A:Decompose> Decompose for Duplicate<A>{
	fn compose(decomposition:Self::Decomposition)->Self{Self(A::compose(decomposition))}
	fn decompose(self)->Self::Decomposition{self.0.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.0.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Option<A>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.map(A::compose)}
	fn decompose(self)->Self::Decomposition{self.map(A::decompose)}
	fn decompose_cloned(&self)->Self::Decomposition{self.as_ref().map(A::decompose_cloned)}
	type Decomposition=Option<A::Decomposition>;
}
impl<A:Decompose> Decompose for Sequential<A>{
	fn compose(decomposition:Self::Decomposition)->Self{Self(A::compose(decomposition))}
	fn decompose(self)->Self::Decomposition{self.0.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.0.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for SoftChoose<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{layer:A::compose(decomposition.0),temperature:decomposition.1}
	}
	fn decompose(self)->Self::Decomposition{(self.layer.decompose(),self.temperature)}
	fn decompose_cloned(&self)->Self::Decomposition{(self.layer.decompose_cloned(),self.temperature)}
	type Decomposition=(A::Decomposition,f32);
}
impl<A:Decompose> Decompose for ToEach<A>{
	fn compose(decomposition:Self::Decomposition)->Self{Self(A::compose(decomposition))}
	fn decompose(self)->Self::Decomposition{self.0.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.0.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Vec<A>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition.into_iter().map(A::compose).collect()}
	fn decompose(self)->Self::Decomposition{self.into_iter().map(A::decompose).collect()}
	fn decompose_cloned(&self)->Self::Decomposition{self.iter().map(A::decompose_cloned).collect()}
	type Decomposition=Vec<A::Decomposition>;
}
impl<A:Op<Output=W>,B:AI<W,X>+Op<Output=X>,C:AI<X,Y>+Op<Output=Y>,D:AI<Y,Z>+Op<Output=Z>,W,X,Y,Z> Op for Sequential<(A,B,C,D)>{
	type Output=Z;
}
impl<A:Op<Output=X>,B:AI<X,Y>+Op<Output=Y>,C:AI<Y,Z>+Op<Output=Z>,X,Y,Z> Op for Sequential<(A,B,C)>{
	type Output=Z;
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
impl<A,B> Op for (A,B){
	type Output=();
}
impl<A,B,C> Op for (A,B,C){
	type Output=();
}
impl<A,B,C,D> Op for (A,B,C,D){
	type Output=();
}
impl<A:AI<X,X>+Op<Output=X>,X> Op for Option<A>{
	type Output=X;
}
impl<A:Op> Op for &A{
	type Output=A::Output;
}
impl<A:Op> Op for &mut A{
	type Output=A::Output;
}
impl<A> Op for AccQ<A>{
	type Output=();
}
impl<A> Op for Cat<A>{
	type Output=();
}
impl<A> Op for SoftChoose<A>{
	type Output=();
}
impl<A> Op for Vec<A>{
	type Output=();
}
impl<B:Backend,K:TensorKind<B>,const N:usize> Decompose for Tensor<B,N,K>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend,S:?Sized+AsRef<str>> From<&S> for BurnValue<B>{
	fn from(value:&S)->Self{Self::Incompatible(value.as_ref().to_string())}
}
impl<B:Backend,const N:usize> AI<(Tensor<B,N>,Tensor<B,N>),Tensor<B,1>> for MseLoss{
	fn forward(&self,(output,target):(Tensor<B,N>,Tensor<B,N>))->Tensor<B,1>{self.forward_no_reduction(output,target).mean()}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,Tensor<B,N>> for AccQ<()>{
	fn forward(&self,input:Tensor<B,N>)->Tensor<B,N>{accumulate_q_burn(self.gamma,input)}
	fn forward_mut(&mut self,input:Tensor<B,N>)->Tensor<B,N>{accumulate_q_burn(self.gamma,input)}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,Tensor<B,N>> for Dropout{
	fn forward(&self,input:Tensor<B,N>)->Tensor<B,N>{Dropout::forward(self,input)}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,Tensor<B,N>> for LayerNorm<B>{
	fn forward(&self,input:Tensor<B,N>)->Tensor<B,N>{LayerNorm::forward(self,input)}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,Tensor<B,N>> for Linear<B>{
	fn forward(&self,input:Tensor<B,N>)->Tensor<B,N>{Linear::forward(self,input)}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,Tensor<B,N>> for Relu{
	fn forward(&self,input:Tensor<B,N>)->Tensor<B,N>{Relu::forward(self,input)}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,Tensor<B,N,Int>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,N>)->Tensor<B,N,Int>{().set_type().soft_choose(self.temperature).forward(input)}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,Vec<u32>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,N>)->Vec<u32>{().set_type().soft_choose(self.temperature).forward(input)}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,u32> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,N>)->u32{().set_type().soft_choose(self.temperature).forward(input)}
}
impl<B:Backend> AI<BurnValue<B>,BurnValue<B>> for BurnLayer<B>{
	fn forward(&self,input:BurnValue<B>)->BurnValue<B>{
		match self{BurnLayer::Dropout(a)=>a.fix_type::<BurnValue<B>>().forward(input),BurnLayer::Embedding(a)=>a.fix_type::<BurnValue<B>>().forward(input),BurnLayer::LayerNorm(a)=>a.fix_type::<BurnValue<B>>().forward(input),BurnLayer::Linear(a)=>a.fix_type::<BurnValue<B>>().forward(input),BurnLayer::Mse(a)=>a.fix_type::<BurnValue<B>>().forward(input),BurnLayer::Relu(a)=>a.fix_type::<BurnValue<B>>().forward(input)}
	}
}
impl<B:Backend> AI<BurnValue<B>,BurnValue<B>> for Dropout{
	fn forward(&self,input:BurnValue<B>)->BurnValue<B>{
		match input{
			BurnValue::F1(x)=>self.forward(x).into(),BurnValue::F2(x)=>self.forward(x).into(),BurnValue::F3(x)=>self.forward(x).into(),BurnValue::F4(x)=>self.forward(x).into(),BurnValue::F5(x)=>self.forward(x).into(),BurnValue::F6(x)=>self.forward(x).into(),BurnValue::F7(x)=>self.forward(x).into(),BurnValue::F8(x)=>self.forward(x).into(),BurnValue::Incompatible(x)=>x.into(),BurnValue::Multi(x)=>x.into_iter().map(|x|self.fix_type::<BurnValue<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"dropout is only available for floats".into()
		}
	}
}
impl<B:Backend> AI<BurnValue<B>,BurnValue<B>> for Embedding<B>{
	fn forward(&self,input:BurnValue<B>)->BurnValue<B>{
		fn apply_embed<B:Backend,const N:usize,const K:usize>(this:&Embedding<B>,x:Tensor<B,N,Int>)->Tensor<B,K>{
			let dims=x.dims();
			let [batch,seq]=[dims[0],dims.iter().skip(1).product()];
			let x=x.reshape([batch,seq]);
			let y=this.forward(x);
			let embed=y.dims().last().copied().unwrap();
			let mut ydims=[0;K];
			ydims[..N].copy_from_slice(&dims);
			ydims[N]=embed;
			y.reshape(ydims)
		}
		fn apply_linear<B:Backend,const N:usize>(this:&Embedding<B>,x:Tensor<B,N>)->Tensor<B,N>{
			Linear{bias:None,weight:this.weight.clone()}.forward(x)
		}
		match input{
			BurnValue::F1(x)=>apply_linear(self,x).into(),BurnValue::F2(x)=>apply_linear(self,x).into(),BurnValue::F3(x)=>apply_linear(self,x).into(),BurnValue::F4(x)=>apply_linear(self,x).into(),BurnValue::F5(x)=>apply_linear(self,x).into(),BurnValue::F6(x)=>apply_linear(self,x).into(),BurnValue::F7(x)=>apply_linear(self,x).into(),BurnValue::F8(x)=>apply_linear(self,x).into(),BurnValue::I1(x)=>apply_embed::<B,1,2>(self,x).into(),BurnValue::I2(x)=>apply_embed::<B,2,3>(self,x).into(),BurnValue::I3(x)=>apply_embed::<B,3,4>(self,x).into(),BurnValue::I4(x)=>apply_embed::<B,4,5>(self,x).into(),BurnValue::I5(x)=>apply_embed::<B,5,6>(self,x).into(),BurnValue::I6(x)=>apply_embed::<B,6,7>(self,x).into(),BurnValue::I7(x)=>apply_embed::<B,7,8>(self,x).into(),BurnValue::I8(_x)=>"embedding output would exceed maximum supported rank".into(),BurnValue::Incompatible(x)=>x.into(),BurnValue::Multi(x)=>x.into_iter().map(|x|self.fix_type::<BurnValue<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"embedding is only available for float or int inputs".into()
		}
	}
}
impl<B:Backend> AI<BurnValue<B>,BurnValue<B>> for LayerNorm<B>{
	fn forward(&self,input:BurnValue<B>)->BurnValue<B>{
		match input{BurnValue::F1(x)=>self.forward(x).into(),BurnValue::F2(x)=>self.forward(x).into(),BurnValue::F3(x)=>self.forward(x).into(),BurnValue::F4(x)=>self.forward(x).into(),BurnValue::F5(x)=>self.forward(x).into(),BurnValue::F6(x)=>self.forward(x).into(),BurnValue::F7(x)=>self.forward(x).into(),BurnValue::F8(x)=>self.forward(x).into(),BurnValue::I1(x)=>self.forward(x.float()).into(),BurnValue::I2(x)=>self.forward(x.float()).into(),BurnValue::I3(x)=>self.forward(x.float()).into(),BurnValue::I4(x)=>self.forward(x.float()).into(),BurnValue::I5(x)=>self.forward(x.float()).into(),BurnValue::I6(x)=>self.forward(x.float()).into(),BurnValue::I7(x)=>self.forward(x.float()).into(),BurnValue::I8(x)=>self.forward(x.float()).into(),BurnValue::Incompatible(x)=>x.into(),BurnValue::Multi(x)=>x.into_iter().map(|x|self.fix_type::<BurnValue<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"layer norm is only supported for numeric inputs".into()}
	}
}
impl<B:Backend> AI<BurnValue<B>,BurnValue<B>> for Linear<B>{
	fn forward(&self,input:BurnValue<B>)->BurnValue<B>{
		match input{BurnValue::F1(x)=>self.forward(x).into(),BurnValue::F2(x)=>self.forward(x).into(),BurnValue::F3(x)=>self.forward(x).into(),BurnValue::F4(x)=>self.forward(x).into(),BurnValue::F5(x)=>self.forward(x).into(),BurnValue::F6(x)=>self.forward(x).into(),BurnValue::F7(x)=>self.forward(x).into(),BurnValue::F8(x)=>self.forward(x).into(),BurnValue::I1(x)=>self.forward(x.float()).into(),BurnValue::I2(x)=>self.forward(x.float()).into(),BurnValue::I3(x)=>self.forward(x.float()).into(),BurnValue::I4(x)=>self.forward(x.float()).into(),BurnValue::I5(x)=>self.forward(x.float()).into(),BurnValue::I6(x)=>self.forward(x.float()).into(),BurnValue::I7(x)=>self.forward(x.float()).into(),BurnValue::I8(x)=>self.forward(x.float()).into(),BurnValue::Incompatible(x)=>x.into(),BurnValue::Multi(x)=>x.into_iter().map(|x|self.fix_type::<BurnValue<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"linear is only supported for numeric inputs".into()}
	}
}
impl<B:Backend> AI<BurnValue<B>,BurnValue<B>> for MseLoss{
	fn forward(&self,input:BurnValue<B>)->BurnValue<B>{
		match input{
			BurnValue::Incompatible(x)=>x.into(),
			BurnValue::Multi(x)=>if x.len()==2{
				let mut x=x.into_iter();
				match (x.next().unwrap(),x.next().unwrap()){(BurnValue::F1(x0),BurnValue::F1(x1))=>AI::forward(self,(x0,x1)).into(),(BurnValue::F2(x0),BurnValue::F2(x1))=>AI::forward(self,(x0,x1)).into(),(BurnValue::F3(x0),BurnValue::F3(x1))=>AI::forward(self,(x0,x1)).into(),(BurnValue::F4(x0),BurnValue::F4(x1))=>AI::forward(self,(x0,x1)).into(),(BurnValue::F5(x0),BurnValue::F5(x1))=>AI::forward(self,(x0,x1)).into(),(BurnValue::F6(x0),BurnValue::F6(x1))=>AI::forward(self,(x0,x1)).into(),(BurnValue::F7(x0),BurnValue::F7(x1))=>AI::forward(self,(x0,x1)).into(),(BurnValue::F8(x0),BurnValue::F8(x1))=>AI::forward(self,(x0,x1)).into(),_=>"mse loss requires input pairs to be float tensors with the same rank".into()}
			}else{
				let y:Vec<BurnValue<B>>=x.into_iter().map(|x|self.fix_type::<BurnValue<B>>().forward(x)).collect();
				y.into()
			},
			_=>"mse loss requires inputs to be in pairs".into()
		}
	}
}
impl<B:Backend> AI<BurnValue<B>,BurnValue<B>> for Relu{
	fn forward(&self,input:BurnValue<B>)->BurnValue<B>{
		match input{BurnValue::F1(x)=>self.forward(x).into(),BurnValue::F2(x)=>self.forward(x).into(),BurnValue::F3(x)=>self.forward(x).into(),BurnValue::F4(x)=>self.forward(x).into(),BurnValue::F5(x)=>self.forward(x).into(),BurnValue::F6(x)=>self.forward(x).into(),BurnValue::F7(x)=>self.forward(x).into(),BurnValue::F8(x)=>self.forward(x).into(),BurnValue::I1(x)=>self.forward(x.float()).into(),BurnValue::I2(x)=>self.forward(x.float()).into(),BurnValue::I3(x)=>self.forward(x.float()).into(),BurnValue::I4(x)=>self.forward(x.float()).into(),BurnValue::I5(x)=>self.forward(x.float()).into(),BurnValue::I6(x)=>self.forward(x.float()).into(),BurnValue::I7(x)=>self.forward(x.float()).into(),BurnValue::I8(x)=>self.forward(x.float()).into(),BurnValue::Incompatible(x)=>x.into(),BurnValue::Multi(x)=>x.into_iter().map(|x|self.fix_type::<BurnValue<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"relu is only supported for numeric inputs".into()}
	}
}
impl<B:Backend> AI<Tensor<B,2,Int>,Tensor<B,3>> for Embedding<B>{
	fn forward(&self,input:Tensor<B,2,Int>)->Tensor<B,3>{Embedding::forward(self,input)}
}
impl<B:Backend> BurnLayer<B>{
	/// creates a linear layer
	pub fn linear(bias:bool,input:usize,output:usize,wscale:f32)->Self{
		let mut l=LinearConfig::new(input,output).with_bias(bias).init(&Default::default());
		l.bias=l.bias.map(|b|b.map(|b|b*wscale));
		l.weight=l.weight.map(|w|w*wscale);
		Self::Linear(l)
	}
	/// creates a relu layer
	pub fn relu()->Self{Self::Relu(Relu)}
}
impl<B:Backend> Decompose for BurnLayer<B>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend> Decompose for BurnValue<B>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend> Decompose for Embedding<B>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend> Decompose for LayerNorm<B>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend> Decompose for Linear<B>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend> Default for BurnValue<B>{
	fn default()->Self{Self::Multi(Vec::new())}
}
impl<B:Backend> From<String> for BurnValue<B>{
	fn from(value:String)->Self{Self::Incompatible(value)}
}
impl<B:Backend> From<Tensor<B,1,Bool>> for BurnValue<B>{
	fn from(value:Tensor<B,1,Bool>)->Self{Self::B1(value)}
}
impl<B:Backend> From<Tensor<B,2,Bool>> for BurnValue<B>{
	fn from(value:Tensor<B,2,Bool>)->Self{Self::B2(value)}
}
impl<B:Backend> From<Tensor<B,3,Bool>> for BurnValue<B>{
	fn from(value:Tensor<B,3,Bool>)->Self{Self::B3(value)}
}
impl<B:Backend> From<Tensor<B,4,Bool>> for BurnValue<B>{
	fn from(value:Tensor<B,4,Bool>)->Self{Self::B4(value)}
}
impl<B:Backend> From<Tensor<B,5,Bool>> for BurnValue<B>{
	fn from(value:Tensor<B,5,Bool>)->Self{Self::B5(value)}
}
impl<B:Backend> From<Tensor<B,6,Bool>> for BurnValue<B>{
	fn from(value:Tensor<B,6,Bool>)->Self{Self::B6(value)}
}
impl<B:Backend> From<Tensor<B,7,Bool>> for BurnValue<B>{
	fn from(value:Tensor<B,7,Bool>)->Self{Self::B7(value)}
}
impl<B:Backend> From<Tensor<B,8,Bool>> for BurnValue<B>{
	fn from(value:Tensor<B,8,Bool>)->Self{Self::B8(value)}
}
impl<B:Backend> From<Tensor<B,1,Float>> for BurnValue<B>{
	fn from(value:Tensor<B,1,Float>)->Self{Self::F1(value)}
}
impl<B:Backend> From<Tensor<B,2,Float>> for BurnValue<B>{
	fn from(value:Tensor<B,2,Float>)->Self{Self::F2(value)}
}
impl<B:Backend> From<Tensor<B,3,Float>> for BurnValue<B>{
	fn from(value:Tensor<B,3,Float>)->Self{Self::F3(value)}
}
impl<B:Backend> From<Tensor<B,4,Float>> for BurnValue<B>{
	fn from(value:Tensor<B,4,Float>)->Self{Self::F4(value)}
}
impl<B:Backend> From<Tensor<B,5,Float>> for BurnValue<B>{
	fn from(value:Tensor<B,5,Float>)->Self{Self::F5(value)}
}
impl<B:Backend> From<Tensor<B,6,Float>> for BurnValue<B>{
	fn from(value:Tensor<B,6,Float>)->Self{Self::F6(value)}
}
impl<B:Backend> From<Tensor<B,7,Float>> for BurnValue<B>{
	fn from(value:Tensor<B,7,Float>)->Self{Self::F7(value)}
}
impl<B:Backend> From<Tensor<B,8,Float>> for BurnValue<B>{
	fn from(value:Tensor<B,8,Float>)->Self{Self::F8(value)}
}
impl<B:Backend> From<Tensor<B,1,Int>> for BurnValue<B>{
	fn from(value:Tensor<B,1,Int>)->Self{Self::I1(value)}
}
impl<B:Backend> From<Tensor<B,2,Int>> for BurnValue<B>{
	fn from(value:Tensor<B,2,Int>)->Self{Self::I2(value)}
}
impl<B:Backend> From<Tensor<B,3,Int>> for BurnValue<B>{
	fn from(value:Tensor<B,3,Int>)->Self{Self::I3(value)}
}
impl<B:Backend> From<Tensor<B,4,Int>> for BurnValue<B>{
	fn from(value:Tensor<B,4,Int>)->Self{Self::I4(value)}
}
impl<B:Backend> From<Tensor<B,5,Int>> for BurnValue<B>{
	fn from(value:Tensor<B,5,Int>)->Self{Self::I5(value)}
}
impl<B:Backend> From<Tensor<B,6,Int>> for BurnValue<B>{
	fn from(value:Tensor<B,6,Int>)->Self{Self::I6(value)}
}
impl<B:Backend> From<Tensor<B,7,Int>> for BurnValue<B>{
	fn from(value:Tensor<B,7,Int>)->Self{Self::I7(value)}
}
impl<B:Backend> From<Tensor<B,8,Int>> for BurnValue<B>{
	fn from(value:Tensor<B,8,Int>)->Self{Self::I8(value)}
}
impl<B:Backend> From<Vec<BurnValue<B>>> for BurnValue<B>{
	fn from(value:Vec<BurnValue<B>>)->Self{Self::Multi(value)}
}
impl<B:Backend> Op for BurnLayer<B>{
	type Output=BurnValue<B>;
}
impl<B:Backend> Op for Embedding<B>{
	type Output=Tensor<B,3>;
}
impl<B:Backend> Op for LayerNorm<B>{
	type Output=Tensor<B,1>;
}
impl<B:Backend> Op for Linear<B>{
	type Output=Tensor<B,1>;
}
impl<X> AI<X,X> for (){
	fn forward(&self,input:X)->X{input}
}
#[derive(Debug,Module)] //TODO more layers
/// enumerates some burn layers
pub enum BurnLayer<B:Backend>{Dropout(Dropout),Embedding(Embedding<B>),LayerNorm(LayerNorm<B>),Linear(Linear<B>),Mse(MseLoss),Relu(Relu)}
#[derive(Clone,Debug)]//TODO implement module for this
/// enumerates burn tensors up to 8 dimensions
pub enum BurnValue<B:Backend>{B1(Tensor<B,1,Bool>),B2(Tensor<B,2,Bool>),B3(Tensor<B,3,Bool>),B4(Tensor<B,4,Bool>),B5(Tensor<B,5,Bool>),B6(Tensor<B,6,Bool>),B7(Tensor<B,7,Bool>),B8(Tensor<B,8,Bool>),F1(Tensor<B,1,Float>),F2(Tensor<B,2,Float>),F3(Tensor<B,3,Float>),F4(Tensor<B,4,Float>),F5(Tensor<B,5,Float>),F6(Tensor<B,6,Float>),F7(Tensor<B,7,Float>),F8(Tensor<B,8,Float>),I1(Tensor<B,1,Int>),I2(Tensor<B,2,Int>),I3(Tensor<B,3,Int>),I4(Tensor<B,4,Int>),I5(Tensor<B,5,Int>),I6(Tensor<B,6,Int>),I7(Tensor<B,7,Int>),I8(Tensor<B,8,Int>),Incompatible(String),Multi(Vec<Self>)}
#[derive(Clone,Copy,Debug,Default)]
/// accumulates cumulative
pub struct AccQ<A>{layer:A,gamma:f32}
#[derive(Clone,Copy,Debug,Default)]
/// autoregressive inference
pub struct Autoregression<A,X>{ai:A,state:Option<X>}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// wrapper for applying ai modules to the same input
pub struct Branch<A>(pub A);
#[derive(Clone,Copy,Debug,Default,Eq,Hash,PartialEq)]
/// wrapper for concatenating tensors in the output
pub struct Cat<A>{dim:usize,layer:A}
#[repr(transparent)]
/// wrapper for implementing decomposed traits on composite AI structures
pub struct Composite<A>(pub A);
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// module for cloning things
pub struct Duplicate<A>(pub A);//TODO replicate that has a number and makes a vec
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// wrapper for applying ai modules sequentially
pub struct Sequential<A>(pub A);
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// fixes the output type of a layer for a particular input type.
pub struct SetType<A:AI<X,Y>,X,Y>{inner:A,phantom:PhantomData<fn(X)->Y>}
#[derive(Clone,Copy,Debug,Default)]
/// chooses from the softmax
pub struct SoftChoose<A>{layer:A,temperature:f32}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// wraps to apply to every element of a vector
pub struct ToEach<A>(pub A);
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
		AccQ{layer:self,gamma}
	}
	/// wraps with a branch operation
	fn branch(self)->Branch<Self> where Branch<Self>:Op,Self:Sized{Branch(self)}
	/// wraps with a cat operation
	fn cat(self,dim:usize)->Cat<Self> where Cat<Self>:Op,Self:Sized{
		Cat{layer:self,dim}
	}
	/// wraps in a composite
	fn composite(self)->Composite<Self> where Self:Sized{Composite(self)}
	/// wraps with a duplicate operation
	fn duplicate(self)->Duplicate<Self> where Duplicate<Self>:Op,Self:Sized{Duplicate(self)}
	/// set type but with the same input and output
	fn fix_type<Z>(self)->SetType<Self,Z,Z> where Self:AI<Z,Z>+Sized{self.set_type()}
	/// creates an autoregressive inference
	fn infer_autoregressive<X,Y>(self,input:X)->Autoregression<Self,Y> where Self:AI<X,Y>+AI<Y,Y>+Sized,Y:Clone{
		let mut ai=self;
		let state=Some(ai.forward_mut(input));
		Autoregression{ai,state}
	}
	/// creates an optional operation
	fn optional(self)->Option<Self> where Self:Sized{Some(self)}
	/// produces a sequential module
	fn sequential(self)->Sequential<Self> where Sequential<Self>:Op,Self:Sized{Sequential(self)}
	/// sets the input output types
	fn set_type<W,Z>(self)->SetType<Self,W,Z> where Self:AI<W,Z>+Sized{
		SetType{inner:self,phantom:PhantomData}
	}
	/// wraps with a choose operation
	fn soft_choose(self,temperature:f32)->SoftChoose<Self> where Self:Sized,SoftChoose<Self>:Op{
		SoftChoose{layer:self,temperature}
	}
	/// suggested output type to help with composition coherence. Ideally, Self should implement AI<X,Self::Output> for some X
	type Output;
}
/// trait to represent supervised learning
pub trait Supervised<X,Y>{
	/// applies a training step, returning the loss
	fn train_step(&mut self,input:X,target:Y)->f32;
	/// applies a validation step, returning the loss
	fn val_step(&self,input:X,target:Y)->f32;
}
use burn::{
	module::{AutodiffModule,ModuleMapper,ModuleVisitor,Quantizer},
	nn::{
		Dropout,Embedding,LayerNorm,Linear,LinearConfig,Relu,loss::MseLoss
	},
	prelude::*,
	record::{FileRecorder,RecorderError},
	tensor::{BasicOps,TensorKind,activation::softmax,backend::AutodiffBackend,quantization::Calibration}
};
use rand::random;
use std::{
	fmt::{Debug,Error as FmtError,Formatter},iter::FromIterator,marker::PhantomData,path::PathBuf
};
