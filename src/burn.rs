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
impl MetricsRenderer for DontRender{
	fn update_train(&mut self,_state:MetricState){}
	fn update_valid(&mut self,_state:MetricState){}
	fn render_train(&mut self,_item:TrainingProgress){}
	fn render_valid(&mut self,_item:TrainingProgress){}
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
impl<A:AI<X,Vec<Tensor<B,N,K>>>,B:Backend,K:BasicOps<B>+TensorKind<B>,X,const N:usize> AI<X,Vec<Tensor<B,N,K>>> for TruncateToMatch<A>{
	fn forward(&self,input:X)->Vec<Tensor<B,N,K>>{
		let input=self.inner().forward(input);
		let dims=input.iter().map(|i|i.dims()).reduce(|d,mut e|{
			d.iter().zip(e.iter_mut()).for_each(|(d,e)|*e=*d.min(e));
			e
		});
		if let Some(d)=dims{
			let ranges=d.map(|x|0..x);
			input.into_iter().map(|x|x.slice(ranges.clone())).collect()
		}else{
			input
		}
	}
	fn forward_mut(&mut self,input:X)->Vec<Tensor<B,N,K>>{
		let input=self.inner_mut().forward_mut(input);
		let dims=input.iter().map(|i|i.dims()).reduce(|d,mut e|{
			d.iter().zip(e.iter_mut()).for_each(|(d,e)|*e=*d.min(e));
			e
		});
		if let Some(d)=dims{
			let ranges=d.map(|x|0..x);
			input.into_iter().map(|x|x.slice(ranges.clone())).collect()
		}else{
			input
		}
	}
}
impl<A:AutodiffBackend,W:Wrappable<B=A>> AutodiffModule<A> for Wrapped<W> where W::Decomposition:AutodiffModule<A>,W::With<A::InnerBackend>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>{
	fn valid(&self)->Self::InnerModule{Wrapped::new(Decompose::compose(self.inner.decompose_cloned().valid()))}
	type InnerModule=Wrapped<W::With<A::InnerBackend>>;
}
impl<A:Decompose> Decompose for Regression<A>{
	fn compose(decomposition:Self::Decomposition)->Self{Self(A::compose(decomposition))}
	fn decompose(self)->Self::Decomposition{self.0.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.0.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Wrappable> Op for Regression<A>{
	type Output=RegressionOutput<A::B>;
}
impl<B:Backend,K:TensorKind<B>,const N:usize> Decompose for Tensor<B,N,K>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend,S:?Sized+AsRef<str>> From<&S> for Value<B>{
	fn from(value:&S)->Self{Self::Incompatible(value.as_ref().to_string())}
}
impl<B:Backend,W:Wrappable<B=B>> Module<B> for Wrapped<W> where W::Decomposition:Module<B>{
	fn collect_devices(&self,devices:Vec<<B as Backend>::Device>)->Vec<<B as Backend>::Device>{self.inner.decompose_cloned().collect_devices(devices)}
	fn devices(&self)->Vec<<B as Backend>::Device>{self.inner.decompose_cloned().devices()}
	fn fork(self,device:&<B as Backend>::Device)->Self{Self::new(W::compose(self.inner.decompose().fork(device)))}
	fn into_record(self)->Self::Record{self.inner.decompose().into_record()}
	fn load_file<F:FileRecorder<B>,P:Into<PathBuf>>(self,filepath:P,recorder:&F,device:&<B as Backend>::Device)->Result<Self,RecorderError>{self.inner.decompose().load_file(filepath,recorder,device).map(|a|Self::new(W::compose(a)))}
	fn load_record(self,record:Self::Record)->Self{Self::new(W::compose(self.inner.decompose().load_record(record)))}
	fn map<Mapper:ModuleMapper<B>>(self,mapper:&mut Mapper)->Self{Self::new(W::compose(self.inner.decompose().map(mapper)))}
	fn num_params(&self)->usize{self.inner.decompose_cloned().num_params()}
	fn quantize_weights(self,quantizer:&mut Quantizer)->Self{Self::new(W::compose(self.inner.decompose().quantize_weights(quantizer)))}
	fn save_file<F:FileRecorder<B>,P:Into<PathBuf>>(self,filepath:P,recorder:&F)->Result<(),RecorderError>{self.inner.decompose().save_file(filepath,recorder)}
	fn to_device(self,device:&<B as Backend>::Device)->Self{Self::new(W::compose(self.inner.decompose().to_device(device)))}
	fn visit<Visitor:ModuleVisitor<B>>(&self,visitor:&mut Visitor){self.inner.decompose_cloned().visit(visitor)}
	type Record=<W::Decomposition as Module<B>>::Record;
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
impl<B:Backend> AI<Value<B>,Value<B>> for Layer<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match self{Layer::Dropout(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Embedding(a)=>a.fix_type::<Value<B>>().forward(input),Layer::LayerNorm(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Linear(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Mse(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Relu(a)=>a.fix_type::<Value<B>>().forward(input)}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Dropout{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{
			Value::F1(x)=>self.forward(x).into(),Value::F2(x)=>self.forward(x).into(),Value::F3(x)=>self.forward(x).into(),Value::F4(x)=>self.forward(x).into(),Value::F5(x)=>self.forward(x).into(),Value::F6(x)=>self.forward(x).into(),Value::F7(x)=>self.forward(x).into(),Value::F8(x)=>self.forward(x).into(),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>x.into_iter().map(|x|self.fix_type::<Value<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"dropout is only available for floats".into()
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Embedding<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
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
			Value::F1(x)=>apply_linear(self,x).into(),Value::F2(x)=>apply_linear(self,x).into(),Value::F3(x)=>apply_linear(self,x).into(),Value::F4(x)=>apply_linear(self,x).into(),Value::F5(x)=>apply_linear(self,x).into(),Value::F6(x)=>apply_linear(self,x).into(),Value::F7(x)=>apply_linear(self,x).into(),Value::F8(x)=>apply_linear(self,x).into(),Value::I1(x)=>apply_embed::<B,1,2>(self,x).into(),Value::I2(x)=>apply_embed::<B,2,3>(self,x).into(),Value::I3(x)=>apply_embed::<B,3,4>(self,x).into(),Value::I4(x)=>apply_embed::<B,4,5>(self,x).into(),Value::I5(x)=>apply_embed::<B,5,6>(self,x).into(),Value::I6(x)=>apply_embed::<B,6,7>(self,x).into(),Value::I7(x)=>apply_embed::<B,7,8>(self,x).into(),Value::I8(_x)=>"embedding output would exceed maximum supported rank".into(),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>x.into_iter().map(|x|self.fix_type::<Value<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"embedding is only available for float or int inputs".into()
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for LayerNorm<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{Value::F1(x)=>self.forward(x).into(),Value::F2(x)=>self.forward(x).into(),Value::F3(x)=>self.forward(x).into(),Value::F4(x)=>self.forward(x).into(),Value::F5(x)=>self.forward(x).into(),Value::F6(x)=>self.forward(x).into(),Value::F7(x)=>self.forward(x).into(),Value::F8(x)=>self.forward(x).into(),Value::I1(x)=>self.forward(x.float()).into(),Value::I2(x)=>self.forward(x.float()).into(),Value::I3(x)=>self.forward(x.float()).into(),Value::I4(x)=>self.forward(x.float()).into(),Value::I5(x)=>self.forward(x.float()).into(),Value::I6(x)=>self.forward(x.float()).into(),Value::I7(x)=>self.forward(x.float()).into(),Value::I8(x)=>self.forward(x.float()).into(),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>x.into_iter().map(|x|self.fix_type::<Value<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"layer norm is only supported for numeric inputs".into()}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Linear<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{Value::F1(x)=>self.forward(x).into(),Value::F2(x)=>self.forward(x).into(),Value::F3(x)=>self.forward(x).into(),Value::F4(x)=>self.forward(x).into(),Value::F5(x)=>self.forward(x).into(),Value::F6(x)=>self.forward(x).into(),Value::F7(x)=>self.forward(x).into(),Value::F8(x)=>self.forward(x).into(),Value::I1(x)=>self.forward(x.float()).into(),Value::I2(x)=>self.forward(x.float()).into(),Value::I3(x)=>self.forward(x.float()).into(),Value::I4(x)=>self.forward(x.float()).into(),Value::I5(x)=>self.forward(x.float()).into(),Value::I6(x)=>self.forward(x.float()).into(),Value::I7(x)=>self.forward(x.float()).into(),Value::I8(x)=>self.forward(x.float()).into(),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>x.into_iter().map(|x|self.fix_type::<Value<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"linear is only supported for numeric inputs".into()}
	}
}
impl<A:AI<X,(Value<B>,Value<B>,Value<B>)>,B:Backend,X> AI<X,RegressionOutput<B>> for Regression<A>{
	fn forward(&self,input:X)->RegressionOutput<B>{
		let (loss,output,target)=self.0.forward(input);
		let loss=match loss{Value::F1(x)=>x,Value::F2(x)=>x.mean(),Value::F3(x)=>x.mean(),Value::F4(x)=>x.mean(),Value::F5(x)=>x.mean(),Value::F6(x)=>x.mean(),Value::F7(x)=>x.mean(),Value::F8(x)=>x.mean(),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let output=match output{Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(1,2),Value::F4(x)=>x.flatten(1,3),Value::F5(x)=>x.flatten(1,4),Value::F6(x)=>x.flatten(1,5),Value::F7(x)=>x.flatten(1,6),Value::F8(x)=>x.flatten(1,7),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let target=match target{Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(1,2),Value::F4(x)=>x.flatten(1,3),Value::F5(x)=>x.flatten(1,4),Value::F6(x)=>x.flatten(1,5),Value::F7(x)=>x.flatten(1,6),Value::F8(x)=>x.flatten(1,7),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		RegressionOutput::new(loss,output,target)
	}
	fn forward_mut(&mut self,input:X)->RegressionOutput<B>{
		let (loss,output,target)=self.0.forward_mut(input);
		let loss=match loss{Value::F1(x)=>x,Value::F2(x)=>x.mean(),Value::F3(x)=>x.mean(),Value::F4(x)=>x.mean(),Value::F5(x)=>x.mean(),Value::F6(x)=>x.mean(),Value::F7(x)=>x.mean(),Value::F8(x)=>x.mean(),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let output=match output{Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(1,2),Value::F4(x)=>x.flatten(1,3),Value::F5(x)=>x.flatten(1,4),Value::F6(x)=>x.flatten(1,5),Value::F7(x)=>x.flatten(1,6),Value::F8(x)=>x.flatten(1,7),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let target=match target{Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(1,2),Value::F4(x)=>x.flatten(1,3),Value::F5(x)=>x.flatten(1,4),Value::F6(x)=>x.flatten(1,5),Value::F7(x)=>x.flatten(1,6),Value::F8(x)=>x.flatten(1,7),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		RegressionOutput::new(loss,output,target)
	}
}
impl<A:AI<X,Value<B>>,B:Backend,X> AI<(X,Value<B>),(Value<B>,Value<B>,Value<B>)> for MSE<A>{
	fn forward(&self,(input,target):(X,Value<B>))->(Value<B>,Value<B>,Value<B>){
		let output=self.0.forward(input);
		let loss=MseLoss.fix_type::<Value<B>>().forward(Value::Multi(vec![output.clone(),target.clone()]));
		(loss,output,target)
	}
	fn forward_mut(&mut self,(input,target):(X,Value<B>))->(Value<B>,Value<B>,Value<B>){
		let output=self.0.forward_mut(input);
		let loss=MseLoss.fix_type::<Value<B>>().forward(Value::Multi(vec![output.clone(),target.clone()]));
		(loss,output,target)
	}
}
impl<A:AutodiffBackend,W:AI<X,(Value<A>,Value<A>,Value<A>)>+Wrappable<B=A>,X> TrainStep<X,RegressionOutput<A>> for Wrapped<Regression<W>> where W::Decomposition:AutodiffModule<A>,W::With<A::InnerBackend>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>{
	fn step(&self,item:X)->TrainOutput<RegressionOutput<A>>{
		let output:RegressionOutput<A>=self.forward(item);
		TrainOutput::new(self,output.loss.backward(),output)
	}
}
impl<B:Backend,W:AI<X,(Value<B>,Value<B>,Value<B>)>+Wrappable<B=B>,X> ValidStep<X,RegressionOutput<B>> for Wrapped<Regression<W>> where W::Decomposition:Module<B>{
	fn step(&self,item:X)->RegressionOutput<B>{self.forward(item)}
}
impl<B:Backend> AI<Value<B>,Value<B>> for MseLoss{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{
			Value::Incompatible(x)=>x.into(),
			Value::Multi(x)=>if x.len()==2{
				let mut x=x.into_iter();
				match (x.next().unwrap(),x.next().unwrap()){(Value::F1(x0),Value::F1(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F2(x0),Value::F2(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F3(x0),Value::F3(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F4(x0),Value::F4(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F5(x0),Value::F5(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F6(x0),Value::F6(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F7(x0),Value::F7(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F8(x0),Value::F8(x1))=>AI::forward(self,(x0,x1)).into(),_=>"mse loss requires input pairs to be float tensors with the same rank".into()}
			}else{
				let y:Vec<Value<B>>=x.into_iter().map(|x|self.fix_type::<Value<B>>().forward(x)).collect();
				y.into()
			},
			_=>"mse loss requires inputs to be in pairs".into()
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Relu{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{Value::F1(x)=>self.forward(x).into(),Value::F2(x)=>self.forward(x).into(),Value::F3(x)=>self.forward(x).into(),Value::F4(x)=>self.forward(x).into(),Value::F5(x)=>self.forward(x).into(),Value::F6(x)=>self.forward(x).into(),Value::F7(x)=>self.forward(x).into(),Value::F8(x)=>self.forward(x).into(),Value::I1(x)=>self.forward(x.float()).into(),Value::I2(x)=>self.forward(x.float()).into(),Value::I3(x)=>self.forward(x.float()).into(),Value::I4(x)=>self.forward(x.float()).into(),Value::I5(x)=>self.forward(x.float()).into(),Value::I6(x)=>self.forward(x.float()).into(),Value::I7(x)=>self.forward(x.float()).into(),Value::I8(x)=>self.forward(x.float()).into(),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>x.into_iter().map(|x|self.fix_type::<Value<B>>().forward(x)).collect::<Vec<_>>().into(),_=>"relu is only supported for numeric inputs".into()}
	}
}
impl<B:Backend> AI<Tensor<B,2,Int>,Tensor<B,3>> for Embedding<B>{
	fn forward(&self,input:Tensor<B,2,Int>)->Tensor<B,3>{Embedding::forward(self,input)}
}
impl<B:Backend> Batcher<B,(Value<B>,Value<B>),(Value<B>,Value<B>)> for StackBatcher{
	fn batch(&self,items:Vec<(Value<B>,Value<B>)>,_device:&<B as Backend>::Device)->(Value<B>,Value<B>){
		let mut items=items.into_iter();
		let (input,target)=if let Some(i)=items.next(){i}else{return Default::default()};
		let (inputs,targets):(Vec<Value<B>>,Vec<Value<B>>)=items.unzip();
		let input=match input{
			Value::F1(x)=>Value::F2(Tensor::stack(Some(x).into_iter().chain(inputs.into_iter().map(|x|if let Value::F1(x)=x{x}else{panic!("incompatible values")})).collect(),0)),
			_=>todo!()
		};
		let target=match target{
			Value::F1(x)=>Value::F2(Tensor::stack(Some(x).into_iter().chain(targets.into_iter().map(|x|if let Value::F1(x)=x{x}else{panic!("incompatible values")})).collect(),0)),
			_=>todo!()
		};
		(input,target)
	}
}
impl<B:Backend> Layer<B>{
	/// creates a linear layer
	pub fn linear(bias:bool,input:usize,output:usize,_wscale:f32)->Self{
		let l=LinearConfig::new(input,output).with_bias(bias).init(&Default::default());
		//TODO make wscale work correctly
		/*if wscale!=1.0{
			l.bias=l.bias.map(|b|b.map(|b|b*wscale));
			l.weight=l.weight.map(|w|w*wscale);
		}*/
		Self::Linear(l)
	}
	/// creates a relu layer
	pub fn relu()->Self{Self::Relu(Relu)}
}
impl<B:Backend> Decompose for Layer<B>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend> Decompose for Value<B>{
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
impl<B:Backend> Default for Value<B>{
	fn default()->Self{Self::Multi(Vec::new())}
}
impl<B:Backend> From<String> for Value<B>{
	fn from(value:String)->Self{Self::Incompatible(value)}
}
impl<B:Backend> From<Tensor<B,1,Bool>> for Value<B>{
	fn from(value:Tensor<B,1,Bool>)->Self{Self::B1(value)}
}
impl<B:Backend> From<Tensor<B,2,Bool>> for Value<B>{
	fn from(value:Tensor<B,2,Bool>)->Self{Self::B2(value)}
}
impl<B:Backend> From<Tensor<B,3,Bool>> for Value<B>{
	fn from(value:Tensor<B,3,Bool>)->Self{Self::B3(value)}
}
impl<B:Backend> From<Tensor<B,4,Bool>> for Value<B>{
	fn from(value:Tensor<B,4,Bool>)->Self{Self::B4(value)}
}
impl<B:Backend> From<Tensor<B,5,Bool>> for Value<B>{
	fn from(value:Tensor<B,5,Bool>)->Self{Self::B5(value)}
}
impl<B:Backend> From<Tensor<B,6,Bool>> for Value<B>{
	fn from(value:Tensor<B,6,Bool>)->Self{Self::B6(value)}
}
impl<B:Backend> From<Tensor<B,7,Bool>> for Value<B>{
	fn from(value:Tensor<B,7,Bool>)->Self{Self::B7(value)}
}
impl<B:Backend> From<Tensor<B,8,Bool>> for Value<B>{
	fn from(value:Tensor<B,8,Bool>)->Self{Self::B8(value)}
}
impl<B:Backend> From<Tensor<B,1,Float>> for Value<B>{
	fn from(value:Tensor<B,1,Float>)->Self{Self::F1(value)}
}
impl<B:Backend> From<Tensor<B,2,Float>> for Value<B>{
	fn from(value:Tensor<B,2,Float>)->Self{Self::F2(value)}
}
impl<B:Backend> From<Tensor<B,3,Float>> for Value<B>{
	fn from(value:Tensor<B,3,Float>)->Self{Self::F3(value)}
}
impl<B:Backend> From<Tensor<B,4,Float>> for Value<B>{
	fn from(value:Tensor<B,4,Float>)->Self{Self::F4(value)}
}
impl<B:Backend> From<Tensor<B,5,Float>> for Value<B>{
	fn from(value:Tensor<B,5,Float>)->Self{Self::F5(value)}
}
impl<B:Backend> From<Tensor<B,6,Float>> for Value<B>{
	fn from(value:Tensor<B,6,Float>)->Self{Self::F6(value)}
}
impl<B:Backend> From<Tensor<B,7,Float>> for Value<B>{
	fn from(value:Tensor<B,7,Float>)->Self{Self::F7(value)}
}
impl<B:Backend> From<Tensor<B,8,Float>> for Value<B>{
	fn from(value:Tensor<B,8,Float>)->Self{Self::F8(value)}
}
impl<B:Backend> From<Tensor<B,1,Int>> for Value<B>{
	fn from(value:Tensor<B,1,Int>)->Self{Self::I1(value)}
}
impl<B:Backend> From<Tensor<B,2,Int>> for Value<B>{
	fn from(value:Tensor<B,2,Int>)->Self{Self::I2(value)}
}
impl<B:Backend> From<Tensor<B,3,Int>> for Value<B>{
	fn from(value:Tensor<B,3,Int>)->Self{Self::I3(value)}
}
impl<B:Backend> From<Tensor<B,4,Int>> for Value<B>{
	fn from(value:Tensor<B,4,Int>)->Self{Self::I4(value)}
}
impl<B:Backend> From<Tensor<B,5,Int>> for Value<B>{
	fn from(value:Tensor<B,5,Int>)->Self{Self::I5(value)}
}
impl<B:Backend> From<Tensor<B,6,Int>> for Value<B>{
	fn from(value:Tensor<B,6,Int>)->Self{Self::I6(value)}
}
impl<B:Backend> From<Tensor<B,7,Int>> for Value<B>{
	fn from(value:Tensor<B,7,Int>)->Self{Self::I7(value)}
}
impl<B:Backend> From<Tensor<B,8,Int>> for Value<B>{
	fn from(value:Tensor<B,8,Int>)->Self{Self::I8(value)}
}
impl<B:Backend> From<Vec<Value<B>>> for Value<B>{
	fn from(value:Vec<Value<B>>)->Self{Self::Multi(value)}
}
impl<B:Backend> Merge for Value<B>{
	fn merge(&mut self,other:Self){
		match (take(self),other){
			(Value::Multi(mut u),Value::Multi(v))=>{
				u.extend(v);
				*self=u.into();
			},
			(Value::Multi(mut u),v)=>if u.len()==0{
				*self=v;
			}else{
				u.push(v);
				*self=u.into();
			},
			(u,Value::Multi(mut v))=>if v.len()==0{
				*self=u;
			}else{
				v.push(u);
				*self=v.into();
			},
			(u,v)=>*self=vec![u,v].into()
		}
	}
}
impl<B:Backend> Op for Embedding<B>{
	type Output=Tensor<B,3>;
}
impl<B:Backend> Op for Layer<B>{
	type Output=Value<B>;
}
impl<B:Backend> Op for LayerNorm<B>{
	type Output=Tensor<B,1>;
}
impl<B:Backend> Op for Linear<B>{
	type Output=Tensor<B,1>;
}
impl<B:Backend> Wrappable for Layer<B>{
	type B=B;
	type With<C:Backend>=Layer<C>;
}
impl<T:?Sized+Op> Shortcuts for T{}
impl<W:AI<X,Y>+Wrappable,X,Y> AI<X,Y> for Wrapped<W>{
	fn forward(&self,input:X)->Y{self.inner.forward(input)}
	fn forward_mut(&mut self,input:X)->Y{self.inner.forward_mut(input)}
}
impl<W:Op+Wrappable> Op for Wrapped<W>{
	type Output=W::Output;
}
impl<W:Wrappable> Decompose for Wrapped<W>{
	fn compose(decomposition:Self::Decomposition)->Self{Self::new(W::compose(decomposition))}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=W::Decomposition;
}
impl<W:Wrappable> Display for Wrapped<W>{
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->Result<(),std::fmt::Error>{write!(f,"todo")}
}
impl<W:Wrappable> From<W> for Wrapped<W>{
	fn from(value:W)->Self{Self::new(value)}
}
impl<W:Wrappable> ModuleDisplay for Wrapped<W> where W::Decomposition:ModuleDisplay{
	fn custom_content(&self,content:Content)->Option<Content>{self.inner.decompose_cloned().custom_content(content)}
	fn custom_settings(&self)->Option<DisplaySettings>{self.inner.decompose_cloned().custom_settings()}
	fn format(&self,passed_settings:DisplaySettings)->String{self.inner.decompose_cloned().format(passed_settings)}
}
impl<W:Wrappable> ModuleDisplayDefault for Wrapped<W> where W::Decomposition:ModuleDisplayDefault{
	fn content(&self,content:Content)->Option<Content>{self.inner.decompose_cloned().content(content)}
	fn num_params(&self)->usize{self.inner.decompose_cloned().num_params()}
}
impl<W:Wrappable> Wrappable for AccQ<W>{
	type B=W::B;
	type With<C:Backend>=AccQ<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Branch<W>{
	type B=W::B;
	type With<C:Backend>=Branch<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Cat<W>{
	type B=W::B;
	type With<C:Backend>=Cat<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Duplicate<W>{
	type B=W::B;
	type With<C:Backend>=Duplicate<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Sequential<W>{
	type B=W::B;
	type With<C:Backend>=Sequential<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Graph<W>{
	type B=W::B;
	type With<C:Backend>=Graph<W::With<C>>;
}
impl<W:Wrappable> Wrappable for MSE<W>{
	type B=W::B;
	type With<C:Backend>=MSE<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Regression<W>{
	type B=W::B;
	type With<C:Backend>=Regression<W::With<C>>;
}
impl<W:Wrappable> Wrappable for SoftChoose<W>{
	type B=W::B;
	type With<C:Backend>=SoftChoose<W::With<C>>;
}
impl<W:Wrappable> Wrappable for ToEach<W>{
	type B=W::B;
	type With<C:Backend>=ToEach<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Unvec<W>{
	type B=W::B;
	type With<C:Backend>=Unvec<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Zip<W>{
	type B=W::B;
	type With<C:Backend>=Zip<W::With<C>>;
}
impl<W:Wrappable> Wrapped<W>{
	/// references the inner value
	pub fn inner(&self)->&W{&self.inner}
	/// references the inner value
	pub fn inner_mut(&mut self)->&mut W{&mut self.inner}
	/// unwraps the inner value
	pub fn into_inner(self)->W{self.inner}
	/// creates a new wrapped value
	pub fn new(inner:W)->Self{
		Self{inner}
	}
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,W:'static+Wrappable<B=A>,Y:'static+ItemLazy+Send+Sync,Z:'static+ItemLazy+Send+Sync> Wrapped<W> where <Self as AutodiffModule<A>>::InnerModule:ValidStep<(Value<B>,Value<B>),Z>,Self:TrainStep<(Value<A>,Value<A>),Y>,W::Decomposition:AutodiffModule<A>,W::With<B>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>+Op<Output=Z>,W:Op<Output=Y>,Y::ItemSync:Adaptor<LossInput<NdArray>>,Z::ItemSync:Adaptor<LossInput<NdArray>>{
	/// trains the model
	pub fn train<O:Optimizer<Self,A>,S:LrScheduler,T:'static+Dataset<(Value<A>,Value<A>)>,V:'static+Dataset<(Value<B>,Value<B>)>>(self,config:&TrainConfig,optimizer:O,scheduler:S,train:T,valid:V)->Self where O::Record:'static,S::Record<A>:'static{
		let batcher=StackBatcher;
		let trainloader=DataLoaderBuilder::new(batcher).batch_size(config.batch_size).shuffle(random()).num_workers(config.workers).build(train);
		let validloader=DataLoaderBuilder::new(batcher).batch_size(config.batch_size).shuffle(random()).num_workers(config.workers).build(valid);

		create_folder(&config.artifact_directory).unwrap();
		let builder=LearnerBuilder::new(&config.artifact_directory).metric_train_numeric(LossMetric::new()).metric_valid_numeric(LossMetric::new()).with_file_checkpointer(CompactRecorder::new()).devices(vec![<W::B as Backend>::Device::default()]).num_epochs(config.epochs);
		let learner=builder.build(self,optimizer,scheduler);
		learner.fit(trainloader,validloader)
	}
}
#[cfg(test)]
mod tests{
	#[test]
	fn learn_xor(){
		type A=Autodiff<Wgpu>;
		let i0=Tensor::<A,1>::from_data(TensorData::new([0.0,0.0].to_vec(),[2]),&Default::default());
		let i1=Tensor::<A,1>::from_data(TensorData::new([0.0,1.0].to_vec(),[2]),&Default::default());
		let i2=Tensor::<A,1>::from_data(TensorData::new([1.0,0.0].to_vec(),[2]),&Default::default());
		let i3=Tensor::<A,1>::from_data(TensorData::new([1.0,1.0].to_vec(),[2]),&Default::default());
		let o0=Tensor::<A,1>::from_data(TensorData::new([0.0].to_vec(),[1]),&Default::default());
		let o1=Tensor::<A,1>::from_data(TensorData::new([1.0].to_vec(),[1]),&Default::default());
		let o2=Tensor::<A,1>::from_data(TensorData::new([1.0].to_vec(),[1]),&Default::default());
		let o3=Tensor::<A,1>::from_data(TensorData::new([0.0].to_vec(),[1]),&Default::default());

		let dataset:Vec<(Tensor<A,1>,Tensor<A,1>)>=[(i0,o0),(i1,o1),(i2,o2),(i3,o3)].into_iter().cycle().take(4000).collect();
		let train=InMemDataset::new(dataset.clone().into_iter().map(|(i,o)|(Value::from(i),Value::from(o))).collect());
		let valid=InMemDataset::new(dataset.into_iter().map(|(i,o)|(Value::from(i.valid()),Value::from(o.valid()))).collect());
		let mut graph:Graph<Layer<A>>=Graph::new();
		let mut l=VertexLabels::new();
		graph.connect(true,l.label("input"),Layer::linear(true,2,10,1.0),l.label("x"));
		graph.connect(true,l.label("x"),Layer::relu(),l.label("y"));
		graph.connect(true,l.label("y"),Layer::linear(false,10,1,1.0),l.label("output"));
		let graph=Unvec(graph.clone()).mse().regression().wrap();
		let graph=graph.train(&TrainConfig::new(),SgdConfig::new().init(),0.01,train,valid);
		let graph=graph.valid().into_inner().0.0;

		let inputval=Value::from(Tensor::<Wgpu,2>::from_data(TensorData::new([0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0].to_vec(),[4,2]),&Default::default()));
		let outputval=graph.forward(inputval);
		if let Value::F2(o)=outputval{
			let target=Tensor::<Wgpu,2>::from_data(TensorData::new([0.0,1.0,1.0,0.0].to_vec(),[4,1]),&Default::default());
			let error=(target-o.clone()).abs().max();
			println!("{}",o);
			assert!(error.into_scalar()<0.1);
		}else{
			panic!("h");
		}
	}
	use burn::{
		backend::{Autodiff,Wgpu},data::dataset::InMemDataset,optim::SgdConfig
	};
	use super::*;
}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
/// metrics renderer implementation that doesn't actually do anything
pub struct DontRender;
#[derive(Debug,Module)]//TODO more layers
/// enumerates some burn layers
pub enum Layer<B:Backend>{Dropout(Dropout),Embedding(Embedding<B>),LayerNorm(LayerNorm<B>),Linear(Linear<B>),Mse(MseLoss),Relu(Relu)}
#[derive(Clone,Debug)]//TODO implement module for this
/// enumerates burn tensors up to 8 dimensions
pub enum Value<B:Backend>{B1(Tensor<B,1,Bool>),B2(Tensor<B,2,Bool>),B3(Tensor<B,3,Bool>),B4(Tensor<B,4,Bool>),B5(Tensor<B,5,Bool>),B6(Tensor<B,6,Bool>),B7(Tensor<B,7,Bool>),B8(Tensor<B,8,Bool>),F1(Tensor<B,1,Float>),F2(Tensor<B,2,Float>),F3(Tensor<B,3,Float>),F4(Tensor<B,4,Float>),F5(Tensor<B,5,Float>),F6(Tensor<B,6,Float>),F7(Tensor<B,7,Float>),F8(Tensor<B,8,Float>),I1(Tensor<B,1,Int>),I2(Tensor<B,2,Int>),I3(Tensor<B,3,Int>),I4(Tensor<B,4,Int>),I5(Tensor<B,5,Int>),I6(Tensor<B,6,Int>),I7(Tensor<B,7,Int>),I8(Tensor<B,8,Int>),Incompatible(String),Multi(Vec<Self>)}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// wrapper for converting loss to regression output
pub struct Regression<A>(pub A);
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
/// batcher that stacks things
pub struct StackBatcher;
#[derive(Config,Debug)]
/// configuration for convenient training through the wrapper
pub struct TrainConfig{
	#[config(default="String::from(\".artifact\")")]
	artifact_directory:String,
	#[config(default="16")]
	batch_size:usize,
	#[config(default="false")]
	checkpoints:bool,
	#[config(default="false")]
	console_rendering:bool,
	#[config(default="10")]
	epochs:usize,
	#[config(default="false")]
	summary:bool,
	#[config(default="4")]
	workers:usize
}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// wraps in a burn wrapper
pub struct Wrapped<W:Wrappable>{inner:W}
/// chained method shortcut trait
pub trait Shortcuts{
	/// wraps in a regression wrapper
	fn regression(self)->Regression<Self> where Regression<Self>:Op,Self:Sized{Regression(self)}
	/// wraps in a burn wrapper
	fn wrap(self)->Wrapped<Self> where Self:Wrappable{Wrapped::new(self)}
}
/// higher kinded type trait to allow rewrapping burn modules in different backends to implement some wrapper features
pub trait Wrappable:Clone+Debug+Decompose+Send{
	type B:Backend;
	type With<C:Backend>:Wrappable<B=C,With<C>=Self::With<C>>+Wrappable<B=C,With<Self::B>=Self>;
}
pub use burn as lib;
use burn::{
	backend::NdArray,
	data::{
		dataset::Dataset,dataloader::{batcher::Batcher,DataLoaderBuilder}
	},
	lr_scheduler::LrScheduler,
	module::{AutodiffModule,Content,DisplaySettings,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor,Quantizer},
	nn::{
		Dropout,Embedding,LayerNorm,Linear,LinearConfig,Relu,loss::MseLoss
	},
	optim::Optimizer,
	prelude::*,
	record::{CompactRecorder,FileRecorder,RecorderError},
	tensor::{BasicOps,TensorKind,activation::softmax,backend::AutodiffBackend},
	train::{
		LearnerBuilder,RegressionOutput,TrainOutput,TrainStep,ValidStep,metric::{Adaptor,ItemLazy,LossInput,LossMetric},renderer::{MetricState,MetricsRenderer,TrainingProgress}
	}
};
use crate::{ai::*,graph::*};
use rand::random;
use std::{
	fmt::{Debug,Display},fs::{create_dir_all as create_folder},mem::take,path::PathBuf
};
