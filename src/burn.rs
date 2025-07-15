fn accumulate_q_burn<B:Backend,const N:usize>(dim:usize,gamma:f32,i:Tensor<B,N>)->Tensor<B,N>{
	let mut q=i.split(1,dim);
	q.iter_mut().rev().fold(None,|future,present|{
		if let Some(f)=future{*present=f*gamma+present.clone()}
		Some(present.clone())
	});
	Tensor::cat(q,dim)
}
fn soft_choose_burn_1<B:Backend,const N:usize>(dim:usize,logits:Tensor<B,N>,temperature:f32)->u32{
	let logits=if dim==N-1{logits}else{logits.movedim(dim,N-1)};
	let distribution=softmax(logits/temperature,N-1).into_data();
	distribution.iter().scan(random(),|choice:&mut f32,weight:f32|Some(*choice-=weight).filter(|_|*choice>=0.0)).count() as u32
}
fn soft_choose_burn_multi<B:Backend,const N:usize>(dim:usize,logits:Tensor<B,N>,temperature:f32)->Vec<u32>{
	let logits=if dim==N-1{logits}else{logits.movedim(dim,N-1)};
	let chunk=logits.dims()[N-1];
	let distribution=softmax(logits/temperature,N-1).into_data().to_vec().unwrap();
	distribution.chunks_exact(chunk).map(|d|d.iter().scan(random(),|choice:&mut f32,weight:&f32|Some(*choice-=weight).filter(|_|*choice>=0.0)).count() as u32).collect()
}
fn soft_choose_burn_tensor<B:Backend,const N:usize>(dim:usize,logits:Tensor<B,N>,temperature:f32)->Tensor<B,N,Int>{//TODO test
	//let logits=if dim==N-1{logits}else{logits.movedim(dim,N-1)}
	let device=logits.device();
	let mut dims=logits.dims();

	dims[N-1]=1;
	Tensor::from_data(TensorData::new(soft_choose_burn_multi(dim,logits,temperature),dims),&device)
}
impl Config{
	/// initializes the layer
	pub fn init<B:Backend>(&self,device:&B::Device)->Layer<B>{
		match self{Config::Dropout(c)=>Layer::Dropout(c.init()),Config::Embedding(c)=>Layer::Embedding(c.init(device)),Config::LayerNorm(c)=>Layer::LayerNorm(c.init(device)),Config::Linear(c)=>Layer::Linear(c.init(device)),Config::CrossEntropy(c)=>Layer::CrossEntropy(c.init(device)),Config::Mse=>Layer::Mse(MseLoss),Config::Relu=>Layer::Relu(Relu::new()),Config::Stack(d)=>Layer::Stack(*d)}
	}
	/// scales the initializer
	pub fn w_scale(self,r:f32)->Self{//TODO probably shorter to wscal a mutable initializer
		match self{
			Config::CrossEntropy(c)=>c.into(),
			Config::Dropout(c)=>c.into(),
			Config::Embedding(c)=>EmbeddingConfig{d_model:c.d_model,initializer:w_scale(c.initializer,r),n_embedding:c.n_embedding}.into(),
			Config::LayerNorm(c)=>c.into(),
			Config::Linear(c)=>LinearConfig{bias:c.bias,d_input:c.d_input,d_output:c.d_output,initializer:w_scale(c.initializer,r)}.into(),
			Config::Mse=>Config::Mse,
			Config::Relu=>Config::Relu,
			Config::Stack(d)=>Config::Stack(d)
		}
	}
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
impl From<CrossEntropyLossConfig> for Config{
	fn from(value:CrossEntropyLossConfig)->Self{Config::CrossEntropy(value)}
}
impl From<DropoutConfig> for Config{
	fn from(value:DropoutConfig)->Self{Config::Dropout(value)}
}
impl From<EmbeddingConfig> for Config{
	fn from(value:EmbeddingConfig)->Self{Config::Embedding(value)}
}
impl From<LayerNormConfig> for Config{
	fn from(value:LayerNormConfig)->Self{Config::LayerNorm(value)}
}
impl From<LinearConfig> for Config{
	fn from(value:LinearConfig)->Self{Config::Linear(value)}
}
impl MetricsRenderer for DontRender{
	fn update_train(&mut self,_state:MetricState){}
	fn update_valid(&mut self,_state:MetricState){}
	fn render_train(&mut self,_item:TrainingProgress){}
	fn render_valid(&mut self,_item:TrainingProgress){}
}
impl Op for Classification<()>{
	type Output=ClassificationOutput<NdArray>;
}
impl Op for Dropout{
	type Output=Tensor<NdArray,1>;
}
impl Op for MseLoss{
	type Output=Tensor<NdArray,1>;
}
impl Op for Relu{
	type Output=Tensor<NdArray,1>;
}
impl Op for Regression<()>{
	type Output=RegressionOutput<NdArray>;
}
impl<A:AI<X,(Value<B>,Value<B>,Value<B>)>,B:Backend,X> AI<X,ClassificationOutput<B>> for Classification<A>{
	fn forward(&self,input:X)->ClassificationOutput<B>{self.with_inner(()).forward(self.inner().forward(input))}
	fn forward_mut(&mut self,input:X)->ClassificationOutput<B>{self.with_inner(()).forward(self.inner_mut().forward_mut(input))}
}
impl<A:AI<X,(Value<B>,Value<B>,Value<B>)>,B:Backend,X> AI<X,RegressionOutput<B>> for Regression<A>{
	fn forward(&self,input:X)->RegressionOutput<B>{self.with_inner(()).forward(self.inner().forward(input))}
	fn forward_mut(&mut self,input:X)->RegressionOutput<B>{self.with_inner(()).forward(self.inner_mut().forward_mut(input))}
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,W:'static+Wrappable<B=A>,Y:'static+ItemLazy+Send+Sync,Z:'static+ItemLazy+Send+Sync> Wrapped<W> where <Self as AutodiffModule<A>>::InnerModule:ValidStep<(Value<B>,Value<B>),Z>,Self:TrainStep<(Value<A>,Value<A>),Y>,W::Decomposition:AutodiffModule<A>,W::With<B>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>+Op<Output=Z>,W:Op<Output=Y>,Y::ItemSync:Adaptor<LossInput<NdArray>>,Z::ItemSync:Adaptor<LossInput<NdArray>>{
	/// trains the model
	pub fn train<O:Optimizer<Self,A>,S:LrScheduler,T:'static+Dataset<(Value<A>,Value<A>)>,V:'static+Dataset<(Value<B>,Value<B>)>>(self,config:&TrainConfig,optimizer:O,scheduler:S,train:T,valid:V)->Self where O::Record:'static,S::Record<A>:'static{
		let batcher=BatchStacker;
		let trainloader=DataLoaderBuilder::new(batcher).batch_size(config.batch_size).shuffle(random()).num_workers(config.workers).build(train);
		let validloader=DataLoaderBuilder::new(batcher).batch_size(config.batch_size).shuffle(random()).num_workers(config.workers).build(valid);

		create_folder(&config.artifact_directory).unwrap();
		let builder=LearnerBuilder::new(&config.artifact_directory).metric_train_numeric(LossMetric::new()).metric_valid_numeric(LossMetric::new());
		let builder=if config.checkpoints{builder.with_file_checkpointer(CompactRecorder::new())}else{builder};
		let builder=if config.console_rendering{builder}else{builder.renderer(DontRender)};
		let builder=builder.devices(vec![<W::B as Backend>::Device::default()]).num_epochs(config.epochs);
		let builder=if config.summary{builder.summary()}else{builder};
		let learner=builder.build(self,optimizer,scheduler);
		learner.fit(trainloader,validloader)
	}
}
impl<A:AutodiffBackend,W:AI<X,(Value<A>,Value<A>,Value<A>)>+Wrappable<B=A>,X> TrainStep<X,ClassificationOutput<A>> for Wrapped<Classification<W>> where W::Decomposition:AutodiffModule<A>,W::With<A::InnerBackend>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>{
	fn step(&self,item:X)->TrainOutput<ClassificationOutput<A>>{
		let output:ClassificationOutput<A>=self.forward(item);
		TrainOutput::new(self,output.loss.backward(),output)
	}
}
impl<A:AutodiffBackend,W:AI<X,(Value<A>,Value<A>,Value<A>)>+Wrappable<B=A>,X> TrainStep<X,RegressionOutput<A>> for Wrapped<Regression<W>> where W::Decomposition:AutodiffModule<A>,W::With<A::InnerBackend>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>{
	fn step(&self,item:X)->TrainOutput<RegressionOutput<A>>{
		let output:RegressionOutput<A>=self.forward(item);
		TrainOutput::new(self,output.loss.backward(),output)
	}
}
impl<A:AutodiffBackend,W:Wrappable<B=A>> AutodiffModule<A> for Wrapped<W> where W::Decomposition:AutodiffModule<A>,W::With<A::InnerBackend>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>{
	fn valid(&self)->Self::InnerModule{Wrapped::new(Decompose::compose(self.inner.decompose_cloned().valid()))}
	type InnerModule=Wrapped<W::With<A::InnerBackend>>;
}
impl<A:Decompose> Decompose for Classification<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Regression<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition)}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Into<Value<B>>,B:Backend> FromIterator<A> for Value<B>{
	fn from_iter<I:IntoIterator<Item=A>>(iter:I)->Self{
		let v:Vec<Value<B>>=iter.into_iter().map(Into::into).collect();
		if v.len()==1{v.into_iter().next().unwrap()}else{v.into()}
	}
}
impl<A:Op<Output=Y>+Wrappable,Y> Op for Classification<A> where Classification<()>:AI<Y,ClassificationOutput<A::B>>{
	type Output=ClassificationOutput<A::B>;
}
impl<A:Op<Output=Y>+Wrappable,Y> Op for Regression<A> where Regression<()>:AI<Y,RegressionOutput<A::B>>{
	type Output=RegressionOutput<A::B>;
}
impl<A:Wrappable<B=B>,B:Backend,D:Wrappable<B=B>> Wrappable for (A,D){
	type B=B;
	type With<C:Backend>=(A::With<C>,D::With<C>);
}
impl<A:Wrappable<B=B>,B:Backend,D:Wrappable<B=B>,E:Wrappable<B=B>> Wrappable for (A,D,E){
	type B=B;
	type With<C:Backend>=(A::With<C>,D::With<C>,E::With<C>);
}
impl<A:Wrappable<B=B>,B:Backend,X:Wrappable<B=B>,Y:Wrappable<B=B>> Wrappable for SetType<A,X,Y>{
	type B=B;
	type With<C:Backend>=SetType<A::With<C>,X::With<C>,Y::With<C>>;
}
impl<A> Classification<A>{
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self where Classification<A>:Op{
		Self{inner}
	}
	/// references the inner value
	pub fn inner(&self)->&A{&self.inner}
	/// references the inner value
	pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
	/// converts into the inner value
	pub fn into_inner(self)->A{self.inner}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->Classification<B> where Classification<B>:Op{Classification::from_inner(inner)}
}
impl<A> Regression<A>{
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self where Regression<A>:Op{
		Self{inner}
	}
	/// references the inner value
	pub fn inner(&self)->&A{&self.inner}
	/// references the inner value
	pub fn inner_mut(&mut self)->&mut A{&mut self.inner}
	/// converts into the inner value
	pub fn into_inner(self)->A{self.inner}
	/// replaces the inner value
	pub fn with_inner<B>(&self,inner:B)->Regression<B> where Regression<B>:Op{Regression::from_inner(inner)}
}
impl<B:Backend,D:WhichDims,K:BasicOps<B>+TensorKind<B>,const N:usize> AI<Vec<Tensor<B,N,K>>,Vec<Tensor<B,N,K>>> for TruncateToMatch<(),D>{
	fn forward(&self,input:Vec<Tensor<B,N,K>>)->Vec<Tensor<B,N,K>>{
		let alignment=self.alignment();
		let which_dims=||self.dims().which_dims();

		if let Some(d)=input.iter().map(|i|i.dims()).reduce(|d,mut e|{
			which_dims().for_each(|n|e[n]=d[n].min(e[n]));
			e
		}){
			input.into_iter().map(|x|{
				let mut ranges=d.map(|x|0..x);
				match alignment{
					Alignment::Center=>{
						let d=x.dims();
						which_dims().for_each(|n|{
							let mid=d[n]/2;
							let width=ranges[n].len()/2;
							ranges[n]=mid-width..mid+width;
						});
					},
					Alignment::Left=>(),
					Alignment::Right=>{
						let d=x.dims();
						which_dims().for_each(|n|ranges[n]=d[n]-ranges[n].len()..d[n]);
					}
				}
				x.slice(ranges)
			}).collect()
		}else{
			input
		}
	}
}
impl<B:Backend,D:WhichDims> AI<Vec<Value<B>>,Vec<Value<B>>> for TruncateToMatch<(),D>{
	fn forward(&self,input:Vec<Value<B>>)->Vec<Value<B>>{// TODO this could work with not necessarily homogenous types. easier if value inplements dims and slice
		let mut input=input.into_iter();

		match input.next(){
			None=>Vec::new(),
			Some(Value::B1(x))=>self.forward_fixed::<Vec<Tensor<B,1,Bool>>>([x].into_iter().chain(input.map(|x|x.try_b1().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::B2(x))=>self.forward_fixed::<Vec<Tensor<B,2,Bool>>>([x].into_iter().chain(input.map(|x|x.try_b2().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::B3(x))=>self.forward_fixed::<Vec<Tensor<B,3,Bool>>>([x].into_iter().chain(input.map(|x|x.try_b3().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::B4(x))=>self.forward_fixed::<Vec<Tensor<B,4,Bool>>>([x].into_iter().chain(input.map(|x|x.try_b4().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::B5(x))=>self.forward_fixed::<Vec<Tensor<B,5,Bool>>>([x].into_iter().chain(input.map(|x|x.try_b5().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::B6(x))=>self.forward_fixed::<Vec<Tensor<B,6,Bool>>>([x].into_iter().chain(input.map(|x|x.try_b6().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::B7(x))=>self.forward_fixed::<Vec<Tensor<B,7,Bool>>>([x].into_iter().chain(input.map(|x|x.try_b7().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::B8(x))=>self.forward_fixed::<Vec<Tensor<B,8,Bool>>>([x].into_iter().chain(input.map(|x|x.try_b8().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::F1(x))=>self.forward_fixed::<Vec<Tensor<B,1>>>([x].into_iter().chain(input.map(|x|x.try_f1().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::F2(x))=>self.forward_fixed::<Vec<Tensor<B,2>>>([x].into_iter().chain(input.map(|x|x.try_f2().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::F3(x))=>self.forward_fixed::<Vec<Tensor<B,3>>>([x].into_iter().chain(input.map(|x|x.try_f3().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::F4(x))=>self.forward_fixed::<Vec<Tensor<B,4>>>([x].into_iter().chain(input.map(|x|x.try_f4().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::F5(x))=>self.forward_fixed::<Vec<Tensor<B,5>>>([x].into_iter().chain(input.map(|x|x.try_f5().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::F6(x))=>self.forward_fixed::<Vec<Tensor<B,6>>>([x].into_iter().chain(input.map(|x|x.try_f6().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::F7(x))=>self.forward_fixed::<Vec<Tensor<B,7>>>([x].into_iter().chain(input.map(|x|x.try_f7().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::F8(x))=>self.forward_fixed::<Vec<Tensor<B,8>>>([x].into_iter().chain(input.map(|x|x.try_f8().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::I1(x))=>self.forward_fixed::<Vec<Tensor<B,1,Int>>>([x].into_iter().chain(input.map(|x|x.try_i1().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::I2(x))=>self.forward_fixed::<Vec<Tensor<B,2,Int>>>([x].into_iter().chain(input.map(|x|x.try_i2().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::I3(x))=>self.forward_fixed::<Vec<Tensor<B,3,Int>>>([x].into_iter().chain(input.map(|x|x.try_i3().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::I4(x))=>self.forward_fixed::<Vec<Tensor<B,4,Int>>>([x].into_iter().chain(input.map(|x|x.try_i4().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::I5(x))=>self.forward_fixed::<Vec<Tensor<B,5,Int>>>([x].into_iter().chain(input.map(|x|x.try_i5().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::I6(x))=>self.forward_fixed::<Vec<Tensor<B,6,Int>>>([x].into_iter().chain(input.map(|x|x.try_i6().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::I7(x))=>self.forward_fixed::<Vec<Tensor<B,7,Int>>>([x].into_iter().chain(input.map(|x|x.try_i7().unwrap())).collect()).into_iter().map(Into::into).collect(),
			Some(Value::I8(x))=>self.forward_fixed::<Vec<Tensor<B,8,Int>>>([x].into_iter().chain(input.map(|x|x.try_i8().unwrap())).collect()).into_iter().map(Into::into).collect(),
			_=>todo!()
		}
	}
}
impl<B:Backend,E:Into<(Value<B>,Value<B>)>> Batcher<B,E,(Value<B>,Value<B>)> for BatchStacker{
	fn batch(&self,items:Vec<E>,_device:&<B as Backend>::Device)->(Value<B>,Value<B>){
		let items=items.into_iter().map(Into::into);
		let (input,target):(Vec<Value<B>>,Vec<Value<B>>)=items.unzip();

		let (input,target)=(Value::Multi(input),Value::Multi(target));
		(input.stack(0),target.stack(0))
	}
}
impl<B:Backend,K:BasicOps<B>+TensorKind<B>,const N:usize> AI<Vec<Tensor<B,N,K>>,Tensor<B,N,K>> for Cat<()>{
	fn forward(&self,input:Vec<Tensor<B,N,K>>)->Tensor<B,N,K>{Tensor::cat(input,self.dim())}
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
impl<B:Backend,W:AI<X,(Value<B>,Value<B>,Value<B>)>+Wrappable<B=B>,X> ValidStep<X,ClassificationOutput<B>> for Wrapped<Classification<W>> where W::Decomposition:Module<B>{
	fn step(&self,item:X)->ClassificationOutput<B>{self.forward(item)}
}
impl<B:Backend,W:AI<X,(Value<B>,Value<B>,Value<B>)>+Wrappable<B=B>,X> ValidStep<X,RegressionOutput<B>> for Wrapped<Regression<W>> where W::Decomposition:Module<B>{
	fn step(&self,item:X)->RegressionOutput<B>{self.forward(item)}
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
impl<B:Backend> AI<Value<B>,Value<B>> for AccQ<()>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{Value::B1(x)=>self.forward(x.float()).into(),Value::B2(x)=>self.forward(x.float()).into(),Value::B3(x)=>self.forward(x.float()).into(),Value::B4(x)=>self.forward(x.float()).into(),Value::B5(x)=>self.forward(x.float()).into(),Value::B6(x)=>self.forward(x.float()).into(),Value::B7(x)=>self.forward(x.float()).into(),Value::B8(x)=>self.forward(x.float()).into(),Value::F1(x)=>self.forward(x).into(),Value::F2(x)=>self.forward(x).into(),Value::F3(x)=>self.forward(x).into(),Value::F4(x)=>self.forward(x).into(),Value::F5(x)=>self.forward(x).into(),Value::F6(x)=>self.forward(x).into(),Value::F7(x)=>self.forward(x).into(),Value::F8(x)=>self.forward(x).into(),Value::I1(x)=>self.forward(x.float()).into(),Value::I2(x)=>self.forward(x.float()).into(),Value::I3(x)=>self.forward(x.float()).into(),Value::I4(x)=>self.forward(x.float()).into(),Value::I5(x)=>self.forward(x.float()).into(),Value::I6(x)=>self.forward(x.float()).into(),Value::I7(x)=>self.forward(x.float()).into(),Value::I8(x)=>self.forward(x.float()).into(),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>Value::Multi(x.into_iter().map(|x|self.forward(x)).collect())}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for SoftChoose<()>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{Value::B1(x)=>self.forward_typed::<Tensor<B,1>,Tensor<B,1,Int>>(x.float()).into(),Value::B2(x)=>self.forward_typed::<Tensor<B,2>,Tensor<B,1,Int>>(x.float()).into(),Value::B3(x)=>self.forward_typed::<Tensor<B,3>,Tensor<B,2,Int>>(x.float()).into(),Value::B4(x)=>self.forward_typed::<Tensor<B,4>,Tensor<B,3,Int>>(x.float()).into(),Value::B5(x)=>self.forward_typed::<Tensor<B,5>,Tensor<B,4,Int>>(x.float()).into(),Value::B6(x)=>self.forward_typed::<Tensor<B,6>,Tensor<B,5,Int>>(x.float()).into(),Value::B7(x)=>self.forward_typed::<Tensor<B,7>,Tensor<B,6,Int>>(x.float()).into(),Value::B8(x)=>self.forward_typed::<Tensor<B,8>,Tensor<B,7,Int>>(x.float()).into(),Value::F1(x)=>self.forward_typed::<Tensor<B,1>,Tensor<B,1,Int>>(x).into(),Value::F2(x)=>self.forward_typed::<Tensor<B,2>,Tensor<B,1,Int>>(x).into(),Value::F3(x)=>self.forward_typed::<Tensor<B,3>,Tensor<B,2,Int>>(x).into(),Value::F4(x)=>self.forward_typed::<Tensor<B,4>,Tensor<B,3,Int>>(x).into(),Value::F5(x)=>self.forward_typed::<Tensor<B,5>,Tensor<B,4,Int>>(x).into(),Value::F6(x)=>self.forward_typed::<Tensor<B,6>,Tensor<B,5,Int>>(x).into(),Value::F7(x)=>self.forward_typed::<Tensor<B,7>,Tensor<B,6,Int>>(x).into(),Value::F8(x)=>self.forward_typed::<Tensor<B,8>,Tensor<B,7,Int>>(x).into(),Value::I1(x)=>self.forward_typed::<Tensor<B,1>,Tensor<B,1,Int>>(x.float()).into(),Value::I2(x)=>self.forward_typed::<Tensor<B,2>,Tensor<B,1,Int>>(x.float()).into(),Value::I3(x)=>self.forward_typed::<Tensor<B,3>,Tensor<B,2,Int>>(x.float()).into(),Value::I4(x)=>self.forward_typed::<Tensor<B,4>,Tensor<B,3,Int>>(x.float()).into(),Value::I5(x)=>self.forward_typed::<Tensor<B,5>,Tensor<B,4,Int>>(x.float()).into(),Value::I6(x)=>self.forward_typed::<Tensor<B,6>,Tensor<B,5,Int>>(x.float()).into(),Value::I7(x)=>self.forward_typed::<Tensor<B,7>,Tensor<B,6,Int>>(x.float()).into(),Value::I8(x)=>self.forward_typed::<Tensor<B,8>,Tensor<B,7,Int>>(x.float()).into(),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>Value::Multi(x.into_iter().map(|x|self.forward(x)).collect())}
	}
}
impl<B:Backend> AI<Value<B>,u32> for SoftChoose<()>{
	fn forward(&self,input:Value<B>)->Value<B>{
		let _todo=input;
		todo!()
		//match input{Value::B1(x)=>self.forward_typed::<Tensor<B,1>,Tensor<B,1,Int>>(x.float()).into(),Value::B2(x)=>self.forward_typed::<Tensor<B,2>,Tensor<B,1,Int>>(x.float()).into(),Value::B3(x)=>self.forward_typed::<Tensor<B,3>,Tensor<B,2,Int>>(x.float()).into(),Value::B4(x)=>self.forward_typed::<Tensor<B,4>,Tensor<B,3,Int>>(x.float()).into(),Value::B5(x)=>self.forward_typed::<Tensor<B,5>,Tensor<B,4,Int>>(x.float()).into(),Value::B6(x)=>self.forward_typed::<Tensor<B,6>,Tensor<B,5,Int>>(x.float()).into(),Value::B7(x)=>self.forward_typed::<Tensor<B,7>,Tensor<B,6,Int>>(x.float()).into(),Value::B8(x)=>self.forward_typed::<Tensor<B,8>,Tensor<B,7,Int>>(x.float()).into(),Value::F1(x)=>self.forward_typed::<Tensor<B,1>,Tensor<B,1,Int>>(x).into(),Value::F2(x)=>self.forward_typed::<Tensor<B,2>,Tensor<B,1,Int>>(x).into(),Value::F3(x)=>self.forward_typed::<Tensor<B,3>,Tensor<B,2,Int>>(x).into(),Value::F4(x)=>self.forward_typed::<Tensor<B,4>,Tensor<B,3,Int>>(x).into(),Value::F5(x)=>self.forward_typed::<Tensor<B,5>,Tensor<B,4,Int>>(x).into(),Value::F6(x)=>self.forward_typed::<Tensor<B,6>,Tensor<B,5,Int>>(x).into(),Value::F7(x)=>self.forward_typed::<Tensor<B,7>,Tensor<B,6,Int>>(x).into(),Value::F8(x)=>self.forward_typed::<Tensor<B,8>,Tensor<B,7,Int>>(x).into(),Value::I1(x)=>self.forward_typed::<Tensor<B,1>,Tensor<B,1,Int>>(x.float()).into(),Value::I2(x)=>self.forward_typed::<Tensor<B,2>,Tensor<B,1,Int>>(x.float()).into(),Value::I3(x)=>self.forward_typed::<Tensor<B,3>,Tensor<B,2,Int>>(x.float()).into(),Value::I4(x)=>self.forward_typed::<Tensor<B,4>,Tensor<B,3,Int>>(x.float()).into(),Value::I5(x)=>self.forward_typed::<Tensor<B,5>,Tensor<B,4,Int>>(x.float()).into(),Value::I6(x)=>self.forward_typed::<Tensor<B,6>,Tensor<B,5,Int>>(x.float()).into(),Value::I7(x)=>self.forward_typed::<Tensor<B,7>,Tensor<B,6,Int>>(x.float()).into(),Value::I8(x)=>self.forward_typed::<Tensor<B,8>,Tensor<B,7,Int>>(x.float()).into(),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>Value::Multi(x.into_iter().map(|x|self.forward(x)).collect())}
	}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,Tensor<B,N>> for AccQ<()>{
	fn forward(&self,input:Tensor<B,N>)->Tensor<B,N>{accumulate_q_burn(self.dim(),self.gamma(),input)}
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
impl<B:Backend,const N:usize> AI<Tensor<B,N>,Vec<u32>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,N>)->Vec<u32>{soft_choose_burn_multi(self.dim(),input,self.temperature())}
}
impl<B:Backend,const N:usize> AI<Tensor<B,N>,u32> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,N>)->u32{soft_choose_burn_1(self.dim(),input,self.temperature())}
}
impl<B:Backend> AI<(Tensor<B,2>,Tensor<B,1,Int>),Tensor<B,1>> for CrossEntropyLoss<B>{// TODO soft label version
	fn forward(&self,(output,target):(Tensor<B,2>,Tensor<B,1,Int>))->Tensor<B,1>{self.forward(output,target)}
}
impl<B:Backend> AI<(Tensor<B,3>,Tensor<B,2,Int>),Tensor<B,1>> for CrossEntropyLoss<B>{
	fn forward(&self,(output,target):(Tensor<B,3>,Tensor<B,2,Int>))->Tensor<B,1>{self.forward(output.flatten(0,1),target.flatten(0,1))}
}
impl<B:Backend> AI<(Tensor<B,4>,Tensor<B,3,Int>),Tensor<B,1>> for CrossEntropyLoss<B>{
	fn forward(&self,(output,target):(Tensor<B,4>,Tensor<B,3,Int>))->Tensor<B,1>{self.forward(output.flatten(0,2),target.flatten(0,2))}
}
impl<B:Backend> AI<(Tensor<B,5>,Tensor<B,4,Int>),Tensor<B,1>> for CrossEntropyLoss<B>{
	fn forward(&self,(output,target):(Tensor<B,5>,Tensor<B,4,Int>))->Tensor<B,1>{self.forward(output.flatten(0,3),target.flatten(0,3))}
}
impl<B:Backend> AI<(Tensor<B,6>,Tensor<B,5,Int>),Tensor<B,1>> for CrossEntropyLoss<B>{
	fn forward(&self,(output,target):(Tensor<B,6>,Tensor<B,5,Int>))->Tensor<B,1>{self.forward(output.flatten(0,4),target.flatten(0,4))}
}
impl<B:Backend> AI<(Tensor<B,7>,Tensor<B,6,Int>),Tensor<B,1>> for CrossEntropyLoss<B>{
	fn forward(&self,(output,target):(Tensor<B,7>,Tensor<B,6,Int>))->Tensor<B,1>{self.forward(output.flatten(0,5),target.flatten(0,5))}
}
impl<B:Backend> AI<(Tensor<B,8>,Tensor<B,7,Int>),Tensor<B,1>> for CrossEntropyLoss<B>{
	fn forward(&self,(output,target):(Tensor<B,8>,Tensor<B,7,Int>))->Tensor<B,1>{self.forward(output.flatten(0,6),target.flatten(0,6))}
}
impl<B:Backend> AI<(Value<B>,Value<B>,Value<B>),ClassificationOutput<B>> for Classification<()>{
	fn forward(&self,(loss,output,target):(Value<B>,Value<B>,Value<B>))->ClassificationOutput<B>{
		let loss=match loss{Value::F1(x)=>x,Value::F2(x)=>x.mean(),Value::F3(x)=>x.mean(),Value::F4(x)=>x.mean(),Value::F5(x)=>x.mean(),Value::F6(x)=>x.mean(),Value::F7(x)=>x.mean(),Value::F8(x)=>x.mean(),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let output=match output{Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(0,1),Value::F4(x)=>x.flatten(0,2),Value::F5(x)=>x.flatten(0,3),Value::F6(x)=>x.flatten(0,4),Value::F7(x)=>x.flatten(0,5),Value::F8(x)=>x.flatten(0,6),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let target=match target{Value::I1(x)=>x,Value::I2(x)=>x.flatten(0,1),Value::I3(x)=>x.flatten(0,2),Value::I4(x)=>x.flatten(0,3),Value::I5(x)=>x.flatten(0,4),Value::I6(x)=>x.flatten(0,5),Value::I7(x)=>x.flatten(0,6),Value::I8(x)=>x.flatten(0,7),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		ClassificationOutput::new(loss,output,target)
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>,Value<B>),RegressionOutput<B>> for Regression<()>{
	fn forward(&self,(loss,output,target):(Value<B>,Value<B>,Value<B>))->RegressionOutput<B>{
		let loss=match loss{Value::F1(x)=>x,Value::F2(x)=>x.mean(),Value::F3(x)=>x.mean(),Value::F4(x)=>x.mean(),Value::F5(x)=>x.mean(),Value::F6(x)=>x.mean(),Value::F7(x)=>x.mean(),Value::F8(x)=>x.mean(),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let output=match output{Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(0,1),Value::F4(x)=>x.flatten(0,2),Value::F5(x)=>x.flatten(0,3),Value::F6(x)=>x.flatten(0,4),Value::F7(x)=>x.flatten(0,5),Value::F8(x)=>x.flatten(0,6),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let target=match target{Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(0,1),Value::F4(x)=>x.flatten(0,2),Value::F5(x)=>x.flatten(0,3),Value::F6(x)=>x.flatten(0,4),Value::F7(x)=>x.flatten(0,5),Value::F8(x)=>x.flatten(0,6),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		RegressionOutput::new(loss,output,target)
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),(Value<B>,Value<B>,Value<B>)> for CrossEntropy<()>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->(Value<B>,Value<B>,Value<B>){
		let loss=CrossEntropyLossConfig::new().init(&Default::default()).fix_type::<Value<B>>().forward(Value::Multi(vec![output.clone(),target.clone()]));
		(loss,output,target)
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),(Value<B>,Value<B>,Value<B>)> for MSE<()>{//TODO make into just cross entropy and squared error and add a mean op for mse
	fn forward(&self,(output,target):(Value<B>,Value<B>))->(Value<B>,Value<B>,Value<B>){
		let loss=MseLoss.fix_type::<Value<B>>().forward(Value::Multi(vec![output.clone(),target.clone()]));
		(loss,output,target)
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),f32> for CrossEntropy<()>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->f32{
		fn average_scalars<B:Backend>(value:Value<B>)->f32{
			match value{
				Value::F1(x)=>x.into_scalar().elem(),
				Value::Multi(x)=>{
					let l=x.len();
					let x:f32=x.into_iter().map(average_scalars).sum();
					x/l as f32
				},
				Value::Incompatible(x)=>panic!("{x}"),
				_=>panic!("internal error: cross entropy loss output should be scalar")
			}
		}
		average_scalars(CrossEntropyLossConfig::new().init(&Default::default()).fix_type::<Value<B>>().forward(Value::Multi(vec![output,target])))
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),f32> for MSE<()>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->f32{
		fn average_scalars<B:Backend>(value:Value<B>)->f32{
			match value{
				Value::F1(x)=>x.into_scalar().elem(),
				Value::Multi(x)=>{
					let l=x.len();
					let x:f32=x.into_iter().map(average_scalars).sum();
					x/l as f32
				},
				Value::Incompatible(x)=>panic!("{x}"),
				_=>panic!("internal error: mse loss output should be scalar")
			}
		}
		average_scalars(MseLoss.fix_type::<Value<B>>().forward(Value::Multi(vec![output,target])))
	}
}
impl<B:Backend> AI<Tensor<B,1>,Tensor<B,1,Int>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,1>)->Tensor<B,1,Int>{
		let (d,t)=(self.dim(),self.temperature());
		soft_choose_burn_tensor(d,input,t)
	}
}
impl<B:Backend> AI<Tensor<B,2>,Tensor<B,1,Int>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,2>)->Tensor<B,1,Int>{
		let (d,t)=(self.dim(),self.temperature());
		soft_choose_burn_tensor(d,input,t).squeeze(d)
	}
}
impl<B:Backend> AI<Tensor<B,3>,Tensor<B,2,Int>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,3>)->Tensor<B,2,Int>{
		let (d,t)=(self.dim(),self.temperature());
		soft_choose_burn_tensor(d,input,t).squeeze(d)
	}
}
impl<B:Backend> AI<Tensor<B,4>,Tensor<B,3,Int>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,4>)->Tensor<B,3,Int>{
		let (d,t)=(self.dim(),self.temperature());
		soft_choose_burn_tensor(d,input,t).squeeze(d)
	}
}
impl<B:Backend> AI<Tensor<B,5>,Tensor<B,4,Int>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,5>)->Tensor<B,4,Int>{
		let (d,t)=(self.dim(),self.temperature());
		soft_choose_burn_tensor(d,input,t).squeeze(d)
	}
}
impl<B:Backend> AI<Tensor<B,6>,Tensor<B,5,Int>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,6>)->Tensor<B,5,Int>{
		let (d,t)=(self.dim(),self.temperature());
		soft_choose_burn_tensor(d,input,t).squeeze(d)
	}
}
impl<B:Backend> AI<Tensor<B,7>,Tensor<B,6,Int>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,7>)->Tensor<B,6,Int>{
		let (d,t)=(self.dim(),self.temperature());
		soft_choose_burn_tensor(d,input,t).squeeze(d)
	}
}
impl<B:Backend> AI<Tensor<B,8>,Tensor<B,7,Int>> for SoftChoose<()>{
	fn forward(&self,input:Tensor<B,8>)->Tensor<B,7,Int>{
		let (d,t)=(self.dim(),self.temperature());
		soft_choose_burn_tensor(d,input,t).squeeze(d)
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for CrossEntropyLoss<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{
			Value::Incompatible(x)=>x.into(),
			Value::Multi(x)=>if x.len()==2{
				let mut x=x.into_iter();
				match (x.next().unwrap(),x.next().unwrap()){(Value::F2(x0),Value::I1(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F3(x0),Value::I2(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F4(x0),Value::I3(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F5(x0),Value::I4(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F6(x0),Value::I5(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F7(x0),Value::I6(x1))=>AI::forward(self,(x0,x1)).into(),(Value::F8(x0),Value::I7(x1))=>AI::forward(self,(x0,x1)).into(),_=>"cross entropy loss requires input pairs to be a float tensor with an int tensor of one lower rank".into()}
			}else{
				let y:Vec<Value<B>>=x.into_iter().map(|x|self.fix_type::<Value<B>>().forward(x)).collect();
				y.into()
			},
			_=>"cross entropy loss loss requires inputs to be in pairs".into()
		}
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
impl<B:Backend> AI<Value<B>,Value<B>> for Layer<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match self{Layer::CrossEntropy(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Dropout(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Embedding(a)=>a.fix_type::<Value<B>>().forward(input),Layer::LayerNorm(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Linear(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Mse(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Relu(a)=>a.fix_type::<Value<B>>().forward(input),Layer::Stack(dim)=>input.stack(*dim)}
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
impl<B:Backend> IntoIterator for Value<B>{
	fn into_iter(self)->Self::IntoIter{self.into_multi().into_iter()}
	type IntoIter=VecIntoIter<Value<B>>;
	type Item=Value<B>;
}
impl<B:Backend> Layer<B>{
	/// creates a linear layer
	pub fn linear(bias:bool,input:usize,output:usize,wscale:f32)->Self{
		let mut l=LinearConfig::new(input,output).with_bias(bias);
		if wscale!=1.0{l.initializer=w_scale(l.initializer,wscale)}
		let l=l.init(&Default::default());
		Self::Linear(l)
	}
	/// creates a relu layer
	pub fn relu()->Self{Self::Relu(Relu)}
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
impl<B:Backend> Op for CrossEntropyLoss<B>{
	type Output=Tensor<B,1>;
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
impl<B:Backend> Value<B>{// TODO more builtin functions // TODO shape
	/// recursively counts the number of tensors within this value, including multi tensors within multi tensors
	pub fn count_recursive(&self)->usize{
		if let Value::Multi(v)=self{v.iter().map(Value::count_recursive).sum()}else{1}
	}
	/// converts to a multiple tensor, then unwraps to a vec of values
	pub fn into_multi(self)->Vec<Value<B>>{
		if let Value::Multi(v)=self{v}else{vec![self]}
	}
	/// tests if this is a multiple tensor
	pub fn is_multi(&self)->bool{
		if let Value::Multi(_x)=self{true}else{false}
	}
	/// returns a shallow count the number of values directly within this one. 1 if not multi, otherwise the len of the vec inside.
	pub fn len(&self)->usize{
		if let Value::Multi(v)=self{v.len()}else{1}
	}
	/// converts to a multiple tensor if not one
	pub fn make_multi(self)->Self{
		if let Value::Multi(v)=self{v.into()}else{vec![self].into()}
	}
	/// stacks the multi tensor, inserting a dimension at d. for singular tensors this has an unsqueezing effect
	pub fn stack(self,d:usize)->Value<B>{//TODO macros could make this look less repetitive
		fn append<X>(mut v:Vec<X>,x:X)->Vec<X>{
			v.push(x);
			v
		}
		match match self{
			Value::B1(x)=>Ok(Value::B2(x.unsqueeze_dim(d))),
			Value::B2(x)=>Ok(Value::B3(x.unsqueeze_dim(d))),
			Value::B3(x)=>Ok(Value::B4(x.unsqueeze_dim(d))),
			Value::B4(x)=>Ok(Value::B5(x.unsqueeze_dim(d))),
			Value::B5(x)=>Ok(Value::B6(x.unsqueeze_dim(d))),
			Value::B6(x)=>Ok(Value::B7(x.unsqueeze_dim(d))),
			Value::B7(x)=>Ok(Value::B8(x.unsqueeze_dim(d))),
			Value::B8(_x)=>Ok("currently cannot increase number of tensor dimensions above 8".into()),
			Value::F1(x)=>Ok(Value::F2(x.unsqueeze_dim(d))),
			Value::F2(x)=>Ok(Value::F3(x.unsqueeze_dim(d))),
			Value::F3(x)=>Ok(Value::F4(x.unsqueeze_dim(d))),
			Value::F4(x)=>Ok(Value::F5(x.unsqueeze_dim(d))),
			Value::F5(x)=>Ok(Value::F6(x.unsqueeze_dim(d))),
			Value::F6(x)=>Ok(Value::F7(x.unsqueeze_dim(d))),
			Value::F7(x)=>Ok(Value::F8(x.unsqueeze_dim(d))),
			Value::F8(_x)=>Ok("currently cannot increase number of tensor dimensions above 8".into()),
			Value::I1(x)=>Ok(Value::I2(x.unsqueeze_dim(d))),
			Value::I2(x)=>Ok(Value::I3(x.unsqueeze_dim(d))),
			Value::I3(x)=>Ok(Value::I4(x.unsqueeze_dim(d))),
			Value::I4(x)=>Ok(Value::I5(x.unsqueeze_dim(d))),
			Value::I5(x)=>Ok(Value::I6(x.unsqueeze_dim(d))),
			Value::I6(x)=>Ok(Value::I7(x.unsqueeze_dim(d))),
			Value::I7(x)=>Ok(Value::I8(x.unsqueeze_dim(d))),
			Value::I8(_x)=>Ok("currently cannot increase number of tensor dimensions above 8".into()),
			Value::Incompatible(x)=>Ok(x.into()),
			Value::Multi(x)=>if x.is_empty(){
				Ok(x.into())
			}else if x.iter().all(Value::is_multi){
				Ok(Value::Multi(x.into_iter().map(|x|x.stack(d)).collect()))
			}else{
				let xl=x.len();
				let mut x=x.into_iter();
				match x.next().unwrap(){
					Value::B1(x0)=>x.map(Value::try_b1).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::B2(Tensor::stack(v,d))),
					Value::B2(x0)=>x.map(Value::try_b2).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::B3(Tensor::stack(v,d))),
					Value::B3(x0)=>x.map(Value::try_b3).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::B4(Tensor::stack(v,d))),
					Value::B4(x0)=>x.map(Value::try_b4).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::B5(Tensor::stack(v,d))),
					Value::B5(x0)=>x.map(Value::try_b5).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::B6(Tensor::stack(v,d))),
					Value::B6(x0)=>x.map(Value::try_b6).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::B7(Tensor::stack(v,d))),
					Value::B7(x0)=>x.map(Value::try_b7).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::B8(Tensor::stack(v,d))),
					Value::B8(_x)=>Ok("currently cannot increase number of tensor dimensions above 8".into()),
					Value::F1(x0)=>x.map(Value::try_f1).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::F2(Tensor::stack(v,d))),
					Value::F2(x0)=>x.map(Value::try_f2).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::F3(Tensor::stack(v,d))),
					Value::F3(x0)=>x.map(Value::try_f3).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::F4(Tensor::stack(v,d))),
					Value::F4(x0)=>x.map(Value::try_f4).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::F5(Tensor::stack(v,d))),
					Value::F5(x0)=>x.map(Value::try_f5).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::F6(Tensor::stack(v,d))),
					Value::F6(x0)=>x.map(Value::try_f6).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::F7(Tensor::stack(v,d))),
					Value::F7(x0)=>x.map(Value::try_f7).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::F8(Tensor::stack(v,d))),
					Value::F8(_x)=>Ok("currently cannot increase number of tensor dimensions above 8".into()),
					Value::I1(x0)=>x.map(Value::try_i1).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::I2(Tensor::stack(v,d))),
					Value::I2(x0)=>x.map(Value::try_i2).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::I3(Tensor::stack(v,d))),
					Value::I3(x0)=>x.map(Value::try_i3).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::I4(Tensor::stack(v,d))),
					Value::I4(x0)=>x.map(Value::try_i4).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::I5(Tensor::stack(v,d))),
					Value::I5(x0)=>x.map(Value::try_i5).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::I6(Tensor::stack(v,d))),
					Value::I6(x0)=>x.map(Value::try_i6).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::I7(Tensor::stack(v,d))),
					Value::I7(x0)=>x.map(Value::try_i7).try_fold(append(Vec::with_capacity(xl),x0),|acc,x|x.map(|x|append(acc,x))).map(|v|Value::I8(Tensor::stack(v,d))),
					Value::I8(_x)=>Ok("currently cannot increase number of tensor dimensions above 8".into()),
					Value::Incompatible(x0)=>Ok(x0.into()),
					Value::Multi(x0)=>Err(x0.into())
				}
			},
		}{
			Err(Value::Incompatible(x))=>x.into(),
			Err(_)=>"incompatible shapes or types for stacking".into(),//TODO more helpful debug info
			Ok(y)=>y,
		}
	}
	/// attempts to unwrap the inner B1 value
	pub fn try_b1(self)->Result<Tensor<B,1,Bool>,Self>{
		if let Value::B1(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B2 value
	pub fn try_b2(self)->Result<Tensor<B,2,Bool>,Self>{
		if let Value::B2(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B3 value
	pub fn try_b3(self)->Result<Tensor<B,3,Bool>,Self>{
		if let Value::B3(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B4 value
	pub fn try_b4(self)->Result<Tensor<B,4,Bool>,Self>{
		if let Value::B4(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B5 value
	pub fn try_b5(self)->Result<Tensor<B,5,Bool>,Self>{
		if let Value::B5(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B6 value
	pub fn try_b6(self)->Result<Tensor<B,6,Bool>,Self>{
		if let Value::B6(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B7 value
	pub fn try_b7(self)->Result<Tensor<B,7,Bool>,Self>{
		if let Value::B7(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B8 value
	pub fn try_b8(self)->Result<Tensor<B,8,Bool>,Self>{
		if let Value::B8(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner F1 value
	pub fn try_f1(self)->Result<Tensor<B,1,Float>,Self>{
		if let Value::F1(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner F2 value
	pub fn try_f2(self)->Result<Tensor<B,2,Float>,Self>{
		if let Value::F2(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner F3 value
	pub fn try_f3(self)->Result<Tensor<B,3,Float>,Self>{
		if let Value::F3(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner F4 value
	pub fn try_f4(self)->Result<Tensor<B,4,Float>,Self>{
		if let Value::F4(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner F5 value
	pub fn try_f5(self)->Result<Tensor<B,5,Float>,Self>{
		if let Value::F5(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner F6 value
	pub fn try_f6(self)->Result<Tensor<B,6,Float>,Self>{
		if let Value::F6(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner F7 value
	pub fn try_f7(self)->Result<Tensor<B,7,Float>,Self>{
		if let Value::F7(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner F8 value
	pub fn try_f8(self)->Result<Tensor<B,8,Float>,Self>{
		if let Value::F8(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner I1 value
	pub fn try_i1(self)->Result<Tensor<B,1,Int>,Self>{
		if let Value::I1(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner I2 value
	pub fn try_i2(self)->Result<Tensor<B,2,Int>,Self>{
		if let Value::I2(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner I3 value
	pub fn try_i3(self)->Result<Tensor<B,3,Int>,Self>{
		if let Value::I3(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner I4 value
	pub fn try_i4(self)->Result<Tensor<B,4,Int>,Self>{
		if let Value::I4(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner I5 value
	pub fn try_i5(self)->Result<Tensor<B,5,Int>,Self>{
		if let Value::I5(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner I6 value
	pub fn try_i6(self)->Result<Tensor<B,6,Int>,Self>{
		if let Value::I6(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner I7 value
	pub fn try_i7(self)->Result<Tensor<B,7,Int>,Self>{
		if let Value::I7(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner I8 value
	pub fn try_i8(self)->Result<Tensor<B,8,Int>,Self>{
		if let Value::I8(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner multi value
	pub fn try_multi(self)->Result<Vec<Value<B>>,Self>{
		if let Value::Multi(v)=self{Ok(v)}else{Err(self)}
	}
}
impl<B:Backend> Wrappable for Layer<B>{
	type B=B;
	type With<C:Backend>=Layer<C>;
}
impl<B:Backend> Wrappable for Value<B>{
	type B=B;
	type With<C:Backend>=Value<C>;
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
impl<W:Wrappable> Wrappable for Classification<W>{
	type B=W::B;
	type With<C:Backend>=Classification<W::With<C>>;
}
impl<W:Wrappable> Wrappable for CrossEntropy<W>{
	type B=W::B;
	type With<C:Backend>=CrossEntropy<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Duplicate<W>{
	type B=W::B;
	type With<C:Backend>=Duplicate<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Graph<W>{
	type B=W::B;
	type With<C:Backend>=Graph<W::With<C>>;
}
impl<W:Wrappable> Wrappable for MSE<W>{
	type B=W::B;
	type With<C:Backend>=MSE<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Map<W>{
	type B=W::B;
	type With<C:Backend>=Map<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Regression<W>{
	type B=W::B;
	type With<C:Backend>=Regression<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Sequential<W>{
	type B=W::B;
	type With<C:Backend>=Sequential<W::With<C>>;
}
impl<W:Wrappable> Wrappable for SoftChoose<W>{
	type B=W::B;
	type With<C:Backend>=SoftChoose<W::With<C>>;
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
#[cfg(test)]
mod tests{
	#[test]
	fn learn_xor(){
		type B=NdArray;
		type A=Autodiff<B>;
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
		let graph=Unvec(graph).mse().set_type::<(Value<A>,Value<A>),(Value<A>,Value<A>,Value<A>)>().regression().wrap();
		let graph=graph.train(&TrainConfig::new(),SgdConfig::new().init(),0.01,train,valid);
		let graph=graph.valid().into_inner().into_inner().into_inner().into_inner();

		let inputval=Value::from(Tensor::<B,2>::from_data(TensorData::new([0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0].to_vec(),[4,2]),&Default::default()));
		let outputval=graph.forward(inputval);
		if let Value::F2(o)=outputval{
			let target=Tensor::<B,2>::from_data(TensorData::new([0.0,1.0,1.0,0.0].to_vec(),[4,1]),&Default::default());
			let error=(target-o.clone()).abs().max();
			println!("{}",o);
			assert!(error.into_scalar()<0.1);
		}else{
			panic!("h");
		}
	}
	use burn::{
		backend::Autodiff,data::dataset::InMemDataset,optim::SgdConfig
	};
	use crate::graph::VertexLabels;
	use super::*;
}
/// scales the initializer
pub fn w_scale(initializer:Initializer,r:f32)->Initializer{
	let r=r as f64;// apparently
	match initializer{
		Initializer::Constant{value}=>Initializer::Constant{value:value*r},
		Initializer::KaimingNormal{gain,fan_out_only}=>Initializer::KaimingNormal{gain:gain*r,fan_out_only},
		Initializer::KaimingUniform{gain,fan_out_only}=>Initializer::KaimingUniform{gain:gain*r,fan_out_only},
		Initializer::Normal{mean,std}=>Initializer::Normal{mean:mean*r,std:std*r},
		Initializer::Ones=>Initializer::Constant{value:r},
		Initializer::Uniform{min,max}=>Initializer::Uniform{min:min*r,max:max*r},
		Initializer::XavierNormal{gain}=>Initializer::XavierNormal{gain:gain*r},
		Initializer::XavierUniform{gain}=>Initializer::XavierUniform{gain:gain*r},
		Initializer::Zeros=>Initializer::Zeros
	}
}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
/// batcher that stacks things
pub struct BatchStacker;
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// wrapper for converting loss to classification output
pub struct Classification<A>{inner:A}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
/// metrics renderer implementation that doesn't actually do anything
pub struct DontRender;
#[derive(Config)]
/// enumerates config for some burn layers
pub enum Config{CrossEntropy(CrossEntropyLossConfig),Dropout(DropoutConfig),Embedding(EmbeddingConfig),LayerNorm(LayerNormConfig),Linear(LinearConfig),Mse,Relu,Stack(usize)}
#[derive(Debug,Module)]//TODO more layers
/// enumerates some burn layers
pub enum Layer<B:Backend>{CrossEntropy(CrossEntropyLoss<B>),Dropout(Dropout),Embedding(Embedding<B>),LayerNorm(LayerNorm<B>),Linear(Linear<B>),Mse(MseLoss),Relu(Relu),Stack(usize)}
#[derive(Clone,Debug)]//TODO implement module for this
/// enumerates burn tensors up to 8 dimensions
pub enum Value<B:Backend>{B1(Tensor<B,1,Bool>),B2(Tensor<B,2,Bool>),B3(Tensor<B,3,Bool>),B4(Tensor<B,4,Bool>),B5(Tensor<B,5,Bool>),B6(Tensor<B,6,Bool>),B7(Tensor<B,7,Bool>),B8(Tensor<B,8,Bool>),F1(Tensor<B,1,Float>),F2(Tensor<B,2,Float>),F3(Tensor<B,3,Float>),F4(Tensor<B,4,Float>),F5(Tensor<B,5,Float>),F6(Tensor<B,6,Float>),F7(Tensor<B,7,Float>),F8(Tensor<B,8,Float>),I1(Tensor<B,1,Int>),I2(Tensor<B,2,Int>),I3(Tensor<B,3,Int>),I4(Tensor<B,4,Int>),I5(Tensor<B,5,Int>),I6(Tensor<B,6,Int>),I7(Tensor<B,7,Int>),I8(Tensor<B,8,Int>),Incompatible(String),Multi(Vec<Self>)}
#[derive(Clone,Copy,Debug,Default,Eq,Hash,Ord,PartialEq,PartialOrd)]
#[repr(transparent)]
/// wrapper for converting loss to regression output
pub struct Regression<A>{inner:A}
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
	/// wraps in a classification wrapper
	fn classification(self)->Classification<Self> where Classification<Self>:Op,Self:Sized{Classification::from_inner(self)}
	/// wraps in a regression wrapper
	fn regression(self)->Regression<Self> where Regression<Self>:Op,Self:Sized{Regression::from_inner(self)}
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
		Dropout,DropoutConfig,Embedding,EmbeddingConfig,Initializer,LayerNorm,LayerNormConfig,Linear,LinearConfig,Relu,loss::{CrossEntropyLoss,CrossEntropyLossConfig,MseLoss}
	},
	optim::Optimizer,
	prelude::*,
	record::{CompactRecorder,FileRecorder,RecorderError},
	tensor::{BasicOps,TensorKind,activation::softmax,backend::AutodiffBackend},
	train::{
		ClassificationOutput,LearnerBuilder,RegressionOutput,TrainOutput,TrainStep,ValidStep,metric::{Adaptor,ItemLazy,LossInput,LossMetric},renderer::{MetricState,MetricsRenderer,TrainingProgress}
	}
};
use crate::{
	ai::{AI,AccQ,Alignment,Branch,Cat,CrossEntropy,Decompose,Duplicate,Map,MSE,Op,Sequential,SetType,SoftChoose,TruncateToMatch,WhichDims,Zip},graph::{Graph,Merge,Unvec}
};
use rand::random;
use std::{
	fmt::{Debug,Display},fs::{create_dir_all as create_folder},iter::FromIterator,mem::take,path::PathBuf,vec::IntoIter as VecIntoIter
};
