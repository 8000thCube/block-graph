impl Decompose for ClassificationLayer{
	fn compose(_decomposition:Self::Decomposition)->Self{Self::default()}
	fn decompose(self){}
	fn decompose_cloned(&self){}
	type Decomposition=();
}
impl Decompose for RegressionLayer{
	fn compose(_decomposition:Self::Decomposition)->Self{Self::default()}
	fn decompose(self){}
	fn decompose_cloned(&self){}
	type Decomposition=();
}
impl MetricsRenderer for DontRender{
	fn update_train(&mut self,_state:MetricState){}
	fn update_valid(&mut self,_state:MetricState){}
	fn render_train(&mut self,_item:TrainingProgress){}
	fn render_valid(&mut self,_item:TrainingProgress){}
}
impl Op for ClassificationLayer{
	type Output=ClassificationOutput<NdArray>;
}
impl Op for RegressionLayer{
	type Output=RegressionOutput<NdArray>;
}
impl<A:AI<X,LossOutput<B>>,B:Backend,X> AI<X,ClassificationOutput<B>> for Classification<A>{
	fn forward(&self,input:X)->ClassificationOutput<B>{self.layer.forward(self.inner.forward(input))}
	fn forward_mut(&mut self,input:X)->ClassificationOutput<B>{self.layer.forward(self.inner.forward_mut(input))}
}
impl<A:AI<X,LossOutput<B>>,B:Backend,X> AI<X,RegressionOutput<B>> for Regression<A>{
	fn forward(&self,input:X)->RegressionOutput<B>{self.layer.forward(self.inner.forward(input))}
	fn forward_mut(&mut self,input:X)->RegressionOutput<B>{self.layer.forward(self.inner.forward_mut(input))}
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,W:'static+Wrappable<B=A>,Y:'static+ItemLazy+Send+Sync,Z:'static+ItemLazy+Send+Sync> Wrapped<W> where <Self as AutodiffModule<A>>::InnerModule:ValidStep<(Value<B>,Value<B>),Z>,Self:TrainStep<(Value<A>,Value<A>),Y>,W::Decomposition:AutodiffModule<A>,W::With<B>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>+Op<Output=Z>,W:Op<Output=Y>,Y::ItemSync:Adaptor<LossInput<NdArray>>,Z::ItemSync:Adaptor<LossInput<NdArray>>{
	/// trains the model
	pub fn train<I:'static+Clone+Debug+Into<(Value<A>,Value<A>)>+Send+Sync,J:'static+Clone+Debug+Into<(Value<B>,Value<B>)>+Send+Sync,O:Optimizer<Self,A>,S:LrScheduler,T:'static+Dataset<I>,V:'static+Dataset<J>>(self,config:&TrainConfig,optimizer:O,scheduler:S,train:T,valid:V)->Self where O::Record:'static,S::Record<A>:'static{
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
impl<A:AutodiffBackend,W:AI<X,LossOutput<A>>+Wrappable<B=A>,X> TrainStep<X,ClassificationOutput<A>> for Wrapped<Classification<W>> where W::Decomposition:AutodiffModule<A>,W::With<A::InnerBackend>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>{
	fn step(&self,item:X)->TrainOutput<ClassificationOutput<A>>{
		let output:ClassificationOutput<A>=self.forward(item);
		TrainOutput::new(self,output.loss.backward(),output)
	}
}
impl<A:AutodiffBackend,W:AI<X,LossOutput<A>>+Wrappable<B=A>,X> TrainStep<X,RegressionOutput<A>> for Wrapped<Regression<W>> where W::Decomposition:AutodiffModule<A>,W::With<A::InnerBackend>:Decompose<Decomposition=<W::Decomposition as AutodiffModule<A>>::InnerModule>{
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
		Self{inner:A::compose(decomposition),layer:Default::default()}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Decompose> Decompose for Regression<A>{
	fn compose(decomposition:Self::Decomposition)->Self{
		Self{inner:A::compose(decomposition),layer:Default::default()}
	}
	fn decompose(self)->Self::Decomposition{self.inner.decompose()}
	fn decompose_cloned(&self)->Self::Decomposition{self.inner.decompose_cloned()}
	type Decomposition=A::Decomposition;
}
impl<A:Op<Output=Y>+Wrappable,Y> Op for Classification<A> where ClassificationLayer:AI<Y,ClassificationOutput<A::B>>{
	type Output=ClassificationOutput<A::B>;
}
impl<A:Op<Output=Y>+Wrappable,Y> Op for Regression<A> where RegressionLayer:AI<Y,RegressionOutput<A::B>>{
	type Output=RegressionOutput<A::B>;
}
impl<A:UnwrapInner> UnwrapInner for Classification<A>{
	fn unwrap_inner(self)->Self::Inner{self.into_inner().unwrap_inner()}
	type Inner=A::Inner;
}
impl<A:UnwrapInner> UnwrapInner for Regression<A>{
	fn unwrap_inner(self)->Self::Inner{self.into_inner().unwrap_inner()}
	type Inner=A::Inner;
}
impl<A:Wrappable<B=B>,B:Backend,D:Wrappable<B=B>> Wrappable for (A,D){
	type B=B;
	type With<C:Backend>=(A::With<C>,D::With<C>);
}
impl<A:Wrappable<B=B>,B:Backend,D:Wrappable<B=B>,E:Wrappable<B=B>> Wrappable for (A,D,E){
	type B=B;
	type With<C:Backend>=(A::With<C>,D::With<C>,E::With<C>);
}
impl<A:Wrappable<B=B>,B:Backend,D:Wrappable<B=B>,E:Wrappable<B=B>,F:Wrappable<B=B>> Wrappable for (A,D,E,F){
	type B=B;
	type With<C:Backend>=(A::With<C>,D::With<C>,E::With<C>,F::With<C>);
}
impl<A:Wrappable<B=B>,B:Backend,D:Wrappable<B=B>,E:Wrappable<B=B>,F:Wrappable<B=B>,G:Wrappable<B=B>> Wrappable for (A,D,E,F,G){
	type B=B;
	type With<C:Backend>=(A::With<C>,D::With<C>,E::With<C>,F::With<C>,G::With<C>);
}
impl<A:Wrappable<B=B>,B:Backend,D:Wrappable<B=B>,E:Wrappable<B=B>,F:Wrappable<B=B>,G:Wrappable<B=B>,H:Wrappable<B=B>> Wrappable for (A,D,E,F,G,H){
	type B=B;
	type With<C:Backend>=(A::With<C>,D::With<C>,E::With<C>,F::With<C>,G::With<C>,H::With<C>);
}
impl<A:Wrappable<B=B>,B:Backend,D:Wrappable<B=B>,E:Wrappable<B=B>,F:Wrappable<B=B>,G:Wrappable<B=B>,H:Wrappable<B=B>,I:Wrappable<B=B>> Wrappable for (A,D,E,F,G,H,I){
	type B=B;
	type With<C:Backend>=(A::With<C>,D::With<C>,E::With<C>,F::With<C>,G::With<C>,H::With<C>,I::With<C>);
}
impl<A:Wrappable<B=B>,B:Backend,D:Wrappable<B=B>,E:Wrappable<B=B>,F:Wrappable<B=B>,G:Wrappable<B=B>,H:Wrappable<B=B>,I:Wrappable<B=B>,J:Wrappable<B=B>> Wrappable for (A,D,E,F,G,H,I,J){
	type B=B;
	type With<C:Backend>=(A::With<C>,D::With<C>,E::With<C>,F::With<C>,G::With<C>,H::With<C>,I::With<C>,J::With<C>);
}
impl<A:Wrappable<B=B>,B:Backend,X:Wrappable<B=B>,Y:Wrappable<B=B>> Wrappable for SetType<A,X,Y>{
	type B=B;
	type With<C:Backend>=SetType<A::With<C>,X::With<C>,Y::With<C>>;
}
impl<A> Classification<A>{
	/// creates from the inner value
	pub fn from_inner(inner:A)->Self where Classification<A>:Op{
		Self{inner,layer:Default::default()}
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
		Self{inner,layer:Default::default()}
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
impl<B:Backend,E:Into<(Value<B>,Value<B>)>> Batcher<B,E,(Value<B>,Value<B>)> for BatchStacker{
	fn batch(&self,items:Vec<E>,_device:&<B as Backend>::Device)->(Value<B>,Value<B>){
		let items=items.into_iter().map(Into::into);
		let (input,target):(Vec<Value<B>>,Vec<Value<B>>)=items.unzip();

		let (input,target)=(Value::Multi(input),Value::Multi(target));
		(input.stack(0),target.stack(0))
	}
}
impl<B:Backend,W:AI<X,LossOutput<B>>+Wrappable<B=B>,X> ValidStep<X,ClassificationOutput<B>> for Wrapped<Classification<W>> where W::Decomposition:Module<B>{
	fn step(&self,item:X)->ClassificationOutput<B>{self.forward(item)}
}
impl<B:Backend,W:AI<X,LossOutput<B>>+Wrappable<B=B>,X> ValidStep<X,RegressionOutput<B>> for Wrapped<Regression<W>> where W::Decomposition:Module<B>{
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
impl<B:Backend,X:Into<Y>,Y> AI<X,Y> for Identity<B>{
	fn forward(&self,input:X)->Y{input.into()}
}
impl<B:Backend> AI<LossOutput<B>,ClassificationOutput<B>> for ClassificationLayer{
	fn forward(&self,lossoutput:LossOutput<B>)->ClassificationOutput<B>{//TODO make work for multi
		let loss=match lossoutput.loss(){Value::F1(x)=>x,Value::F2(x)=>x.mean(),Value::F3(x)=>x.mean(),Value::F4(x)=>x.mean(),Value::F5(x)=>x.mean(),Value::F6(x)=>x.mean(),Value::F7(x)=>x.mean(),Value::F8(x)=>x.mean(),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to classification output")};
		let output=match lossoutput.output(){Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(0,1),Value::F4(x)=>x.flatten(0,2),Value::F5(x)=>x.flatten(0,3),Value::F6(x)=>x.flatten(0,4),Value::F7(x)=>x.flatten(0,5),Value::F8(x)=>x.flatten(0,6),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to classification output")};
		let target=match lossoutput.target(){Value::I1(x)=>x,Value::I2(x)=>x.flatten(0,1),Value::I3(x)=>x.flatten(0,2),Value::I4(x)=>x.flatten(0,3),Value::I5(x)=>x.flatten(0,4),Value::I6(x)=>x.flatten(0,5),Value::I7(x)=>x.flatten(0,6),Value::I8(x)=>x.flatten(0,7),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to classification output")};
		ClassificationOutput::new(loss,output,target)
	}
}
impl<B:Backend> AI<LossOutput<B>,RegressionOutput<B>> for RegressionLayer{
	fn forward(&self,lossoutput:LossOutput<B>)->RegressionOutput<B>{
		let loss=match lossoutput.loss(){Value::F1(x)=>x,Value::F2(x)=>x.mean(),Value::F3(x)=>x.mean(),Value::F4(x)=>x.mean(),Value::F5(x)=>x.mean(),Value::F6(x)=>x.mean(),Value::F7(x)=>x.mean(),Value::F8(x)=>x.mean(),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let output=match lossoutput.output(){Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(0,1),Value::F4(x)=>x.flatten(0,2),Value::F5(x)=>x.flatten(0,3),Value::F6(x)=>x.flatten(0,4),Value::F7(x)=>x.flatten(0,5),Value::F8(x)=>x.flatten(0,6),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		let target=match lossoutput.target(){Value::F1(x)=>x.unsqueeze(),Value::F2(x)=>x,Value::F3(x)=>x.flatten(0,1),Value::F4(x)=>x.flatten(0,2),Value::F5(x)=>x.flatten(0,3),Value::F6(x)=>x.flatten(0,4),Value::F7(x)=>x.flatten(0,5),Value::F8(x)=>x.flatten(0,6),Value::Incompatible(e)=>panic!("{e}"),_=>panic!("cannot convert non floats to regression output")};
		RegressionOutput::new(loss,output,target)
	}
}
impl<B:Backend> Decompose for Identity<B>{
	fn compose(_decomposition:Self::Decomposition)->Self{new()}
	fn decompose(self){}
	fn decompose_cloned(&self){}
	type Decomposition=();
}
impl<B:Backend> Op for Identity<B>{
	type Output=();
}
impl<B:Backend> Wrappable for Identity<B>{
	type B=B;
	type With<C:Backend>=Identity<C>;
}
impl<B:Backend> Wrappable for Layer<B>{
	type B=B;
	type With<C:Backend>=Layer<C>;
}
impl<B:Backend> Wrappable for LossOutput<B>{
	type B=B;
	type With<C:Backend>=LossOutput<C>;
}
impl<B:Backend> Wrappable for Value<B>{
	type B=B;
	type With<C:Backend>=Value<C>;
}
impl<C:Backend,W:ToBackend<C,OnBackend=W::With<C>>+Wrappable> ToBackend<C> for Wrapped<W>{
	fn to_backend_device(self,device:&C::Device)->Self::OnBackend{
		Wrapped{inner:self.inner.to_backend_device(device)}
	}
	type OnBackend=Wrapped<W::With<C>>;
}
impl<T:?Sized+Op> Shortcuts for T{}
impl<W:AI<X,Y>+Wrappable,X,Y> AI<X,Y> for Wrapped<W>{
	fn forward(&self,input:X)->Y{self.inner.forward(input)}
	fn forward_mut(&mut self,input:X)->Y{self.inner.forward_mut(input)}
}
impl<W:Op+Wrappable> Op for Wrapped<W>{
	type Output=W::Output;
}
impl<W:UnwrapInner+Wrappable> UnwrapInner for Wrapped<W>{
	fn unwrap_inner(self)->Self::Inner{self.into_inner().unwrap_inner()}
	type Inner=W::Inner;
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
impl<W:Wrappable> Wrappable for Abs<W>{
	type B=W::B;
	type With<C:Backend>=Abs<W::With<C>>;
}
impl<W:Wrappable> Wrappable for AccQ<W>{
	type B=W::B;
	type With<C:Backend>=AccQ<W::With<C>>;
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
impl<W:Wrappable> Wrappable for Inner<W>{
	type B=W::B;
	type With<C:Backend>=Inner<W::With<C>>;
}
impl<W:Wrappable> Wrappable for Mean<W>{
	type B=W::B;
	type With<C:Backend>=Mean<W::With<C>>;
}
impl<W:Wrappable> Wrappable for SquaredError<W>{
	type B=W::B;
	type With<C:Backend>=SquaredError<W::With<C>>;
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
impl<W:Wrappable> Wrappable for Choose<W>{
	type B=W::B;
	type With<C:Backend>=Choose<W::With<C>>;
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
		graph.connect(true,"input",Layer::linear(true,2,10,1.0),"x");
		graph.connect(true,"x",Layer::relu(),"y");
		graph.connect(true,"y",Layer::linear(false,10,1,1.0),"output");

		let graph=Unvec(graph).wrap_inner().squared_error().set_type::<(Value<A>,Value<A>),LossOutput<A>>().regression().wrap();
		let graph=graph.train(&TrainConfig::new().with_checkpoints(false),SgdConfig::new().init(),0.01,train,valid);
		let graph=graph.valid().unwrap_inner();

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
	use super::*;
}
mod layer;
mod value;
/// starts the building of an ai structure in chained method style from an identity operation
pub fn new<B:Backend>()->Identity<B>{
	Identity{phantom:PhantomData}
}
#[derive(Clone,Copy,Debug,Default)]
/// batcher that stacks things
pub struct BatchStacker;
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for converting loss to classification output
pub struct Classification<A>{inner:A,layer:ClassificationLayer}
#[derive(Clone,Copy,Debug,Default)]
/// layer for converting loss to classification output
pub struct ClassificationLayer{seal:PhantomData<()>}
#[derive(Clone,Copy,Debug,Default)]
/// metrics renderer implementation that doesn't actually do anything
pub struct DontRender;
#[derive(Clone,Copy,Debug,Default)]
/// identity version that knows what backend
pub struct Identity<B:Backend>{phantom:PhantomData<B>}
#[derive(Clone,Copy,Debug,Default)]
/// wrapper for converting loss to regression output
pub struct Regression<A>{inner:A,layer:RegressionLayer}
#[derive(Clone,Copy,Debug,Default)]
/// layer for converting loss to regression output
pub struct RegressionLayer{seal:PhantomData<()>}
#[derive(Config,Debug)]
/// configuration for convenient training through the wrapper
pub struct TrainConfig{
	#[config(default="String::from(\".artifact\")")]
	artifact_directory:String,
	#[config(default="16")]
	batch_size:usize,
	#[config(default="true")]
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
#[derive(Clone,Copy,Debug,Default)]
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
/// trait for switching the backend of a module
pub trait ToBackend<B:Backend>:Sized{
	/// moves the module to the backend with the device
	fn to_backend_device(self,device:&B::Device)->Self::OnBackend;
	/// moves the module to the backend with the device
	fn to_backend(self)->Self::OnBackend{self.to_backend_device(&Default::default())}
	/// the type on the new backend
	type OnBackend;
}
/// higher kinded type trait to allow rewrapping burn modules in different backends to implement some wrapper features
pub trait Wrappable:Clone+Debug+Decompose+Send{
	type B:Backend;
	type With<C:Backend>:Wrappable<B=C,With<C>=Self::With<C>>+Wrappable<B=C,With<Self::B>=Self>;
}
pub use burn as lib;
pub use layer::{Attention,AttentionConfig,AttentionMask,BiasConfig,CacheKV,Config,Layer,KQV,KQVConfig};
pub use value::{Kind,LossOutput,Shape,Value};
use burn::{
	backend::NdArray,
	data::{
		dataset::Dataset,dataloader::{batcher::Batcher,DataLoaderBuilder}
	},
	lr_scheduler::LrScheduler,
	module::{AutodiffModule,Content,DisplaySettings,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor,Quantizer},
	optim::Optimizer,
	prelude::*,
	record::{CompactRecorder,FileRecorder,RecorderError},
	tensor::backend::AutodiffBackend,
	train::{
		ClassificationOutput,LearnerBuilder,RegressionOutput,TrainOutput,TrainStep,ValidStep,metric::{Adaptor,ItemLazy,LossInput,LossMetric},renderer::{MetricState,MetricsRenderer,TrainingProgress}
	}
};
use crate::{
	AI,Decompose,Graph,Inner,Op,UnwrapInner,Unvec,builtin::{Abs,AccQ,Cat,Choose,CrossEntropy,Duplicate,Map,Mean,Sequential,SetType,SquaredError,Zip}
};
use rand::random;
use std::{
	fmt::{Debug,Display},fs::{create_dir_all as create_folder},marker::PhantomData,path::PathBuf
};
