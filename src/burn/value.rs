bicop_num!(Add,add,add_scalar);
bicop_num!(Div,div,div_scalar);
bicop_num!(Mul,mul,mul_scalar);
bicop_num!(Rem,rem,remainder_scalar);
bicop_num!(Sub,sub,sub_scalar);
fn broadcast_multi<B:Backend,F:FnMut(Value<B>,Value<B>)->Value<B>>(u:Vec<Value<B>>,v:Vec<Value<B>>,mut f:F)->Value<B>{
	if u.len()==1{
		u.into_iter().cycle().zip(v).map(|(x,y)|f(x,y)).collect()
	}else if v.len()==1{
		u.into_iter().zip(v.into_iter().cycle()).map(|(x,y)|f(x,y)).collect()
	}else if u.len()==v.len(){
		u.into_iter().zip(v).map(|(x,y)|f(x,y)).collect()
	}else{
		"mismatched lengths".into()
	}
}
fn hard_choose_burn_1<B:Backend,const N:usize>(dim:i32,distribution:Tensor<B,N>)->u32{
	let dim=if dim<0{N-(-dim) as usize}else{dim as usize};
	let distribution=if dim==N-1{distribution}else{distribution.movedim(dim,N-1)}.into_data();
	let sum=distribution.iter().fold(0.0,|acc:f32,weight:f32|acc+weight);

	distribution.iter().scan(random::<f32>()*sum,|choice:&mut f32,weight:f32|Some(*choice-=weight).filter(|_|*choice>=0.0)).count() as u32
}
fn hard_choose_burn_multi<B:Backend,const N:usize>(dim:i32,distribution:Tensor<B,N>)->Vec<u32>{
	let dim=if dim<0{N-(-dim) as usize}else{dim as usize};

	let chunk=distribution.dims()[dim];
	let distribution=if dim==N-1{distribution}else{distribution.movedim(dim,N-1)}.into_data().to_vec().unwrap();

	distribution.chunks_exact(chunk).map(|d|{
		let sum=d.iter().fold(0.0,|acc:f32,weight:&f32|acc+weight);
		d.iter().scan(random::<f32>()*sum,|choice:&mut f32,weight:&f32|Some(*choice-=weight).filter(|_|*choice>=0.0)).count() as u32
	}).collect()
}
fn hard_choose_burn_tensor<B:Backend,const N:usize>(dim:i32,distribution:Tensor<B,N>)->Tensor<B,N,Int>{//TODO test this
	let dim=if dim<0{N-(-dim) as usize}else{dim as usize};
	let device=distribution.device();
	let mut dims=distribution.dims();

	dims[N-1]=1;
	let r:Tensor<B,N,Int>=Tensor::from_data(TensorData::new(hard_choose_burn_multi(dim as i32,distribution),dims),&device);

	r.movedim(N-1,dim)
}
fn slice_slice<B:Backend,K:BasicOps<B>+TensorKind<B>,const N:usize>(ranges:&[Range<usize>],tensor:Tensor<B,N,K>)->Tensor<B,N,K>{
	let mut n=0;
	let mut acc=||{
		let a=n;
		n+=1;
		a
	};

	match ranges.len(){0=>tensor,1=>tensor.slice([0;1].map(|_|ranges[acc()].clone())),2=>tensor.slice([0;2].map(|_|ranges[acc()].clone())),3=>tensor.slice([0;3].map(|_|ranges[acc()].clone())),4=>tensor.slice([0;4].map(|_|ranges[acc()].clone())),5=>tensor.slice([0;5].map(|_|ranges[acc()].clone())),6=>tensor.slice([0;6].map(|_|ranges[acc()].clone())),7=>tensor.slice([0;7].map(|_|ranges[acc()].clone())),8=>tensor.slice([0;8].map(|_|ranges[acc()].clone())),_=>panic!("too many ranges for current max 8 dims")}
}
fn soft_choose_burn_1<B:Backend,const N:usize>(dim:i32,logits:Tensor<B,N>,temperature:f32)->u32{
	let dim=if dim<0{N-(-dim) as usize}else{dim as usize};
	let logits=if dim==N-1{logits}else{logits.movedim(dim,N-1)};
	let distribution=softmax(logits/temperature,N-1).into_data();
	distribution.iter().scan(random(),|choice:&mut f32,weight:f32|Some(*choice-=weight).filter(|_|*choice>=0.0)).count() as u32
}
fn soft_choose_burn_multi<B:Backend,const N:usize>(dim:i32,logits:Tensor<B,N>,temperature:f32)->Vec<u32>{
	let dim=if dim<0{N-(-dim) as usize}else{dim as usize};
	let logits=if dim==N-1{logits}else{logits.movedim(dim,N-1)};
	let chunk=logits.dims()[N-1];
	let distribution=softmax(logits/temperature,N-1).into_data().to_vec().unwrap();
	distribution.chunks_exact(chunk).map(|d|d.iter().scan(random(),|choice:&mut f32,weight:&f32|Some(*choice-=weight).filter(|_|*choice>=0.0)).count() as u32).collect()
}
fn soft_choose_burn_tensor<B:Backend,const N:usize>(dim:i32,logits:Tensor<B,N>,temperature:f32)->Tensor<B,N,Int>{//TODO test this
	let dim=if dim<0{N-(-dim) as usize}else{dim as usize};
	let device=logits.device();
	let mut dims=logits.dims();

	dims[N-1]=1;
	let r:Tensor<B,N,Int>=Tensor::from_data(TensorData::new(soft_choose_burn_multi(dim as i32,logits,temperature),dims),&device);
	r.movedim(N-1,dim)
}
impl AsRef<Self> for Shape{//TODO more reref stuff
	fn as_ref(&self)->&Self{self}
}
impl Shape{
	/// counts the number of components if possible. returns none if incompatible or if a non recursive multi shape of more than 0 tensors
	pub fn count(&self)->Option<usize>{
		match self{
			Shape::Incompatible(_e)=>None,
			Shape::Multi(n)=>if *n==0{Some(0)}else{None},
			Shape::Recursive(v)=>{
				let mut s=0;
				for v in v{s+=v.count()?}
				Some(s)
			},
			X1(x)=>Some(x.iter().product()),
			X2(x)=>Some(x.iter().product()),
			X3(x)=>Some(x.iter().product()),
			X4(x)=>Some(x.iter().product()),
			X5(x)=>Some(x.iter().product()),
			X6(x)=>Some(x.iter().product()),
			X7(x)=>Some(x.iter().product()),
			X8(x)=>Some(x.iter().product())
		}
	}
	/// converts to the eight dimensional array type by extending with ones. The original data will be placed according to the alignment. Multi and incompatible types will be all ones
	pub fn to_array(self,alignment:Alignment)->[usize;8]{
		let mut result=[1;8];
		let slice=match &self{Shape::Incompatible(_e)=>return result,Shape::Multi(_v)=>return result,Shape::Recursive(_r)=>return result,X1(x)=>x.as_slice(),X2(x)=>x.as_slice(),X3(x)=>x.as_slice(),X4(x)=>x.as_slice(),X5(x)=>x.as_slice(),X6(x)=>x.as_slice(),X7(x)=>x.as_slice(),X8(x)=>x.as_slice()};
		let l=slice.len();
		match alignment{Alignment::Center=>result[4-l/2..][..l].copy_from_slice(slice),Alignment::Left=>result[..l].copy_from_slice(slice),Alignment::Right=>result[8-l..].copy_from_slice(slice)}
		result
	}
}
impl<'a,B:Backend> Deserialize<'a> for Value<B>{
	fn deserialize<D:Deserializer<'a>>(deserializer:D)->Result<Self,D::Error>{ValueData::deserialize(deserializer).map(Into::into)}
}
impl<A:AutodiffBackend> AutodiffModule<A> for Value<A>{
	fn valid(&self)->Self::InnerModule{
		match self{B1(x)=>B1(x.valid()),B2(x)=>B2(x.valid()),B3(x)=>B3(x.valid()),B4(x)=>B4(x.valid()),B5(x)=>B5(x.valid()),B6(x)=>B6(x.valid()),B7(x)=>B7(x.valid()),B8(x)=>B8(x.valid()),F1(x)=>F1(x.valid()),F2(x)=>F2(x.valid()),F3(x)=>F3(x.valid()),F4(x)=>F4(x.valid()),F5(x)=>F5(x.valid()),F6(x)=>F6(x.valid()),F7(x)=>F7(x.valid()),F8(x)=>F8(x.valid()),I1(x)=>I1(x.valid()),I2(x)=>I2(x.valid()),I3(x)=>I3(x.valid()),I4(x)=>I4(x.valid()),I5(x)=>I5(x.valid()),I6(x)=>I6(x.valid()),I7(x)=>I7(x.valid()),I8(x)=>I8(x.valid()),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.iter().map(|x|x.valid()).collect()}
	}
	type InnerModule=Value<A::InnerBackend>;
}
impl<A:Into<Value<B>>,B:Backend> FromIterator<A> for Value<B>{
	fn from_iter<I:IntoIterator<Item=A>>(iter:I)->Self{Value::Multi(iter.into_iter().map(Into::into).collect())}
}
impl<B:Backend,K:'static+TensorKind<B>,const N:usize> From<Result<Tensor<B,N,K>,String>> for Value<B>{
	fn from(value:Result<Tensor<B,N,K>,String>)->Self{
		match value{Err(e)=>e.into(),Ok(t)=>t.into()}
	}
}
impl<B:Backend,K:'static+TensorKind<B>,const N:usize> From<Tensor<B,N,K>> for Value<B>{
	fn from(value:Tensor<B,N,K>)->Self{
		let kind=TypeId::of::<K>();
		let kind=if kind==TypeId::of::<Bool>(){Kind::Bool}else if kind==TypeId::of::<Float>(){Kind::Float}else if kind==TypeId::of::<Int>(){Kind::Int}else{return "only bool, float, and int tensors with dimensions 1-8 are currently supported".into()};

		let v=unsafe{
			match (N,kind){(1,Kind::Bool)=>B1(mem::transmute_copy(&value)),(2,Kind::Bool)=>B2(mem::transmute_copy(&value)),(3,Kind::Bool)=>B3(mem::transmute_copy(&value)),(4,Kind::Bool)=>B4(mem::transmute_copy(&value)),(5,Kind::Bool)=>B5(mem::transmute_copy(&value)),(6,Kind::Bool)=>B6(mem::transmute_copy(&value)),(7,Kind::Bool)=>B7(mem::transmute_copy(&value)),(8,Kind::Bool)=>B8(mem::transmute_copy(&value)),(1,Kind::Float)=>F1(mem::transmute_copy(&value)),(2,Kind::Float)=>F2(mem::transmute_copy(&value)),(3,Kind::Float)=>F3(mem::transmute_copy(&value)),(4,Kind::Float)=>F4(mem::transmute_copy(&value)),(5,Kind::Float)=>F5(mem::transmute_copy(&value)),(6,Kind::Float)=>F6(mem::transmute_copy(&value)),(7,Kind::Float)=>F7(mem::transmute_copy(&value)),(8,Kind::Float)=>F8(mem::transmute_copy(&value)),(1,Kind::Int)=>I1(mem::transmute_copy(&value)),(2,Kind::Int)=>I2(mem::transmute_copy(&value)),(3,Kind::Int)=>I3(mem::transmute_copy(&value)),(4,Kind::Int)=>I4(mem::transmute_copy(&value)),(5,Kind::Int)=>I5(mem::transmute_copy(&value)),(6,Kind::Int)=>I6(mem::transmute_copy(&value)),(7,Kind::Int)=>I7(mem::transmute_copy(&value)),(8,Kind::Int)=>I8(mem::transmute_copy(&value)),_=>return "only bool, float, and int tensors with dimensions 1-8 are currently supported".into()}
		};
		mem::forget(value);
		v
	}
}
impl<B:Backend,K:'static+TensorKind<B>,const N:usize> TryFrom<Value<B>> for Tensor<B,N,K>{
	fn try_from(value:Value<B>)->Result<Self,Self::Error>{
		let kind=TypeId::of::<K>();
		let kind=if kind==TypeId::of::<Bool>(){Kind::Bool}else if kind==TypeId::of::<Float>(){Kind::Float}else if kind==TypeId::of::<Int>(){Kind::Int}else{return Err(value)};

		if Some(N)!=value.rank()||kind!=value.kind(){return Err(value)}
		let r=unsafe{
			match &value{B1(x)=>mem::transmute_copy(x),B2(x)=>mem::transmute_copy(x),B3(x)=>mem::transmute_copy(x),B4(x)=>mem::transmute_copy(x),B5(x)=>mem::transmute_copy(x),B6(x)=>mem::transmute_copy(x),B7(x)=>mem::transmute_copy(x),B8(x)=>mem::transmute_copy(x),F1(x)=>mem::transmute_copy(x),F2(x)=>mem::transmute_copy(x),F3(x)=>mem::transmute_copy(x),F4(x)=>mem::transmute_copy(x),F5(x)=>mem::transmute_copy(x),F6(x)=>mem::transmute_copy(x),F7(x)=>mem::transmute_copy(x),F8(x)=>mem::transmute_copy(x),I1(x)=>mem::transmute_copy(x),I2(x)=>mem::transmute_copy(x),I3(x)=>mem::transmute_copy(x),I4(x)=>mem::transmute_copy(x),I5(x)=>mem::transmute_copy(x),I6(x)=>mem::transmute_copy(x),I7(x)=>mem::transmute_copy(x),I8(x)=>mem::transmute_copy(x),_=>panic!("internal error")}
		};
		mem::forget(value);
		Ok(r)
	}
	type Error=Value<B>;
}
impl<B:Backend,S:?Sized+AsRef<str>> From<&S> for Value<B>{
	fn from(value:&S)->Self{Self::Incompatible(value.as_ref().to_string())}
}
impl<B:Backend,const D:usize> AI<Value<B>,Value<B>> for BatchNorm<B,D>{
	fn forward(&self,input:Value<B>)->Value<B>{
		fn f<B:Backend,const D:usize,const E:usize,const F:usize>(norm:&BatchNorm<B,D>,x:Tensor<B,E>)->Value<B>{
			let norm:BatchNorm<B,F>=BatchNorm{beta:norm.beta.clone(),epsilon:norm.epsilon.clone(),gamma:norm.gamma.clone(),momentum:norm.momentum.clone(),running_mean:norm.running_mean.clone(),running_var:norm.running_var.clone()};
			norm.forward(x).into()
		}
		match input.float(){
			F1(x)=>AI::forward(self,F1(x).unsqueeze().unsqueeze()).squeeze().squeeze(),
			F2(x)=>AI::forward(self,F2(x).unsqueeze()).squeeze(),
			F3(x)=>f::<B,D,3,1>(self,x),
			F4(x)=>f::<B,D,4,2>(self,x),
			F5(x)=>f::<B,D,5,3>(self,x),
			F6(x)=>f::<B,D,6,4>(self,x),
			F7(x)=>f::<B,D,7,5>(self,x),
			F8(x)=>f::<B,D,8,6>(self,x),
			Value::Incompatible(e)=>e.into(),
			Value::Multi(v)=>v.into_iter().map(|x|AI::forward(self,x)).collect(),
			_=>panic!("internal error")
		}
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),Vec<f32>> for CrossEntropyLayer{
	fn forward(&self,input:(Value<B>,Value<B>))->Vec<f32>{
		let output:Value<B>=self.forward(input);
		output.into_float_vec()
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),LossOutput<B>> for CrossEntropyLayer{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->LossOutput<B>{
		let loss=self.forward((output.clone(),target.clone()));
		LossOutput::new(loss,output,target)
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),Value<B>> for CrossEntropyLayer{// TODO make smoothing and such work on burn specific one
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Value<B>{
		fn ff<B:Backend,const N:usize>(dim:i32,y:Tensor<B,N>,t:Tensor<B,N>,temperature:f32)->Result<Tensor<B,N>,String>{
			let dim=if dim<0{N-(-dim) as usize}else{dim as usize};
			let (ydims,tdims)=(y.dims(),t.dims());
			if ydims==tdims{
				let logy=if temperature.is_nan(){y.log()}else{log_softmax(y/temperature,dim)};
				Ok(logy*t.neg())
			}else{
				Err(format!("incompatible shapes to cross entropy. ydims: {ydims:?} tdims: {tdims:?}"))
			}
		}
		fn fi<B:Backend,const N:usize,const K:usize>(dim:i32,y:Tensor<B,N>,t:Tensor<B,K,Int>,temperature:f32)->Result<Tensor<B,K>,String>{
			let dim=if dim<0{N-(-dim) as usize}else{dim as usize};
			let (ydims,tdims)=(y.dims(),t.dims());
			if ydims.iter().enumerate().filter_map(|(n,y)|(n!=dim).then_some(y)).eq(tdims.iter()){
				let logy=if temperature.is_nan(){y.log()}else{log_softmax(y/temperature,dim)};
				Ok(logy.gather(dim,t.unsqueeze_dim(dim)).neg().squeeze(dim))
			}else{
				Err(format!("incompatible shapes to cross entropy along dimension {dim}. ydims: {ydims:?} tdims: {tdims:?}"))
			}
		}
		let (dim,temp)=(self.get_dim(),self.get_temperature());

		match match (output,target){
			(F1(y),F1(t))=>ff(dim,y,t,temp).map(Into::into),
			(F2(y),F2(t))=>ff(dim,y,t,temp).map(Into::into),
			(F3(y),F3(t))=>ff(dim,y,t,temp).map(Into::into),
			(F4(y),F4(t))=>ff(dim,y,t,temp).map(Into::into),
			(F5(y),F5(t))=>ff(dim,y,t,temp).map(Into::into),
			(F6(y),F6(t))=>ff(dim,y,t,temp).map(Into::into),
			(F7(y),F7(t))=>ff(dim,y,t,temp).map(Into::into),
			(F8(y),F8(t))=>ff(dim,y,t,temp).map(Into::into),
			(F1(y),I1(t))=>fi(dim,y.unsqueeze::<2>(),t,temp).map(Into::into),
			(F2(y),I1(t))=>fi(dim,y,t,temp).map(Into::into),
			(F3(y),I2(t))=>fi(dim,y,t,temp).map(Into::into),
			(F4(y),I3(t))=>fi(dim,y,t,temp).map(Into::into),
			(F5(y),I4(t))=>fi(dim,y,t,temp).map(Into::into),
			(F6(y),I5(t))=>fi(dim,y,t,temp).map(Into::into),
			(F7(y),I6(t))=>fi(dim,y,t,temp).map(Into::into),
			(F7(y),I7(t))=>fi(dim,y,t,temp).map(Into::into),
			(Value::Incompatible(y),_)=>Err(y),
			(_,Value::Incompatible(t))=>Err(t),// TODO broadcast multi
			(Value::Multi(y),Value::Multi(t))=>if y.len()==t.len(){Ok(Value::Multi(y.into_iter().zip(t).map(|x|self.forward_typed::<_,Value<B>>(x)).collect()))}else{Err("mismatched lengths".into())},
			_=>Err("incompatible".into())
		}{
			Err(e)=>Value::Incompatible(e),Ok(x)=>x
		}
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),Value<B>> for CrossEntropyLoss<B>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Value<B>{
		let mut op=().fix_type::<Value<B>>().cross_entropy(1.0);
		if !self.logits{op.set_temperature(f32::NAN)}
		op.forward((output,target))
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),LossOutput<B>> for SquaredErrorLayer{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->LossOutput<B>{
		let loss=self.forward((output.clone(),target.clone()));
		LossOutput::new(loss,output,target)
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),Value<B>> for SquaredErrorLayer{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Value<B>{
		fn f<B:Backend,const N:usize>(y:Tensor<B,N>,t:Tensor<B,N>)->Value<B>{
			if y.dims()==t.dims(){MseLoss.forward_no_reduction(y,t).into()}else{"compatible inputs for squared error are float tensors of the same shape".into()}
		}
		match (output.float(),target.float()){(F1(y),F1(t))=>f(y,t),(F2(y),F2(t))=>f(y,t),(F3(y),F3(t))=>f(y,t),(F4(y),F4(t))=>f(y,t),(F5(y),F5(t))=>f(y,t),(F6(y),F6(t))=>f(y,t),(F7(y),F7(t))=>f(y,t),(F8(y),F8(t))=>f(y,t),(Value::Incompatible(y),_)=>y.into(),(_,Value::Incompatible(t))=>t.into(),(Value::Multi(y),t)=>broadcast_multi(y,t.into_multi(),|y,t|self.forward((y,t))),(y,Value::Multi(t))=>broadcast_multi(y.into_multi(),t,|y,t|self.forward((y,t))),_=>"compatible inputs for squared error are float tensors of the same shape".into()}
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),Vec<f32>> for SquaredErrorLayer{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Vec<f32>{
		let error:Value<B>=self.forward((output,target));
		error.into_float_vec()
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),f32> for SquaredErrorLayer{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->f32{().fix_type::<Value<B>>().squared_error().mean().forward((output,target))}
}
impl<B:Backend> AI<Value<B>,Tensor<B,1>> for MeanLayer{
	fn forward(&self,input:Value<B>)->Tensor<B,1>{
		fn avg<B:Backend,const N:usize>(x:Tensor<B,N>)->Tensor<B,1>{x.mean()}
		let l=input.len();

		if l==0{return Tensor::from_data(TensorData::new(vec![f32::NAN],[1]),&Default::default())}
		match input.float(){F1(x)=>avg(x),F2(x)=>avg(x),F3(x)=>avg(x),F4(x)=>avg(x),F5(x)=>avg(x),F6(x)=>avg(x),F7(x)=>avg(x),F8(x)=>avg(x),Value::Incompatible(e)=>panic!("Could not reduce to a scalar due to incompatibility: {e}"),Value::Multi(v)=>v.into_iter().map(|x|self.forward(x)).reduce(|x:Tensor<B,1>,y:Tensor<B,1>|x+y).unwrap()/l as f32,_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Tensor<B,1>> for SumLayer{
	fn forward(&self,input:Value<B>)->Tensor<B,1>{
		fn sum<B:Backend,const N:usize>(x:Tensor<B,N>)->Tensor<B,1>{x.sum()}
		let l=input.len();

		if l==0{return Tensor::from_data(TensorData::new(vec![f32::NAN],[1]),&Default::default())}
		match input.float(){F1(x)=>sum(x),F2(x)=>sum(x),F3(x)=>sum(x),F4(x)=>sum(x),F5(x)=>sum(x),F6(x)=>sum(x),F7(x)=>sum(x),F8(x)=>sum(x),Value::Incompatible(e)=>panic!("Could not reduce to a scalar due to incompatibility: {e}"),Value::Multi(v)=>v.into_iter().map(|x|self.forward(x)).reduce(|x:Tensor<B,1>,y:Tensor<B,1>|x+y).unwrap(),_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Conv2d<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		fn f<B:Backend,const N:usize>(input:Tensor<B,N>,layer:&Conv2d<B>)->Value<B>{// TODO dimension check
			let mut dims=input.dims();
			let n:usize=dims.iter().product();

			let c=if N<3{1}else{dims[N-3]};
			let h=if N<2{1}else{dims[N-2]};
			let w=dims[N-1];

			let b=n/(c*h*w);
			let output=layer.forward(input.reshape([b,c,h,w]));

			let [_b,c,h,w]=output.dims();

			dims[N-1]=w;
			if N<3&&c!=1{return F3(output.reshape([c,h,w]))}else if N>=3{dims[N-3]=c}
			if N<2&&h!=1{return F2(output.reshape([h,w]))}else if N>=2{dims[N-2]=h}
			output.reshape(dims).into()
		}
		let l=self;

		match input.float(){F1(x)=>f(x,l),F2(x)=>f(x,l),F3(x)=>f(x,l),F4(x)=>f(x,l),F5(x)=>f(x,l),F6(x)=>f(x,l),F7(x)=>f(x,l),F8(x)=>f(x,l),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|AI::forward(self,x)).collect(),_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for CrossEntropyLayer{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{
			Value::Incompatible(e)=>e.into(),
			Value::Multi(v)=>if v.len()==2{
				let [output,target]=v.try_into().unwrap();
				self.forward((output,target))
			}else{
				v.into_iter().map(|x|self.forward(x)).collect()
			},
			_=>"cross entropy inputs must be in pairs".into()
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for CrossEntropyLoss<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		let mut op=CrossEntropyLayer::new(1.0);
		if !self.logits{op.set_temperature(f32::NAN)}
		op.forward(input)
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for MeanLayer{
	fn forward(&self,input:Value<B>)->Value<B>{
		fn avg<B:Backend,const N:usize,const K:usize>(d:i32,x:Tensor<B,N>)->Tensor<B,K>{
			let d=if d<0{N-((-d) as usize)}else{d as usize};
			x.mean_dim(d).squeeze(d)
		}
		let l=input.len();

		if l==0{return input}
		match self.get_reduction_mode(){
			ReductionMode::Component=>F1(self.forward(input)),
			ReductionMode::Dim(d)=>{
				if let Some(r)=input.rank(){
					if d>=r as i32||d<(-(r as i32)){return format!("rank {r} is too low to cat along dimension {d}").into()}
				}
				match input.float(){F1(x)=>F1(x.mean()),F2(x)=>F1(avg(d,x)),F3(x)=>F2(avg(d,x)),F4(x)=>F3(avg(d,x)),F5(x)=>F4(avg(d,x)),F6(x)=>F5(avg(d,x)),F7(x)=>F6(avg(d,x)),F8(x)=>F7(avg(d,x)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|self.forward(x)).reduce(|x:Value<B>,y:Value<B>|x+y).unwrap()/l as f32,_=>panic!("internal error")}
			},
			ReductionMode::Tensor=>match input.float(){Value::Multi(v)=>v.into_iter().reduce(|x,y|x+y).unwrap()/l as f32,x=>x}
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for MseLoss{
	fn forward(&self,input:Value<B>)->Value<B>{SquaredErrorLayer::new().forward(input)}
}
impl<B:Backend> AI<Value<B>,Value<B>> for SquaredErrorLayer{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input{
			Value::Incompatible(e)=>e.into(),
			Value::Multi(v)=>if v.len()==2{
				let [output,target]=v.try_into().unwrap();
				self.forward((output,target))
			}else{
				v.into_iter().map(|x|self.forward(x)).collect()
			},
			_=>"squared error inputs must be in pairs".into()
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for SumLayer{
	fn forward(&self,input:Value<B>)->Value<B>{
		fn sum<B:Backend,const N:usize,const K:usize>(d:i32,x:Tensor<B,N>)->Tensor<B,K>{
			let d=if d<0{N-((-d) as usize)}else{d as usize};
			x.mean_dim(d).squeeze(d)
		}
		let l=input.len();

		if l==0{return input}
		match self.get_reduction_mode(){
			ReductionMode::Component=>F1(self.forward(input)),
			ReductionMode::Dim(d)=>{
				if let Some(r)=input.rank(){
					if d>=r as i32||d<(-(r as i32)){return format!("rank {r} is too low to cat along dimension {d}").into()}
				}
				match input.float(){F1(x)=>F1(x.sum()),F2(x)=>F1(sum(d,x)),F3(x)=>F2(sum(d,x)),F4(x)=>F3(sum(d,x)),F5(x)=>F4(sum(d,x)),F6(x)=>F5(sum(d,x)),F7(x)=>F6(sum(d,x)),F8(x)=>F7(sum(d,x)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|self.forward(x)).reduce(|x:Value<B>,y:Value<B>|x+y).unwrap(),_=>panic!("internal error")}
			},
			ReductionMode::Tensor=>match input.float(){Value::Multi(v)=>v.into_iter().reduce(|x,y|x+y).unwrap(),x=>x}
		}
	}
}
impl<B:Backend> AI<Value<B>,f32> for MeanLayer{
	fn forward(&self,input:Value<B>)->f32{
		let y:Tensor<B,1>=self.forward(input);
		y.into_scalar().to_f32()
	}
}
impl<B:Backend> AI<Value<B>,f32> for SumLayer{
	fn forward(&self,input:Value<B>)->f32{
		let y:Tensor<B,1>=self.forward(input);
		y.into_scalar().to_f32()
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for AccQLayer{
	fn forward(&self,input:Value<B>)->Value<B>{
		fn acc_q<B:Backend,const N:usize>(dim:i32,gamma:f32,i:Tensor<B,N>)->Tensor<B,N>{
			let dim=if dim<0{N-(-dim) as usize}else{dim as usize};
			let mut q=i.split(1,dim);
			q.iter_mut().rev().fold(None,|future,present|{
				if let Some(f)=future{*present=f*gamma+present.clone()}
				Some(present.clone())
			});
			Tensor::cat(q,dim)
		}
		let (dim,gamma)=(self.get_dim(),self.get_gamma());

		match input.float(){F1(x)=>F1(acc_q(dim,gamma,x)),F2(x)=>F2(acc_q(dim,gamma,x)),F3(x)=>F3(acc_q(dim,gamma,x)),F4(x)=>F4(acc_q(dim,gamma,x)),F5(x)=>F5(acc_q(dim,gamma,x)),F6(x)=>F6(acc_q(dim,gamma,x)),F7(x)=>F7(acc_q(dim,gamma,x)),F8(x)=>F8(acc_q(dim,gamma,x)),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>Value::Multi(x.into_iter().map(|x|self.forward(x)).collect()),_=>panic!("unexpected non float value")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for CatLayer{
	fn forward(&self,input:Value<B>)->Value<B>{input.cat(self.get_dim())}
}
impl<B:Backend> AI<Value<B>,u32> for ChooseLayer{
	fn forward(&self,input:Value<B>)->u32{
		let (dim,temperature)=(self.get_dim(),self.get_temperature());

		match input.float(){
			F1(x)=>if temperature.is_nan(){hard_choose_burn_1(dim,x)}else{soft_choose_burn_1(dim,x,temperature)},
			F2(x)=>if temperature.is_nan(){hard_choose_burn_1(dim,x)}else{soft_choose_burn_1(dim,x,temperature)},
			F3(x)=>if temperature.is_nan(){hard_choose_burn_1(dim,x)}else{soft_choose_burn_1(dim,x,temperature)},
			F4(x)=>if temperature.is_nan(){hard_choose_burn_1(dim,x)}else{soft_choose_burn_1(dim,x,temperature)},
			F5(x)=>if temperature.is_nan(){hard_choose_burn_1(dim,x)}else{soft_choose_burn_1(dim,x,temperature)},
			F6(x)=>if temperature.is_nan(){hard_choose_burn_1(dim,x)}else{soft_choose_burn_1(dim,x,temperature)},
			F7(x)=>if temperature.is_nan(){hard_choose_burn_1(dim,x)}else{soft_choose_burn_1(dim,x,temperature)},
			F8(x)=>if temperature.is_nan(){hard_choose_burn_1(dim,x)}else{soft_choose_burn_1(dim,x,temperature)},
			Value::Incompatible(e)=>panic!("Could not create scalar due to incompatibility: {e}"),
			Value::Multi(v)=>if v.len()==1{self.forward(v.into_iter().next().unwrap())}else{panic!("Cannot soft choose one scalar from multiple values")},
			_=>panic!("internal error")
		}
	}
}
impl<B:Backend> AI<Value<B>,Vec<u32>> for ChooseLayer{
	fn forward(&self,input:Value<B>)->Vec<u32>{
		let (dim,temperature)=(self.get_dim(),self.get_temperature());

		match input.float(){
			F1(x)=>if temperature.is_nan(){hard_choose_burn_multi(dim,x)}else{soft_choose_burn_multi(dim,x,temperature)},
			F2(x)=>if temperature.is_nan(){hard_choose_burn_multi(dim,x)}else{soft_choose_burn_multi(dim,x,temperature)},
			F3(x)=>if temperature.is_nan(){hard_choose_burn_multi(dim,x)}else{soft_choose_burn_multi(dim,x,temperature)},
			F4(x)=>if temperature.is_nan(){hard_choose_burn_multi(dim,x)}else{soft_choose_burn_multi(dim,x,temperature)},
			F5(x)=>if temperature.is_nan(){hard_choose_burn_multi(dim,x)}else{soft_choose_burn_multi(dim,x,temperature)},
			F6(x)=>if temperature.is_nan(){hard_choose_burn_multi(dim,x)}else{soft_choose_burn_multi(dim,x,temperature)},
			F7(x)=>if temperature.is_nan(){hard_choose_burn_multi(dim,x)}else{soft_choose_burn_multi(dim,x,temperature)},
			F8(x)=>if temperature.is_nan(){hard_choose_burn_multi(dim,x)}else{soft_choose_burn_multi(dim,x,temperature)},
			Value::Incompatible(e)=>panic!("Could not create vector due to incompatibility: {e}"),
			Value::Multi(v)=>v.into_iter().flat_map(|x|self.forward_typed::<_,Vec<u32>>(x)).collect(),
			_=>panic!("internal error")
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Dropout{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input.float(){F1(x)=>F1(self.forward(x)),F2(x)=>F2(self.forward(x)),F3(x)=>F3(self.forward(x)),F4(x)=>F4(self.forward(x)),F5(x)=>F5(self.forward(x)),F6(x)=>F6(self.forward(x)),F7(x)=>F7(self.forward(x)),F8(x)=>F8(self.forward(x)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(|x|AI::forward(self,x)).collect()),_=>panic!("internal error")}
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
		match input{F1(x)=>apply_linear(self,x).into(),F2(x)=>apply_linear(self,x).into(),F3(x)=>apply_linear(self,x).into(),F4(x)=>apply_linear(self,x).into(),F5(x)=>apply_linear(self,x).into(),F6(x)=>apply_linear(self,x).into(),F7(x)=>apply_linear(self,x).into(),F8(x)=>apply_linear(self,x).into(),I1(x)=>apply_embed::<B,1,2>(self,x).into(),I2(x)=>apply_embed::<B,2,3>(self,x).into(),I3(x)=>apply_embed::<B,3,4>(self,x).into(),I4(x)=>apply_embed::<B,4,5>(self,x).into(),I5(x)=>apply_embed::<B,5,6>(self,x).into(),I6(x)=>apply_embed::<B,6,7>(self,x).into(),I7(x)=>apply_embed::<B,7,8>(self,x).into(),I8(_x)=>"embedding output would exceed maximum supported rank".into(),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>x.into_iter().map(|x|AI::forward(self,x)).collect::<Vec<_>>().into(),_=>"embedding is only available for float or int inputs".into()}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for LayerNorm<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		fn f<B:Backend,const N:usize>(input:Tensor<B,N>,layer:&LayerNorm<B>)->Value<B>{
			let b=layer.beta.dims();
			let g=layer.gamma.dims();
			let i=input.dims();

			if b!=g{return format!("malformed layer norm. beta dims: {b:?}. gamma dims: {g:?}.").into()}
			if b.last()!=i.last(){return format!("layer norm for dimension {b:?} is not compatible with input dimensions {i:?}. The last dimension must match the norm dimension.").into()}
			layer.forward(input).into()
		}
		let l=self;

		match input.float(){F1(x)=>f(x,l),F2(x)=>f(x,l),F3(x)=>f(x,l),F4(x)=>f(x,l),F5(x)=>f(x,l),F6(x)=>f(x,l),F7(x)=>f(x,l),F8(x)=>f(x,l),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(|x|AI::forward(self,x)).collect()),_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Linear<B>{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input.float(){F1(x)=>F1(self.forward(x)),F2(x)=>F2(self.forward(x)),F3(x)=>F3(self.forward(x)),F4(x)=>F4(self.forward(x)),F5(x)=>F5(self.forward(x)),F6(x)=>F6(self.forward(x)),F7(x)=>F7(self.forward(x)),F8(x)=>F8(self.forward(x)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(|x|AI::forward(self,x)).collect()),_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Relu{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input.float(){F1(x)=>F1(self.forward(x)),F2(x)=>F2(self.forward(x)),F3(x)=>F3(self.forward(x)),F4(x)=>F4(self.forward(x)),F5(x)=>F5(self.forward(x)),F6(x)=>F6(self.forward(x)),F7(x)=>F7(self.forward(x)),F8(x)=>F8(self.forward(x)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(|x|AI::forward(self,x)).collect()),_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for SoftmaxLayer{
	fn forward(&self,input:Value<B>)->Value<B>{
		fn f<B:Backend,const N:usize>(dim:i32,temperature:f32,x:Tensor<B,N>)->Tensor<B,N>{
			let dim=if dim<0{N-(-dim) as usize}else{dim as usize};
			softmax(x/temperature,dim)
		}
		let (dim,temperature)=(self.get_dim(),self.get_temperature());

		match input.float(){F1(x)=>F1(f(dim,temperature,x)),F2(x)=>F2(f(dim,temperature,x)),F3(x)=>F3(f(dim,temperature,x)),F4(x)=>F4(f(dim,temperature,x)),F5(x)=>F5(f(dim,temperature,x)),F6(x)=>F6(f(dim,temperature,x)),F7(x)=>F7(f(dim,temperature,x)),F8(x)=>F8(f(dim,temperature,x)),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>x.into_iter().map(|x|self.forward(x)).collect(),_=>panic!("unexpected non float value")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for ChooseLayer{
	fn forward(&self,input:Value<B>)->Value<B>{//TODO hard choose
		let (dim,temperature)=(self.get_dim(),self.get_temperature());
		let d=if dim<0{input.rank().unwrap_or(8)-((-dim) as usize)}else{dim as usize};

		match input.float(){
			F1(x)=>I1(if temperature.is_nan(){hard_choose_burn_tensor(dim,x)}else{soft_choose_burn_tensor(dim,x,temperature)}),
			F2(x)=>I1(if temperature.is_nan(){hard_choose_burn_tensor(dim,x)}else{soft_choose_burn_tensor(dim,x,temperature)}.squeeze(d)),
			F3(x)=>I2(if temperature.is_nan(){hard_choose_burn_tensor(dim,x)}else{soft_choose_burn_tensor(dim,x,temperature)}.squeeze(d)),
			F4(x)=>I3(if temperature.is_nan(){hard_choose_burn_tensor(dim,x)}else{soft_choose_burn_tensor(dim,x,temperature)}.squeeze(d)),
			F5(x)=>I4(if temperature.is_nan(){hard_choose_burn_tensor(dim,x)}else{soft_choose_burn_tensor(dim,x,temperature)}.squeeze(d)),
			F6(x)=>I5(if temperature.is_nan(){hard_choose_burn_tensor(dim,x)}else{soft_choose_burn_tensor(dim,x,temperature)}.squeeze(d)),
			F7(x)=>I6(if temperature.is_nan(){hard_choose_burn_tensor(dim,x)}else{soft_choose_burn_tensor(dim,x,temperature)}.squeeze(d)),
			F8(x)=>I7(if temperature.is_nan(){hard_choose_burn_tensor(dim,x)}else{soft_choose_burn_tensor(dim,x,temperature)}.squeeze(d)),
			Value::Incompatible(e)=>e.into(),
			Value::Multi(v)=>Value::Multi(v.into_iter().map(|v|self.forward_typed::<_,Value<B>>(v)).collect()),
			_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for MaxPool2d{
	fn forward(&self,input:Value<B>)->Value<B>{
		fn f<B:Backend,const N:usize>(pool:&MaxPool2d,x:Tensor<B,N>)->Value<B>{
			match N{
				0=>panic!("internal error"),
				1=>f::<B,2>(pool,x.unsqueeze()).squeeze(),
				2=>f::<B,3>(pool,x.unsqueeze()).squeeze(),
				3=>f::<B,4>(pool,x.unsqueeze()).squeeze(),
				4=>pool.forward(Value::from(x).unwrap_f4()).into(),
				_=>{
					let mut dims=x.dims();

					let [channels,h,w]=[dims[N-3],dims[N-2],dims[N-1]];
					let big:usize=dims.iter().take(N-3).product();
					let y=x.reshape([big,channels,h,w]);

					dims[N-3..].copy_from_slice(&y.dims()[1..]);

					let y=pool.forward(y);
					y.reshape(dims).into()
				}
			}
		}
		match input.float(){
			F1(x)=>f(self,x),
			F2(x)=>f(self,x),
			F3(x)=>f(self,x),
			F4(x)=>f(self,x),
			F5(x)=>f(self,x),
			F6(x)=>f(self,x),
			F7(x)=>f(self,x),
			F8(x)=>f(self,x),
			Value::Incompatible(e)=>e.into(),
			Value::Multi(v)=>v.into_iter().map(|x|AI::forward(self,x)).collect(),
			_=>panic!("Internal error")
		}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for RotaryEncoding<B>{
	fn forward(&self,input:Value<B>)->Value<B>{AI::forward(self,(input,0)).0}
}
impl<B:Backend> AI<(Value<B>,usize),(Value<B>,usize)> for RotaryEncoding<B>{
	fn forward(&self,(input,offset):(Value<B>,usize))->(Value<B>,usize){
		fn apply<B:Backend,const D:usize>(a:&RotaryEncoding<B>,input:Tensor<B,D>,offset:usize)->Value<B>{
			assert!(D>=2);
			const MAX_KERNEL:usize=65535;		// the library form of this operation frequently exceeds the max kernel group dimension of 2^16-1
			let device=input.device();
			let freq=&a.freq_complex;
			let shape=input.shape();

			let (context,key)=(shape.dims[D-2],shape.dims[D-1]);
			let [distance,head,_2]=freq.dims();

			if context>distance{return "context length must not exceed rotary distance".into()}
			if key%head!=0{return "input dimension must be a multiple of head".into()}
			let count=shape.num_elements();
			let big=count/(context*key);
			let heads=key/head;
			let group=count/head;				// apparently this was determined empirically from error messages
			let input=input.reshape([big,context,heads,head]).swap_dims(1,2).reshape([big*heads,context,head/2,2]);
			let sign=Tensor::<B,2>::from_floats([[1.0,0.0,0.0,1.0],[0.0,-1.0,1.0,0.0]],&device).unsqueeze();

			let chunks=input.chunk(group.div_ceil(MAX_KERNEL),0).into_iter().map(|x|{
				let smaller=x.dims()[0];
				let x=x.matmul(sign.clone()).reshape([smaller,context,head,2])*freq.clone().slice([offset..context+offset]).unsqueeze();
				x.sum_dim(3)
			}).collect();
			Tensor::cat(chunks,0).reshape([big,heads,context,head]).swap_dims(1,2).reshape::<D,_>(shape).into()
		}

		(match input.float(){F1(x)=>apply(self,x.unsqueeze::<2>(),offset).squeeze(),F2(x)=>apply(self,x,offset),F3(x)=>apply(self,x,offset),F4(x)=>apply(self,x,offset),F5(x)=>apply(self,x,offset),F6(x)=>apply(self,x,offset),F7(x)=>apply(self,x,offset),F8(x)=>apply(self,x,offset),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|AI::forward(self,(x,offset)).0).collect(),_=>panic!("internal error")},offset)
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Tanh{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input.float(){F1(x)=>F1(self.forward(x)),F2(x)=>F2(self.forward(x)),F3(x)=>F3(self.forward(x)),F4(x)=>F4(self.forward(x)),F5(x)=>F5(self.forward(x)),F6(x)=>F6(self.forward(x)),F7(x)=>F7(self.forward(x)),F8(x)=>F8(self.forward(x)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(|x|AI::forward(self,x)).collect()),_=>panic!("internal error")}
	}
}
impl<B:Backend> Abs for Value<B>{
	fn abs(self)->Self::Output{
		match self{B1(x)=>B1(x),B2(x)=>B2(x),B3(x)=>B3(x),B4(x)=>B4(x),B5(x)=>B5(x),B6(x)=>B6(x),B7(x)=>B7(x),B8(x)=>B8(x),F1(x)=>F1(x.abs()),F2(x)=>F2(x.abs()),F3(x)=>F3(x.abs()),F4(x)=>F4(x.abs()),F5(x)=>F5(x.abs()),F6(x)=>F6(x.abs()),F7(x)=>F7(x.abs()),F8(x)=>F8(x.abs()),I1(x)=>I1(x.abs()),I2(x)=>I2(x.abs()),I3(x)=>I3(x.abs()),I4(x)=>I4(x.abs()),I5(x)=>I5(x.abs()),I6(x)=>I6(x.abs()),I7(x)=>I7(x.abs()),I8(x)=>I8(x.abs()),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(Value::abs).collect()}
	}
	type Output=Value<B>;
}
impl<B:Backend> AsRef<Self> for Value<B>{
	fn as_ref(&self)->&Self{self}
}
impl<B:Backend> Decompose for LossOutput<B>{
	fn compose((loss,output,target):Self::Decomposition)->Self{Self::new(loss,output,target)}
	fn decompose(self)->Self::Decomposition{(self.loss(),self.output(),self.target())}
	fn decompose_cloned(&self)->Self::Decomposition{(self.loss(),self.output(),self.target())}
	type Decomposition=(Value<B>,Value<B>,Value<B>);
}
impl<B:Backend> Decompose for Value<B>{
	fn compose(decomposition:Self::Decomposition)->Self{decomposition}
	fn decompose(self)->Self::Decomposition{self}
	fn decompose_cloned(&self)->Self::Decomposition{self.clone()}
	type Decomposition=Self;
}
impl<B:Backend> Default for Value<B>{
	fn default()->Self{Self::Multi(Vec::new())}
}
impl<B:Backend> Display for Value<B>{
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->FmtResult{write!(f,"todo")}
}
impl<B:Backend> From<Vec<bool>> for Value<B>{
	fn from(value:Vec<bool>)->Self{
		let l=value.len();
		let t:Tensor<B,1,Bool>=Tensor::from_data(TensorData::new(value,[l]),&Default::default());

		t.into()
	}
}
impl<B:Backend> From<Vec<f32>> for Value<B>{
	fn from(value:Vec<f32>)->Self{
		let l=value.len();
		let t:Tensor<B,1>=Tensor::from_data(TensorData::new(value,[l]),&Default::default());

		t.into()
	}
}
impl<B:Backend> From<Vec<i32>> for Value<B>{
	fn from(value:Vec<i32>)->Self{
		let l=value.len();
		let t:Tensor<B,1,Int>=Tensor::from_data(TensorData::new(value,[l]),&Default::default());

		t.into()
	}
}
impl<B:Backend> From<Vec<u32>> for Value<B>{
	fn from(value:Vec<u32>)->Self{
		let l=value.len();
		let t:Tensor<B,1,Int>=Tensor::from_data(TensorData::new(value,[l]),&Default::default());

		t.into()
	}
}
impl<B:Backend> ModuleDisplay for Value<B>{
	fn custom_content(&self,_content:Content)->Option<Content>{None}
	fn custom_settings(&self)->Option<DisplaySettings>{None}
	fn format(&self,s:DisplaySettings)->String{
		match self{
			B1(x)=>x.format(s),
			B2(x)=>x.format(s),
			B3(x)=>x.format(s),
			B4(x)=>x.format(s),
			B5(x)=>x.format(s),
			B6(x)=>x.format(s),
			B7(x)=>x.format(s),
			B8(x)=>x.format(s),
			F1(x)=>x.format(s),
			F2(x)=>x.format(s),
			F3(x)=>x.format(s),
			F4(x)=>x.format(s),
			F5(x)=>x.format(s),
			F6(x)=>x.format(s),
			F7(x)=>x.format(s),
			F8(x)=>x.format(s),
			I1(x)=>x.format(s),
			I2(x)=>x.format(s),
			I3(x)=>x.format(s),
			I4(x)=>x.format(s),
			I5(x)=>x.format(s),
			I6(x)=>x.format(s),
			I7(x)=>x.format(s),
			I8(x)=>x.format(s),
			Value::Incompatible(e)=>e.to_string(),
			Value::Multi(v)=>"[".chars().chain(v.iter().flat_map(|x|{
				let x:Vec<char>=x.format(s.clone()).chars().chain(", ".chars()).collect();
				x
			})).chain("]".chars()).collect()
		}
	}
}
impl<B:Backend> ModuleDisplayDefault for Value<B>{
	fn content(&self,content:Content)->Option<Content>{Some(content)}
	fn num_params(&self)->usize{Module::num_params(self)}
}
impl<B:Backend> From<String> for Value<B>{
	fn from(value:String)->Self{Self::Incompatible(value)}
}
impl<B:Backend> From<Value<B>> for ValueData{
	fn from(value:Value<B>)->Self{
		match value{B1(x)=>BX(x.into_data()),B2(x)=>BX(x.into_data()),B3(x)=>BX(x.into_data()),B4(x)=>BX(x.into_data()),B5(x)=>BX(x.into_data()),B6(x)=>BX(x.into_data()),B7(x)=>BX(x.into_data()),B8(x)=>BX(x.into_data()),F1(x)=>FX(x.into_data()),F2(x)=>FX(x.into_data()),F3(x)=>FX(x.into_data()),F4(x)=>FX(x.into_data()),F5(x)=>FX(x.into_data()),F6(x)=>FX(x.into_data()),F7(x)=>FX(x.into_data()),F8(x)=>FX(x.into_data()),I1(x)=>IX(x.into_data()),I2(x)=>IX(x.into_data()),I3(x)=>IX(x.into_data()),I4(x)=>IX(x.into_data()),I5(x)=>IX(x.into_data()),I6(x)=>IX(x.into_data()),I7(x)=>IX(x.into_data()),I8(x)=>IX(x.into_data()),Value::Incompatible(e)=>ValueData::Incompatible(e),Value::Multi(v)=>ValueData::Multi(v.into_iter().map(ValueData::from).collect())}
	}
}
impl<B:Backend> From<ValueData> for Value<B>{
	fn from(value:ValueData)->Self{
		let device=Default::default();
		match value{
			BX(data)=>match data.shape.len(){1=>B1(Tensor::from_data(data,&device)),2=>B2(Tensor::from_data(data,&device)),3=>B3(Tensor::from_data(data,&device)),4=>B4(Tensor::from_data(data,&device)),5=>B5(Tensor::from_data(data,&device)),6=>B6(Tensor::from_data(data,&device)),7=>B7(Tensor::from_data(data,&device)),8=>B8(Tensor::from_data(data,&device)),_=>panic!("tensor ranks above 8 are currently not supported")},
			FX(data)=>match data.shape.len(){1=>F1(Tensor::from_data(data,&device)),2=>F2(Tensor::from_data(data,&device)),3=>F3(Tensor::from_data(data,&device)),4=>F4(Tensor::from_data(data,&device)),5=>F5(Tensor::from_data(data,&device)),6=>F6(Tensor::from_data(data,&device)),7=>F7(Tensor::from_data(data,&device)),8=>F8(Tensor::from_data(data,&device)),_=>panic!("tensor ranks above 8 are currently not supported")},
			IX(data)=>match data.shape.len(){1=>I1(Tensor::from_data(data,&device)),2=>I2(Tensor::from_data(data,&device)),3=>I3(Tensor::from_data(data,&device)),4=>I4(Tensor::from_data(data,&device)),5=>I5(Tensor::from_data(data,&device)),6=>I6(Tensor::from_data(data,&device)),7=>I7(Tensor::from_data(data,&device)),8=>I8(Tensor::from_data(data,&device)),_=>panic!("tensor ranks above 8 are currently not supported")},
			ValueData::Incompatible(e)=>e.into(),
			ValueData::Multi(v)=>v.into_iter().map(Value::from).collect(),
		}
	}
}
impl<B:Backend> From<Vec<Value<B>>> for Value<B>{
	fn from(value:Vec<Value<B>>)->Self{Self::Multi(value)}
}
impl<B:Backend> IntoIterator for Value<B>{
	fn into_iter(self)->Self::IntoIter{self.into_multi().into_iter()}
	type IntoIter=VecIntoIter<Value<B>>;
	type Item=Value<B>;
}
impl<B:Backend> LossOutput<B>{
	/// references the loss
	pub fn loss(&self)->Value<B>{self.loss.clone()}
	/// creates a new loss output
	pub fn new(loss:Value<B>,output:Value<B>,target:Value<B>)->Self{
		Self{loss,output,target}
	}
	/// gets the output
	pub fn output(&self)->Value<B>{self.output.clone()}
	/// gets the target
	pub fn target(&self)->Value<B>{self.target.clone()}
}
impl<B:Backend> Merge for Value<B>{
	fn merge(&mut self,other:Self){
		match (mem::take(self),other){
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
				v.insert(0,u);
				*self=v.into();
			},
			(u,v)=>*self=vec![u,v].into()
		}
	}
}
impl<B:Backend> Module<B> for Value<B>{
	fn collect_devices(&self,devices:Vec<<B as Backend>::Device>)->Vec<<B as Backend>::Device>{
		match self{B1(x)=>x.collect_devices(devices),B2(x)=>x.collect_devices(devices),B3(x)=>x.collect_devices(devices),B4(x)=>x.collect_devices(devices),B5(x)=>x.collect_devices(devices),B6(x)=>x.collect_devices(devices),B7(x)=>x.collect_devices(devices),B8(x)=>x.collect_devices(devices),F1(x)=>x.collect_devices(devices),F2(x)=>x.collect_devices(devices),F3(x)=>x.collect_devices(devices),F4(x)=>x.collect_devices(devices),F5(x)=>x.collect_devices(devices),F6(x)=>x.collect_devices(devices),F7(x)=>x.collect_devices(devices),F8(x)=>x.collect_devices(devices),I1(x)=>x.collect_devices(devices),I2(x)=>x.collect_devices(devices),I3(x)=>x.collect_devices(devices),I4(x)=>x.collect_devices(devices),I5(x)=>x.collect_devices(devices),I6(x)=>x.collect_devices(devices),I7(x)=>x.collect_devices(devices),I8(x)=>x.collect_devices(devices),Value::Incompatible(_e)=>devices,Value::Multi(v)=>v.iter().fold(devices,|devices,x|x.collect_devices(devices))}
	}
	fn devices(&self)->Vec<<B as Backend>::Device>{self.collect_devices(Vec::new())}
	fn fork(self,device:&<B as Backend>::Device)->Self{
		match self{B1(x)=>B1(x.fork(device)),B2(x)=>B2(x.fork(device)),B3(x)=>B3(x.fork(device)),B4(x)=>B4(x.fork(device)),B5(x)=>B5(x.fork(device)),B6(x)=>B6(x.fork(device)),B7(x)=>B7(x.fork(device)),B8(x)=>B8(x.fork(device)),F1(x)=>F1(x.fork(device)),F2(x)=>F2(x.fork(device)),F3(x)=>F3(x.fork(device)),F4(x)=>F4(x.fork(device)),F5(x)=>F5(x.fork(device)),F6(x)=>F6(x.fork(device)),F7(x)=>F7(x.fork(device)),F8(x)=>F8(x.fork(device)),I1(x)=>I1(x.fork(device)),I2(x)=>I2(x.fork(device)),I3(x)=>I3(x.fork(device)),I4(x)=>I4(x.fork(device)),I5(x)=>I5(x.fork(device)),I6(x)=>I6(x.fork(device)),I7(x)=>I7(x.fork(device)),I8(x)=>I8(x.fork(device)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|x.fork(device)).collect()}
	}
	fn into_record(self)->Self::Record{ConstantRecord}
	fn load_file<F:FileRecorder<B>,P:Into<PathBuf>>(self,_filepath:P,_recorder:&F,_device:&<B as Backend>::Device)->Result<Self,RecorderError>{Ok(self)}
	fn load_record(self,_record:Self::Record)->Self{self}
	fn map<Mapper:ModuleMapper<B>>(self,mapper:&mut Mapper)->Self{
		match self{B1(x)=>B1(x.map(mapper)),B2(x)=>B2(x.map(mapper)),B3(x)=>B3(x.map(mapper)),B4(x)=>B4(x.map(mapper)),B5(x)=>B5(x.map(mapper)),B6(x)=>B6(x.map(mapper)),B7(x)=>B7(x.map(mapper)),B8(x)=>B8(x.map(mapper)),F1(x)=>F1(x.map(mapper)),F2(x)=>F2(x.map(mapper)),F3(x)=>F3(x.map(mapper)),F4(x)=>F4(x.map(mapper)),F5(x)=>F5(x.map(mapper)),F6(x)=>F6(x.map(mapper)),F7(x)=>F7(x.map(mapper)),F8(x)=>F8(x.map(mapper)),I1(x)=>I1(x.map(mapper)),I2(x)=>I2(x.map(mapper)),I3(x)=>I3(x.map(mapper)),I4(x)=>I4(x.map(mapper)),I5(x)=>I5(x.map(mapper)),I6(x)=>I6(x.map(mapper)),I7(x)=>I7(x.map(mapper)),I8(x)=>I8(x.map(mapper)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|x.map(mapper)).collect()}
	}
	fn num_params(&self)->usize{
		match self{B1(x)=>Module::num_params(x),B2(x)=>Module::num_params(x),B3(x)=>Module::num_params(x),B4(x)=>Module::num_params(x),B5(x)=>Module::num_params(x),B6(x)=>Module::num_params(x),B7(x)=>Module::num_params(x),B8(x)=>Module::num_params(x),F1(x)=>Module::num_params(x),F2(x)=>Module::num_params(x),F3(x)=>Module::num_params(x),F4(x)=>Module::num_params(x),F5(x)=>Module::num_params(x),F6(x)=>Module::num_params(x),F7(x)=>Module::num_params(x),F8(x)=>Module::num_params(x),I1(x)=>Module::num_params(x),I2(x)=>Module::num_params(x),I3(x)=>Module::num_params(x),I4(x)=>Module::num_params(x),I5(x)=>Module::num_params(x),I6(x)=>Module::num_params(x),I7(x)=>Module::num_params(x),I8(x)=>Module::num_params(x),Value::Incompatible(_e)=>0,Value::Multi(v)=>v.into_iter().map(|x|Module::num_params(x)).sum()}
	}
	fn quantize_weights(self,quantizer:&mut Quantizer)->Self{
		match self{B1(x)=>B1(x.quantize_weights(quantizer)),B2(x)=>B2(x.quantize_weights(quantizer)),B3(x)=>B3(x.quantize_weights(quantizer)),B4(x)=>B4(x.quantize_weights(quantizer)),B5(x)=>B5(x.quantize_weights(quantizer)),B6(x)=>B6(x.quantize_weights(quantizer)),B7(x)=>B7(x.quantize_weights(quantizer)),B8(x)=>B8(x.quantize_weights(quantizer)),F1(x)=>F1(x.quantize_weights(quantizer)),F2(x)=>F2(x.quantize_weights(quantizer)),F3(x)=>F3(x.quantize_weights(quantizer)),F4(x)=>F4(x.quantize_weights(quantizer)),F5(x)=>F5(x.quantize_weights(quantizer)),F6(x)=>F6(x.quantize_weights(quantizer)),F7(x)=>F7(x.quantize_weights(quantizer)),F8(x)=>F8(x.quantize_weights(quantizer)),I1(x)=>I1(x.quantize_weights(quantizer)),I2(x)=>I2(x.quantize_weights(quantizer)),I3(x)=>I3(x.quantize_weights(quantizer)),I4(x)=>I4(x.quantize_weights(quantizer)),I5(x)=>I5(x.quantize_weights(quantizer)),I6(x)=>I6(x.quantize_weights(quantizer)),I7(x)=>I7(x.quantize_weights(quantizer)),I8(x)=>I8(x.quantize_weights(quantizer)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|x.quantize_weights(quantizer)).collect()}
	}
	fn save_file<F:FileRecorder<B>,P:Into<PathBuf>>(self,_filepath:P,_recorder:&F)->Result<(),RecorderError>{
		Ok(())
	}
	fn to_device(self,device:&<B as Backend>::Device)->Self{
		match self{B1(x)=>B1(x.to_device(device)),B2(x)=>B2(x.to_device(device)),B3(x)=>B3(x.to_device(device)),B4(x)=>B4(x.to_device(device)),B5(x)=>B5(x.to_device(device)),B6(x)=>B6(x.to_device(device)),B7(x)=>B7(x.to_device(device)),B8(x)=>B8(x.to_device(device)),F1(x)=>F1(x.to_device(device)),F2(x)=>F2(x.to_device(device)),F3(x)=>F3(x.to_device(device)),F4(x)=>F4(x.to_device(device)),F5(x)=>F5(x.to_device(device)),F6(x)=>F6(x.to_device(device)),F7(x)=>F7(x.to_device(device)),F8(x)=>F8(x.to_device(device)),I1(x)=>I1(x.to_device(device)),I2(x)=>I2(x.to_device(device)),I3(x)=>I3(x.to_device(device)),I4(x)=>I4(x.to_device(device)),I5(x)=>I5(x.to_device(device)),I6(x)=>I6(x.to_device(device)),I7(x)=>I7(x.to_device(device)),I8(x)=>I8(x.to_device(device)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|x.to_device(device)).collect()}
	}
	fn visit<Visitor:ModuleVisitor<B>>(&self,visitor:&mut Visitor){
		match self{B1(x)=>x.visit(visitor),B2(x)=>x.visit(visitor),B3(x)=>x.visit(visitor),B4(x)=>x.visit(visitor),B5(x)=>x.visit(visitor),B6(x)=>x.visit(visitor),B7(x)=>x.visit(visitor),B8(x)=>x.visit(visitor),F1(x)=>x.visit(visitor),F2(x)=>x.visit(visitor),F3(x)=>x.visit(visitor),F4(x)=>x.visit(visitor),F5(x)=>x.visit(visitor),F6(x)=>x.visit(visitor),F7(x)=>x.visit(visitor),F8(x)=>x.visit(visitor),I1(x)=>x.visit(visitor),I2(x)=>x.visit(visitor),I3(x)=>x.visit(visitor),I4(x)=>x.visit(visitor),I5(x)=>x.visit(visitor),I6(x)=>x.visit(visitor),I7(x)=>x.visit(visitor),I8(x)=>x.visit(visitor),Value::Incompatible(_e)=>(),Value::Multi(v)=>v.iter().for_each(|x|x.visit(visitor))}
	}
	type Record=ConstantRecord;
}
impl<B:Backend> Serialize for Value<B>{
	fn serialize<S:Serializer>(&self,serializer:S)->Result<S::Ok,S::Error>{ValueData::from(self.clone()).serialize(serializer)}
}
impl<B:Backend> Value<B>{//TODO scalars
	/// tests if all values are true
	pub fn all(self)->Value<B>{
		fn f<B:Backend,const N:usize>(x:Tensor<B,N,Bool>)->Value<B>{x.all().into()}
		match self.bool(){B1(x)=>f(x),B2(x)=>f(x),B3(x)=>f(x),B4(x)=>f(x),B5(x)=>f(x),B6(x)=>f(x),B7(x)=>f(x),B8(x)=>f(x),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(Value::all).collect(),_=>panic!("internal error")}
	}
	/// tests if all values are true along the dim
	pub fn all_dim(self,d:i32)->Value<B>{
		fn f<B:Backend,const N:usize>(d:i32,x:Tensor<B,N,Bool>)->Value<B>{
			if d>=N as i32||d<(-(N as i32)){return format!("rank {N} is too low to all along dimension {d}").into()}
			let d=if d<0{N-((-d) as usize)}else{d as usize};
			x.all_dim(d).into()
		}
		match self.bool(){B1(x)=>f(d,x),B2(x)=>f(d,x),B3(x)=>f(d,x),B4(x)=>f(d,x),B5(x)=>f(d,x),B6(x)=>f(d,x),B7(x)=>f(d,x),B8(x)=>f(d,x),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|v|v.all_dim(d)).collect(),_=>panic!("internal error")}
	}
	/// tests if any values are true
	pub fn any(self)->Value<B>{
		fn f<B:Backend,const N:usize>(x:Tensor<B,N,Bool>)->Value<B>{x.any().into()}
		match self.bool(){B1(x)=>f(x),B2(x)=>f(x),B3(x)=>f(x),B4(x)=>f(x),B5(x)=>f(x),B6(x)=>f(x),B7(x)=>f(x),B8(x)=>f(x),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(Value::any).collect(),_=>panic!("internal error")}
	}
	/// tests if any values are true along the dim
	pub fn any_dim(self,d:i32)->Value<B>{
		fn f<B:Backend,const N:usize>(d:i32,x:Tensor<B,N,Bool>)->Value<B>{
			if d>=N as i32||d<(-(N as i32)){return format!("rank {N} is too low to any along dimension {d}").into()}
			let d=if d<0{N-((-d) as usize)}else{d as usize};
			x.any_dim(d).into()
		}
		match self.bool(){B1(x)=>f(d,x),B2(x)=>f(d,x),B3(x)=>f(d,x),B4(x)=>f(d,x),B5(x)=>f(d,x),B6(x)=>f(d,x),B7(x)=>f(d,x),B8(x)=>f(d,x),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|v|v.any_dim(d)).collect(),_=>panic!("internal error")}
	}
	/// casts to a bool tensor if not one
	pub fn bool(self)->Value<B>{
		match self{B1(x)=>B1(x),B2(x)=>B2(x),B3(x)=>B3(x),B4(x)=>B4(x),B5(x)=>B5(x),B6(x)=>B6(x),B7(x)=>B7(x),B8(x)=>B8(x),F1(x)=>B1(x.bool()),F2(x)=>B2(x.bool()),F3(x)=>B3(x.bool()),F4(x)=>B4(x.bool()),F5(x)=>B5(x.bool()),F6(x)=>B6(x.bool()),F7(x)=>B7(x.bool()),F8(x)=>B8(x.bool()),I1(x)=>B1(x.bool()),I2(x)=>B2(x.bool()),I3(x)=>B3(x.bool()),I4(x)=>B4(x.bool()),I5(x)=>B5(x.bool()),I6(x)=>B6(x.bool()),I7(x)=>B7(x.bool()),I8(x)=>B8(x.bool()),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(Value::bool).collect())}
	}
	/// concatenates the multi tensor along dimension d
	pub fn cat(self,d:i32)->Self{
		fn f<B:Backend,I:Iterator<Item=Tensor<B,N,K>>,K:BasicOps<B>+TensorKind<B>,const N:usize>(d:i32,x0:Tensor<B,N,K>,tensors:I)->Value<B> where Tensor<B,N,K>:Into<Value<B>>{
			if d>=N as i32||d<(-(N as i32)){return format!("rank {N} is too low to cat along dimension {d}").into()}
			let d=if d<0{N-((-d) as usize)}else{d as usize};
			let shape=x0.dims();
			let tensors:Vec<Tensor<B,N,K>>=once(x0).chain(tensors).collect();

			if let Err(e)=tensors.iter().try_for_each(|x|{
				let mut xshape=x.dims();
				xshape[d]=shape[d];
				if shape==xshape{Ok(())}else{Err("mismatched shapes {shape:?}, {xshape:?}")}
			}){
				return e.into()
			}

			Tensor::cat(tensors,d).into()
		}
 		let v=if let Value::Multi(v)=self{v}else{return self};

		if let Some(n)=v.iter().position(Value::is_incompatible){return v.into_iter().nth(n).unwrap()}
		if v.iter().all(Value::is_multi){return v.into_iter().map(|x|x.cat(d)).collect()}
		if v.iter().any(Value::is_multi){return "cannot mix single and multi values in a cat operation".into()}
		let variant=mem::discriminant(&v[0]);

		if v.iter().any(|x|mem::discriminant(x)!=variant){return "cannot mix variants in a cat operation".into()}
		let mut v=v.into_iter();

		match v.next().unwrap(){B1(x0)=>f(d,x0,v.map(Value::unwrap_b1)),B2(x0)=>f(d,x0,v.map(Value::unwrap_b2)),B3(x0)=>f(d,x0,v.map(Value::unwrap_b3)),B4(x0)=>f(d,x0,v.map(Value::unwrap_b4)),B5(x0)=>f(d,x0,v.map(Value::unwrap_b5)),B6(x0)=>f(d,x0,v.map(Value::unwrap_b6)),B7(x0)=>f(d,x0,v.map(Value::unwrap_b7)),B8(x0)=>f(d,x0,v.map(Value::unwrap_b8)),F1(x0)=>f(d,x0,v.map(Value::unwrap_f1)),F2(x0)=>f(d,x0,v.map(Value::unwrap_f2)),F3(x0)=>f(d,x0,v.map(Value::unwrap_f3)),F4(x0)=>f(d,x0,v.map(Value::unwrap_f4)),F5(x0)=>f(d,x0,v.map(Value::unwrap_f5)),F6(x0)=>f(d,x0,v.map(Value::unwrap_f6)),F7(x0)=>f(d,x0,v.map(Value::unwrap_f7)),F8(x0)=>f(d,x0,v.map(Value::unwrap_f8)),I1(x0)=>f(d,x0,v.map(Value::unwrap_i1)),I2(x0)=>f(d,x0,v.map(Value::unwrap_i2)),I3(x0)=>f(d,x0,v.map(Value::unwrap_i3)),I4(x0)=>f(d,x0,v.map(Value::unwrap_i4)),I5(x0)=>f(d,x0,v.map(Value::unwrap_i5)),I6(x0)=>f(d,x0,v.map(Value::unwrap_i6)),I7(x0)=>f(d,x0,v.map(Value::unwrap_i7)),I8(x0)=>f(d,x0,v.map(Value::unwrap_i8)),Value::Incompatible(_e)=>panic!("internal error not handled in correct location"),Value::Multi(_e)=>panic!("internal error not handled in correct location")}
	}
	/// counts the number of components in the tensor
	pub fn count(&self)->usize{
		match self{
			B1(x)=>x.dims().iter().product(),
			B2(x)=>x.dims().iter().product(),
			B3(x)=>x.dims().iter().product(),
			B4(x)=>x.dims().iter().product(),
			B5(x)=>x.dims().iter().product(),
			B6(x)=>x.dims().iter().product(),
			B7(x)=>x.dims().iter().product(),
			B8(x)=>x.dims().iter().product(),
			F1(x)=>x.dims().iter().product(),
			F2(x)=>x.dims().iter().product(),
			F3(x)=>x.dims().iter().product(),
			F4(x)=>x.dims().iter().product(),
			F5(x)=>x.dims().iter().product(),
			F6(x)=>x.dims().iter().product(),
			F7(x)=>x.dims().iter().product(),
			F8(x)=>x.dims().iter().product(),
			I1(x)=>x.dims().iter().product(),
			I2(x)=>x.dims().iter().product(),
			I3(x)=>x.dims().iter().product(),
			I4(x)=>x.dims().iter().product(),
			I5(x)=>x.dims().iter().product(),
			I6(x)=>x.dims().iter().product(),
			I7(x)=>x.dims().iter().product(),
			I8(x)=>x.dims().iter().product(),
			Value::Incompatible(_e)=>0,
			Value::Multi(v)=>v.iter().map(Value::count).sum()
		}
	}
	/// creates a new empty value
	pub fn empty()->Self{Self::Multi(Vec::new())}
	/// casts to a float tensor if not one
	pub fn float(self)->Value<B>{
		match self{B1(x)=>F1(x.float()),B2(x)=>F2(x.float()),B3(x)=>F3(x.float()),B4(x)=>F4(x.float()),B5(x)=>F5(x.float()),B6(x)=>F6(x.float()),B7(x)=>F7(x.float()),B8(x)=>F8(x.float()),F1(x)=>F1(x),F2(x)=>F2(x),F3(x)=>F3(x),F4(x)=>F4(x),F5(x)=>F5(x),F6(x)=>F6(x),F7(x)=>F7(x),F8(x)=>F8(x),I1(x)=>F1(x.float()),I2(x)=>F2(x.float()),I3(x)=>F3(x.float()),I4(x)=>F4(x.float()),I5(x)=>F5(x.float()),I6(x)=>F6(x.float()),I7(x)=>F7(x.float()),I8(x)=>F8(x.float()),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(Value::float).collect())}
	}
	/// gather
	pub fn gather(self,dim:i32,indices:Value<B>)->Self{
		fn b<B:Backend,const N:usize>(d:i32,data:Tensor<B,N,Bool>,indices:Tensor<B,N,Int>)->Value<B>{f(d,data.int(),indices).bool()}
		fn f<B:Backend,K:'static+BasicOps<B>+Numeric<B>+TensorKind<B>,const N:usize>(d:i32,data:Tensor<B,N,K>,indices:Tensor<B,N,Int>)->Value<B>{
			let d=if d<0{N-((-d) as usize)}else{d as usize};
			if d>=N{format!("dim {d} must be less than rank {N}").into()}else{data.gather(d,indices).into()}
		}

		match (self,indices){(B1(x),I1(i))=>b(dim,x,i),(B2(x),I2(i))=>b(dim,x,i),(B3(x),I3(i))=>b(dim,x,i),(B4(x),I4(i))=>b(dim,x,i),(B5(x),I5(i))=>b(dim,x,i),(B6(x),I6(i))=>b(dim,x,i),(B7(x),I7(i))=>b(dim,x,i),(B8(x),I8(i))=>b(dim,x,i),(F1(x),I1(i))=>f(dim,x,i),(F2(x),I2(i))=>f(dim,x,i),(F3(x),I3(i))=>f(dim,x,i),(F4(x),I4(i))=>f(dim,x,i),(F5(x),I5(i))=>f(dim,x,i),(F6(x),I6(i))=>f(dim,x,i),(F7(x),I7(i))=>f(dim,x,i),(F8(x),I8(i))=>f(dim,x,i),(I1(x),I1(i))=>f(dim,x,i),(I2(x),I2(i))=>f(dim,x,i),(I3(x),I3(i))=>f(dim,x,i),(I4(x),I4(i))=>f(dim,x,i),(I5(x),I5(i))=>f(dim,x,i),(I6(x),I6(i))=>f(dim,x,i),(I7(x),I7(i))=>f(dim,x,i),(I8(x),I8(i))=>f(dim,x,i),(Value::Incompatible(e),_)=>e.into(),(_,Value::Incompatible(e))=>e.into(),(Value::Multi(u),Value::Multi(v))=>u.into_iter().zip(v).map(|(u,v)|u.gather(dim,v)).collect(),_=>"gather is only available for tensors of matching dimensions with int indices".into()}
	}
	/// casts to a int tensor if not one
	pub fn int(self)->Value<B>{
		match self{B1(x)=>I1(x.int()),B2(x)=>I2(x.int()),B3(x)=>I3(x.int()),B4(x)=>I4(x.int()),B5(x)=>I5(x.int()),B6(x)=>I6(x.int()),B7(x)=>I7(x.int()),B8(x)=>I8(x.int()),F1(x)=>I1(x.int()),F2(x)=>I2(x.int()),F3(x)=>I3(x.int()),F4(x)=>I4(x.int()),F5(x)=>I5(x.int()),F6(x)=>I6(x.int()),F7(x)=>I7(x.int()),F8(x)=>I8(x.int()),I1(x)=>I1(x),I2(x)=>I2(x),I3(x)=>I3(x),I4(x)=>I4(x),I5(x)=>I5(x),I6(x)=>I6(x),I7(x)=>I7(x),I8(x)=>I8(x),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(Value::int).collect())}
	}
	/// converts to a flattened vector of floats, ignoring incompatibility errors
	pub fn into_float_vec(self)->Vec<f32>{
		fn cat_vec<T>(mut a:Vec<T>,b:Vec<T>)->Vec<T>{
			a.extend(b);
			a
		}
		fn to_vec<B:Backend,const N:usize>(x:Tensor<B,N>)->Vec<f32>{x.into_data().to_vec().unwrap_or_default()}

		match self.float(){F1(x)=>to_vec(x),F2(x)=>to_vec(x),F3(x)=>to_vec(x),F4(x)=>to_vec(x),F5(x)=>to_vec(x),F6(x)=>to_vec(x),F7(x)=>to_vec(x),F8(x)=>to_vec(x),Value::Incompatible(_e)=>Vec::new(),Value::Multi(v)=>v.into_iter().map(Value::into_float_vec).reduce(cat_vec).unwrap_or_default(),_=>panic!("internal error")}
	}
	/// tests if the tensor is empty. incompatible isn't considered empty for the purposes of this function
	pub fn is_empty(&self)->bool{self.len()==0}
	/// tests if the tensor represents the result of an incompatible input and operation
	pub fn is_incompatible(&self)->bool{
		if let Value::Incompatible(_x)=self{true}else{false}
	}
	/// converts to a multiple tensor, then unwraps to a vec of values
	pub fn into_multi(self)->Vec<Value<B>>{
		if let Value::Multi(v)=self{v}else{vec![self]}
	}
	/// tests if this is a multiple tensor
	pub fn is_multi(&self)->bool{
		if let Value::Multi(_x)=self{true}else{false}
	}
	/// shallow iteration over the contained values
	pub fn iter(&self)->SliceIter<'_,Self>{
		if let Value::Multi(v)=self{v.iter()}else{slice::from_ref(self).iter()}
	}
	/// returns the kind of value
	pub fn kind(&self)->Kind{
		match self{B1(_x)=>Kind::Bool,B2(_x)=>Kind::Bool,B3(_x)=>Kind::Bool,B4(_x)=>Kind::Bool,B5(_x)=>Kind::Bool,B6(_x)=>Kind::Bool,B7(_x)=>Kind::Bool,B8(_x)=>Kind::Bool,F1(_x)=>Kind::Float,F2(_x)=>Kind::Float,F3(_x)=>Kind::Float,F4(_x)=>Kind::Float,F5(_x)=>Kind::Float,F6(_x)=>Kind::Float,F7(_x)=>Kind::Float,F8(_x)=>Kind::Float,I1(_x)=>Kind::Int,I2(_x)=>Kind::Int,I3(_x)=>Kind::Int,I4(_x)=>Kind::Int,I5(_x)=>Kind::Int,I6(_x)=>Kind::Int,I7(_x)=>Kind::Int,I8(_x)=>Kind::Int,Value::Incompatible(_v)=>Kind::Incompatible,Value::Multi(_v)=>Kind::Multi}
	}
	/// returns a shallow count the number of values directly within this one. 1 if not multi, otherwise the len of the vec inside.
	pub fn len(&self)->usize{
		if let Value::Multi(v)=self{v.len()}else{1}
	}
	/// recursively counts the number of tensors within this value, including multi tensors within multi tensors
	pub fn len_recursive(&self)->usize{
		if let Value::Multi(v)=self{v.iter().map(Value::len_recursive).sum()}else{1}
	}
	/// mask filling
	pub fn mask_fill(self,mask:Value<B>,v:f32)->Self{
		let (x,mask)=self.promote_rank(mask.bool());
		match (x,mask){
			(B1(x),B1(m))=>B1(x.int().mask_fill(m,v).bool()),
			(B2(x),B2(m))=>B2(x.int().mask_fill(m,v).bool()),
			(B3(x),B3(m))=>B3(x.int().mask_fill(m,v).bool()),
			(B4(x),B4(m))=>B4(x.int().mask_fill(m,v).bool()),
			(B5(x),B5(m))=>B5(x.int().mask_fill(m,v).bool()),
			(B6(x),B6(m))=>B6(x.int().mask_fill(m,v).bool()),
			(B7(x),B7(m))=>B7(x.int().mask_fill(m,v).bool()),
			(B8(x),B8(m))=>B8(x.int().mask_fill(m,v).bool()),
			(F1(x),B1(m))=>F1(x.mask_fill(m,v)),
			(F2(x),B2(m))=>F2(x.mask_fill(m,v)),
			(F3(x),B3(m))=>F3(x.mask_fill(m,v)),
			(F4(x),B4(m))=>F4(x.mask_fill(m,v)),
			(F5(x),B5(m))=>F5(x.mask_fill(m,v)),
			(F6(x),B6(m))=>F6(x.mask_fill(m,v)),
			(F7(x),B7(m))=>F7(x.mask_fill(m,v)),
			(F8(x),B8(m))=>F8(x.mask_fill(m,v)),
			(I1(x),B1(m))=>I1(x.mask_fill(m,v)),
			(I2(x),B2(m))=>I2(x.mask_fill(m,v)),
			(I3(x),B3(m))=>I3(x.mask_fill(m,v)),
			(I4(x),B4(m))=>I4(x.mask_fill(m,v)),
			(I5(x),B5(m))=>I5(x.mask_fill(m,v)),
			(I6(x),B6(m))=>I6(x.mask_fill(m,v)),
			(I7(x),B7(m))=>I7(x.mask_fill(m,v)),
			(I8(x),B8(m))=>I8(x.mask_fill(m,v)),
			(Value::Incompatible(e),_)=>e.into(),
			(_,Value::Incompatible(e))=>e.into(),
			(Value::Multi(x),m)=>broadcast_multi(x,m.into_multi(),|x,m|x.mask_fill(m,v)),
			(x,Value::Multi(m))=>broadcast_multi(x.into_multi(),m,|x,m|x.mask_fill(m,v)),
			_=>panic!("internal error")
		}
	}
	/// casts to a multiple tensor if not one
	pub fn multi(self)->Self{
		if let Value::Multi(v)=self{v.into()}else{vec![self].into()}
	}
	/// creates a new multi tensor from the data and shape // TODO make work for bool and int too
	pub fn new<S:Into<Shape>>(data:&[f32],device:&B::Device,shape:S)->Self{
		match shape.into(){
			Shape::Incompatible(e)=>e.into(),
			Shape::Multi(l)=>data.chunks(l).map(|d|Value::new(d,device,X1([d.len()]))).collect(),
			Shape::Recursive(v)=>v.into_iter().scan(data,|data,s|{
				let v=Value::new(*data,device,s);
				*data=&data[..v.count()];
				Some(v)
			}).collect(),
			X1(s)=>F1(Tensor::from_data(TensorData::new(data[..s.iter().product::<usize>()].to_vec(),s),device)),
			X2(s)=>F2(Tensor::from_data(TensorData::new(data[..s.iter().product::<usize>()].to_vec(),s),device)),
			X3(s)=>F3(Tensor::from_data(TensorData::new(data[..s.iter().product::<usize>()].to_vec(),s),device)),
			X4(s)=>F4(Tensor::from_data(TensorData::new(data[..s.iter().product::<usize>()].to_vec(),s),device)),
			X5(s)=>F5(Tensor::from_data(TensorData::new(data[..s.iter().product::<usize>()].to_vec(),s),device)),
			X6(s)=>F6(Tensor::from_data(TensorData::new(data[..s.iter().product::<usize>()].to_vec(),s),device)),
			X7(s)=>F7(Tensor::from_data(TensorData::new(data[..s.iter().product::<usize>()].to_vec(),s),device)),
			X8(s)=>F8(Tensor::from_data(TensorData::new(data[..s.iter().product::<usize>()].to_vec(),s),device)),
		}
	}
	/// promotes the values to make them compatible if possible. bools can become floats or ints, ints can become floats, any can become multi, and lower ranks can be unsqueezed to higher ranks. This is a shallow operation, so tensors inside multi will be unaffected. incompatible with non multi will return the input
	pub fn promote(self,rhs:Value<B>)->(Value<B>,Value<B>){
		let (l,r)=self.promote_kind(rhs);
		l.promote_rank(r)
	}
	/// promotes the values to make them match if possible. bools can become floats or ints, ints can become floats, any can become multi. This is a shallow operation, so tensors inside multi will be unaffected. incompatible with non multi will return the input
	pub fn promote_kind(self,rhs:Value<B>)->(Value<B>,Value<B>){
		let (lk,rk)=(self.kind(),rhs.kind());

		let (mut l,mut r)=(self,rhs);
		if lk==rk{()}else if lk==Kind::Multi{r=r.multi()}else if rk==Kind::Multi{l=l.multi()}else if lk==Kind::Float{r=r.float()}else if rk==Kind::Float{l=l.float()}else if lk==Kind::Int{r=r.int()}else if rk==Kind::Int{l=l.int()}else if lk==Kind::Incompatible{return (l,r)}else if rk==Kind::Incompatible{return (l,r)}
		(l,r)
	}
	/// promotes the values to make them match if possible. lower ranks can be unsqueezed to higher ranks. This is a shallow operation, so tensors inside multi will be unaffected. incompatible with non multi will return the input
	pub fn promote_rank(self,rhs:Value<B>)->(Value<B>,Value<B>){
		let (mut l,mut r)=(self,rhs);
		let (mut lr,mut rr)=if let (Some(l),Some(r))=(l.rank(),r.rank()){(l,r)}else{return (l,r)};
		while lr<rr{
			l=l.unsqueeze();
			lr+=1;
		}
		while lr>rr{
			r=r.unsqueeze();
			rr+=1;
		}
		(l,r)
	}
	/// returns the number of axes of the tensor, or none if incompatible or multi
	pub fn rank(&self)->Option<usize>{
		match self{B1(_x)=>Some(1),B2(_x)=>Some(2),B3(_x)=>Some(3),B4(_x)=>Some(4),B5(_x)=>Some(5),B6(_x)=>Some(6),B7(_x)=>Some(7),B8(_x)=>Some(8),F1(_x)=>Some(1),F2(_x)=>Some(2),F3(_x)=>Some(3),F4(_x)=>Some(4),F5(_x)=>Some(5),F6(_x)=>Some(6),F7(_x)=>Some(7),F8(_x)=>Some(8),I1(_x)=>Some(1),I2(_x)=>Some(2),I3(_x)=>Some(3),I4(_x)=>Some(4),I5(_x)=>Some(5),I6(_x)=>Some(6),I7(_x)=>Some(7),I8(_x)=>Some(8),Value::Incompatible(_x)=>None,Value::Multi(_x)=>None}
	}
	/// gets the shape of the tensor. Use the recursive version to recursively get the multi shape
	pub fn shape(&self)->Shape{
		match self{B1(x)=>Shape::X1(x.dims()),B2(x)=>Shape::X2(x.dims()),B3(x)=>Shape::X3(x.dims()),B4(x)=>Shape::X4(x.dims()),B5(x)=>Shape::X5(x.dims()),B6(x)=>Shape::X6(x.dims()),B7(x)=>Shape::X7(x.dims()),B8(x)=>Shape::X8(x.dims()),F1(x)=>Shape::X1(x.dims()),F2(x)=>Shape::X2(x.dims()),F3(x)=>Shape::X3(x.dims()),F4(x)=>Shape::X4(x.dims()),F5(x)=>Shape::X5(x.dims()),F6(x)=>Shape::X6(x.dims()),F7(x)=>Shape::X7(x.dims()),F8(x)=>Shape::X8(x.dims()),I1(x)=>Shape::X1(x.dims()),I2(x)=>Shape::X2(x.dims()),I3(x)=>Shape::X3(x.dims()),I4(x)=>Shape::X4(x.dims()),I5(x)=>Shape::X5(x.dims()),I6(x)=>Shape::X6(x.dims()),I7(x)=>Shape::X7(x.dims()),I8(x)=>Shape::X8(x.dims()),Value::Incompatible(x)=>Shape::Incompatible(x.clone()),Value::Multi(x)=>Shape::Multi(x.len())}
	}
	/// gets the shape of the tensor. Use the non recusive function if deep shape structure of multi is not required
	pub fn shape_recursive(&self)->Shape{
		if let Value::Multi(x)=self{Shape::Recursive(x.iter().map(Value::shape_recursive).collect())}else{self.shape()}
	}
	/// shifts the components right n places, maintaining the current dimensions, filling the left spot with v cast to the appropriate type
	pub fn shift(self,d:i32,n:i32,v:f32)->Self{
		fn b<B:Backend,const N:usize>(d:i32,n:i32,v:f32,x:Tensor<B,N,Bool>)->Value<B>{f(d,n,if v==0.0{0.0}else{1.0},x.int()).bool()}
		fn f<B:Backend,K:'static+BasicOps<B>+Numeric<B>+TensorKind<B>,const N:usize>(d:i32,n:i32,v:f32,x:Tensor<B,N,K>)->Value<B>{
			let device=x.device();
			let d=if d<0{N-((-d) as usize)}else{d as usize};
			let mut paddims=x.dims();
			let mut slicedims=paddims.map(|n|0..n);

			paddims[d]=n.abs() as usize;
			slicedims[d]=if n<0{(-n) as usize..slicedims[d].end}else{0..slicedims[d].end.saturating_sub(n as usize)};
			if slicedims[d].len()==0{return x.full_like(v).into()}
			let pad:Tensor<B,N,K>=Tensor::full(paddims,v,&device);
			let slice=x.slice(slicedims);

			Tensor::cat(if n<0{vec![slice,pad]}else{vec![pad,slice]},d).into()
		}
		if n==0{return self}

		match self{B1(x)=>b(d,n,v,x),B2(x)=>b(d,n,v,x),B3(x)=>b(d,n,v,x),B4(x)=>b(d,n,v,x),B5(x)=>b(d,n,v,x),B6(x)=>b(d,n,v,x),B7(x)=>b(d,n,v,x),B8(x)=>b(d,n,v,x),F1(x)=>f(d,n,v,x),F2(x)=>f(d,n,v,x),F3(x)=>f(d,n,v,x),F4(x)=>f(d,n,v,x),F5(x)=>f(d,n,v,x),F6(x)=>f(d,n,v,x),F7(x)=>f(d,n,v,x),F8(x)=>f(d,n,v,x),I1(x)=>f(d,n,v,x),I2(x)=>f(d,n,v,x),I3(x)=>f(d,n,v,x),I4(x)=>f(d,n,v,x),I5(x)=>f(d,n,v,x),I6(x)=>f(d,n,v,x),I7(x)=>f(d,n,v,x),I8(x)=>f(d,n,v,x),Value::Incompatible(e)=>e.into(),Value::Multi(x)=>x.into_iter().map(|x|x.shift(d,n,v)).collect()}
	}
	/// returns a value containing the elements selected from the given ranges. If this is a multi tensor the slice will be applied to each sub tensor
	pub fn slice<A:AsRef<[R]>,R:RangeBounds<usize>>(self,ranges:A)->Self{
		let ranges=ranges.as_ref();
		let len=ranges.len();
		if let Value::Incompatible(x)=self{return x.into()}
		let rank=self.rank().unwrap_or(len);
		let shape=self.shape();

		let mut normalizedranges=[0;8].map(|_|0..0);
		for ((d,n),r) in shape.clone().to_array(Alignment::Left).into_iter().zip(normalizedranges.iter_mut()).zip(ranges){
			n.start=match r.start_bound(){Excluded(&x)=>x+1,Included(&x)=>x,Unbounded=>0};
			n.end=match r.end_bound(){Excluded(&x)=>x,Included(&x)=>x+1,Unbounded=>d};
		}
		if len>rank{return format!("Length of ranges argument must be less than the the value's rank. len: {len} ranges: {normalizedranges:?} rank: {rank} shape: {shape:?}").into()}
		for (d,n) in shape.clone().to_array(Alignment::Left).into_iter().zip(normalizedranges.iter()).take(len){
			if n.start>=n.end{return format!("Empty or reverse ranges are currently not supported. ranges: {normalizedranges:?}").into()}
			if d<n.end{return format!("Cannot index beyond the end of a dimension. ranges: {normalizedranges:?} shape: {shape:?}").into()}
		}
		let ranges=&normalizedranges[..len];

		match self{B1(x)=>B1(slice_slice(ranges,x)),B2(x)=>B2(slice_slice(ranges,x)),B3(x)=>B3(slice_slice(ranges,x)),B4(x)=>B4(slice_slice(ranges,x)),B5(x)=>B5(slice_slice(ranges,x)),B6(x)=>B6(slice_slice(ranges,x)),B7(x)=>B7(slice_slice(ranges,x)),B8(x)=>B8(slice_slice(ranges,x)),F1(x)=>F1(slice_slice(ranges,x)),F2(x)=>F2(slice_slice(ranges,x)),F3(x)=>F3(slice_slice(ranges,x)),F4(x)=>F4(slice_slice(ranges,x)),F5(x)=>F5(slice_slice(ranges,x)),F6(x)=>F6(slice_slice(ranges,x)),F7(x)=>F7(slice_slice(ranges,x)),F8(x)=>F8(slice_slice(ranges,x)),I1(x)=>I1(slice_slice(ranges,x)),I2(x)=>I2(slice_slice(ranges,x)),I3(x)=>I3(slice_slice(ranges,x)),I4(x)=>I4(slice_slice(ranges,x)),I5(x)=>I5(slice_slice(ranges,x)),I6(x)=>I6(slice_slice(ranges,x)),I7(x)=>I7(slice_slice(ranges,x)),I8(x)=>I8(slice_slice(ranges,x)),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>Value::Multi(x.into_iter().map(|x|x.slice(ranges)).collect())}
	}
	/// splits into chunks along the dimension, or the multi vector if dim is None
	pub fn split<I:Into<Option<i32>>>(self,chunksize:usize,dim:I)->Self{
		fn f<B:Backend,K:'static+BasicOps<B>+TensorKind<B>,const N:usize>(dim:i32,size:usize,tensor:Tensor<B,N,K>)->Value<B>{
			if dim>=N as i32||dim<(-(N as i32)){return format!("rank {N} is too low to split along dimension {dim}").into()}
			let dim=if dim<0{N-((-dim) as usize)}else{dim as usize};

			tensor.split(dim,size).into_iter().map(Value::from).collect()
		}
		let c=if chunksize==0{return "cannot split into chunks of 0 size".into()}else{chunksize};

		if let Some(d)=dim.into(){
			match self{B1(x)=>f(d,c,x),B2(x)=>f(d,c,x),B3(x)=>f(d,c,x),B4(x)=>f(d,c,x),B5(x)=>f(d,c,x),B6(x)=>f(d,c,x),B7(x)=>f(d,c,x),B8(x)=>f(d,c,x),F1(x)=>f(d,c,x),F2(x)=>f(d,c,x),F3(x)=>f(d,c,x),F4(x)=>f(d,c,x),F5(x)=>f(d,c,x),F6(x)=>f(d,c,x),F7(x)=>f(d,c,x),F8(x)=>f(d,c,x),I1(x)=>f(d,c,x),I2(x)=>f(d,c,x),I3(x)=>f(d,c,x),I4(x)=>f(d,c,x),I5(x)=>f(d,c,x),I6(x)=>f(d,c,x),I7(x)=>f(d,c,x),I8(x)=>f(d,c,x),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|x.split(c,Some(d))).collect()}
		}else{
			let v=self.into_multi();
			v.chunks(chunksize).map(|c|Value::from(c.to_vec())).collect()
		}
	}
	/// squeeze 0
	pub fn squeeze(self)->Self{self.squeeze_dim(0)}
	/// removes a dimension of size 1 at position d. incompatible if dimension at position d is not 1
	pub fn squeeze_dim(self,d:i32)->Self{
		fn f<B:Backend,K:BasicOps<B>+TensorKind<B>,const D:usize,const N:usize>(x:Tensor<B,D,K>,d:i32)->Result<Tensor<B,N,K>,String>{
			let d=if d<0{D-((-d) as usize)}else{d as usize};

			if d>=D{return Err(format!("dim {d} must be less than {D}"))}
			let xdim=x.dims()[d];

			if xdim==1{Ok(x.squeeze(d))}else{Err(format!("cannot squeeze a dim of size not equal to 1. dim {d} was {xdim}"))}
		}
		match match self{B1(_x)=>Err("currently cannot decrease the number of tensor dimensions below 1".into()),B2(x)=>f(x,d).map(B1),B3(x)=>f(x,d).map(B2),B4(x)=>f(x,d).map(B3),B5(x)=>f(x,d).map(B4),B6(x)=>f(x,d).map(B5),B7(x)=>f(x,d).map(B6),B8(x)=>f(x,d).map(B7),F1(_x)=>Err("currently cannot decrease the number of tensor dimensions below 1".into()),F2(x)=>f(x,d).map(F1),F3(x)=>f(x,d).map(F2),F4(x)=>f(x,d).map(F3),F5(x)=>f(x,d).map(F4),F6(x)=>f(x,d).map(F5),F7(x)=>f(x,d).map(F6),F8(x)=>f(x,d).map(F7),I1(_x)=>Err("currently cannot decrease the number of tensor dimensions below 1".into()),I2(x)=>f(x,d).map(I1),I3(x)=>f(x,d).map(I2),I4(x)=>f(x,d).map(I3),I5(x)=>f(x,d).map(I4),I6(x)=>f(x,d).map(I5),I7(x)=>f(x,d).map(I6),I8(x)=>f(x,d).map(I7),Value::Incompatible(e)=>Err(e),Value::Multi(v)=>Ok(v.into_iter().map(|x|x.squeeze_dim(d)).collect())}{Err(e)=>e.into(),Ok(x)=>x}
	}
	/// stacks the multi tensor, inserting a dimension at d, or N+d+1 if d is negative. for singular tensors this has an unsqueezing effect
	pub fn stack(self,d:i32)->Self{self.unsqueeze_dim(d).cat(d)}
	/// attempts to unwrap the inner incompatible value
	pub fn try_incompatible(self)->Result<String,Self>{
		if let Value::Incompatible(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner multi value
	pub fn try_multi(self)->Result<Vec<Value<B>>,Self>{
		if let Value::Multi(v)=self{Ok(v)}else{Err(self)}
	}
	/// unsqueeze 0
	pub fn unsqueeze(self)->Self{self.unsqueeze_dim(0)}
	/// inserts a dimension of size 1 at position d, or N+d+1 if d is negative
	pub fn unsqueeze_dim(self,d:i32)->Self{
		fn f<B:Backend,K:BasicOps<B>+TensorKind<B>,const D:usize,const N:usize>(x:Tensor<B,D,K>,d:i32)->Tensor<B,N,K>{
			x.unsqueeze_dim(if d<0{D-((-d) as usize)+1}else{d as usize})
		}
		if let Some(r)=self.rank(){
			let e=if d<0{r-((-d) as usize)+1}else{d as usize};
			if e>r{return format!("dim {e} must be less than or equal to rank {r}").into()}
		}
		match self{B1(x)=>B2(f(x,d)),B2(x)=>B3(f(x,d)),B3(x)=>B4(f(x,d)),B4(x)=>B5(f(x,d)),B5(x)=>B6(f(x,d)),B6(x)=>B7(f(x,d)),B7(x)=>B8(f(x,d)),B8(_x)=>"currently can't increase number of tensor dimensions above 8".into(),F1(x)=>F2(f(x,d)),F2(x)=>F3(f(x,d)),F3(x)=>F4(f(x,d)),F4(x)=>F5(f(x,d)),F5(x)=>F6(f(x,d)),F6(x)=>F7(f(x,d)),F7(x)=>F8(f(x,d)),F8(_x)=>"currently can't increase number of tensor dimensions above 8".into(),I1(x)=>I2(f(x,d)),I2(x)=>I3(f(x,d)),I3(x)=>I4(f(x,d)),I4(x)=>I5(f(x,d)),I5(x)=>I6(f(x,d)),I6(x)=>I7(f(x,d)),I7(x)=>I8(f(x,d)),I8(_x)=>"currently can't increase number of tensor dimensions above 8".into(),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|x.unsqueeze_dim(d)).collect()}
	}
	#[track_caller]
	/// attempts to unwrap the inner incompatible value
	pub fn unwrap_incompatible(self)->String{self.try_incompatible().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner multi value
	pub fn unwrap_multi(self)->Vec<Value<B>>{self.try_multi().unwrap()}
	/// zeros like
	pub fn zeros_like(&self)->Value<B>{// TODO this could be more efficient for bool
		match self{B1(x)=>B1(x.clone().int().zeros_like().bool()),B2(x)=>B2(x.clone().int().zeros_like().bool()),B3(x)=>B3(x.clone().int().zeros_like().bool()),B4(x)=>B4(x.clone().int().zeros_like().bool()),B5(x)=>B5(x.clone().int().zeros_like().bool()),B6(x)=>B6(x.clone().int().zeros_like().bool()),B7(x)=>B7(x.clone().int().zeros_like().bool()),B8(x)=>B8(x.clone().int().zeros_like().bool()),F1(x)=>F1(x.zeros_like()),F2(x)=>F2(x.zeros_like()),F3(x)=>F3(x.zeros_like()),F4(x)=>F4(x.zeros_like()),F5(x)=>F5(x.zeros_like()),F6(x)=>F6(x.zeros_like()),F7(x)=>F7(x.zeros_like()),F8(x)=>F8(x.zeros_like()),I1(x)=>I1(x.zeros_like()),I2(x)=>I2(x.zeros_like()),I3(x)=>I3(x.zeros_like()),I4(x)=>I4(x.zeros_like()),I5(x)=>I5(x.zeros_like()),I6(x)=>I6(x.zeros_like()),I7(x)=>I7(x.zeros_like()),I8(x)=>I8(x.zeros_like()),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.iter().map(Value::zeros_like).collect()}
	}
	try_unwrap!(Tensor<B,1,Bool>,try_b1,unwrap_b1);
	try_unwrap!(Tensor<B,2,Bool>,try_b2,unwrap_b2);
	try_unwrap!(Tensor<B,3,Bool>,try_b3,unwrap_b3);
	try_unwrap!(Tensor<B,4,Bool>,try_b4,unwrap_b4);
	try_unwrap!(Tensor<B,5,Bool>,try_b5,unwrap_b5);
	try_unwrap!(Tensor<B,6,Bool>,try_b6,unwrap_b6);
	try_unwrap!(Tensor<B,7,Bool>,try_b7,unwrap_b7);
	try_unwrap!(Tensor<B,8,Bool>,try_b8,unwrap_b8);
	try_unwrap!(Tensor<B,1,Float>,try_f1,unwrap_f1);
	try_unwrap!(Tensor<B,2,Float>,try_f2,unwrap_f2);
	try_unwrap!(Tensor<B,3,Float>,try_f3,unwrap_f3);
	try_unwrap!(Tensor<B,4,Float>,try_f4,unwrap_f4);
	try_unwrap!(Tensor<B,5,Float>,try_f5,unwrap_f5);
	try_unwrap!(Tensor<B,6,Float>,try_f6,unwrap_f6);
	try_unwrap!(Tensor<B,7,Float>,try_f7,unwrap_f7);
	try_unwrap!(Tensor<B,8,Float>,try_f8,unwrap_f8);
	try_unwrap!(Tensor<B,1,Int>,try_i1,unwrap_i1);
	try_unwrap!(Tensor<B,2,Int>,try_i2,unwrap_i2);
	try_unwrap!(Tensor<B,3,Int>,try_i3,unwrap_i3);
	try_unwrap!(Tensor<B,4,Int>,try_i4,unwrap_i4);
	try_unwrap!(Tensor<B,5,Int>,try_i5,unwrap_i5);
	try_unwrap!(Tensor<B,6,Int>,try_i6,unwrap_i6);
	try_unwrap!(Tensor<B,7,Int>,try_i7,unwrap_i7);
	try_unwrap!(Tensor<B,8,Int>,try_i8,unwrap_i8);
}
macro_rules! bicop_num{
	($trait:ident,$traitfn:ident,$traitscalar:ident)=>(
		impl<B:Backend,E:Copy+ElementConversion> $trait<E> for &Value<B>{
			fn $traitfn(self,rhs:E)->Value<B>{self.clone().$traitfn(rhs)}
			type Output=Value<B>;
		}
		impl<B:Backend,E:Copy+ElementConversion> $trait<E> for Value<B>{
			fn $traitfn(self,rhs:E)->Value<B>{
				match self{B1(x)=>I1(x.int().$traitscalar(rhs)),B2(x)=>I2(x.int().$traitscalar(rhs)),B3(x)=>I3(x.int().$traitscalar(rhs)),B4(x)=>I4(x.int().$traitscalar(rhs)),B5(x)=>I5(x.int().$traitscalar(rhs)),B6(x)=>I6(x.int().$traitscalar(rhs)),B7(x)=>I7(x.int().$traitscalar(rhs)),B8(x)=>I8(x.int().$traitscalar(rhs)),F1(x)=>F1(x.$traitscalar(rhs)),F2(x)=>F2(x.$traitscalar(rhs)),F3(x)=>F3(x.$traitscalar(rhs)),F4(x)=>F4(x.$traitscalar(rhs)),F5(x)=>F5(x.$traitscalar(rhs)),F6(x)=>F6(x.$traitscalar(rhs)),F7(x)=>F7(x.$traitscalar(rhs)),F8(x)=>F8(x.$traitscalar(rhs)),I1(x)=>I1(x.$traitscalar(rhs)),I2(x)=>I2(x.$traitscalar(rhs)),I3(x)=>I3(x.$traitscalar(rhs)),I4(x)=>I4(x.$traitscalar(rhs)),I5(x)=>I5(x.$traitscalar(rhs)),I6(x)=>I6(x.$traitscalar(rhs)),I7(x)=>I7(x.$traitscalar(rhs)),I8(x)=>I8(x.$traitscalar(rhs)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>v.into_iter().map(|x|x.$traitfn(rhs)).collect()}
			}
			type Output=Value<B>;
		}
		impl<B:Backend> $trait<&Value<B>> for &Value<B>{
			fn $traitfn(self,rhs:&Value<B>)->Value<B>{self.clone().$traitfn(rhs.clone())}
			type Output=Value<B>;
		}
		impl<B:Backend> $trait<&Value<B>> for Value<B>{
			fn $traitfn(self,rhs:&Value<B>)->Value<B>{self.$traitfn(rhs.clone())}
			type Output=Value<B>;
		}
		impl<B:Backend> $trait<Value<B>> for &Value<B>{
			fn $traitfn(self,rhs:Value<B>)->Value<B>{self.clone().$traitfn(rhs)}
			type Output=Value<B>;
		}
		impl<B:Backend> $trait<Value<B>> for Value<B>{
			fn $traitfn(self,rhs:Value<B>)->Value<B>{// TODO check shape broadcast compatibility
				match self.promote(rhs){(B1(l),B1(r))=>I1(l.int().$traitfn(r.int())),(B2(l),B2(r))=>I2(l.int().$traitfn(r.int())),(B3(l),B3(r))=>I3(l.int().$traitfn(r.int())),(B4(l),B4(r))=>I4(l.int().$traitfn(r.int())),(B5(l),B5(r))=>I5(l.int().$traitfn(r.int())),(B6(l),B6(r))=>I6(l.int().$traitfn(r.int())),(B7(l),B7(r))=>I7(l.int().$traitfn(r.int())),(B8(l),B8(r))=>I8(l.int().$traitfn(r.int())),(F1(l),F1(r))=>F1(l.$traitfn(r)),(F2(l),F2(r))=>F2(l.$traitfn(r)),(F3(l),F3(r))=>F3(l.$traitfn(r)),(F4(l),F4(r))=>F4(l.$traitfn(r)),(F5(l),F5(r))=>F5(l.$traitfn(r)),(F6(l),F6(r))=>F6(l.$traitfn(r)),(F7(l),F7(r))=>F7(l.$traitfn(r)),(F8(l),F8(r))=>F8(l.$traitfn(r)),(I1(l),I1(r))=>I1(l.$traitfn(r)),(I2(l),I2(r))=>I2(l.$traitfn(r)),(I3(l),I3(r))=>I3(l.$traitfn(r)),(I4(l),I4(r))=>I4(l.$traitfn(r)),(I5(l),I5(r))=>I5(l.$traitfn(r)),(I6(l),I6(r))=>I6(l.$traitfn(r)),(I7(l),I7(r))=>I7(l.$traitfn(r)),(I8(l),I8(r))=>I8(l.$traitfn(r)),(Value::Incompatible(e),_)=>e.into(),(_,Value::Incompatible(e))=>e.into(),(Value::Multi(l),r)=>broadcast_multi(l,r.into_multi(),$trait::$traitfn),(l,Value::Multi(r))=>broadcast_multi(l.into_multi(),r,$trait::$traitfn),_=>panic!("couldn't promote types for $traitfn")}
			}
			type Output=Value<B>;
		}
	);
}
macro_rules! try_unwrap{
	($tensor:ty,$try_unwrap:ident,$unwrap:ident)=>{
		/// attempts to unwrap the inner value
		pub fn $try_unwrap(self)->Result<$tensor,Self>{self.try_into()}
		#[track_caller]
		/// attempts to unwrap the inner value
		pub fn $unwrap(self)->$tensor{self.try_into().unwrap()}
	}
}

#[derive(Clone,Copy,Debug,Eq,PartialEq,Deserialize,Serialize)]
/// enumerates kinds for values
pub enum Kind{Bool,Float,Incompatible,Int,Multi}
#[derive(Clone,Debug,Deserialize,Serialize)]// TODO eq that doesn't include the payload of incompatible
/// tensor shapes for Value
pub enum Shape{Incompatible(String),Multi(usize),Recursive(Vec<Shape>),X1([usize;1]),X2([usize;2]),X3([usize;3]),X4([usize;4]),X5([usize;5]),X6([usize;6]),X7([usize;7]),X8([usize;8])}
#[derive(Clone,Debug)]
/// enumerates burn tensors up to 8 dimensions, along with a variant to represent operation compatibility errors, and a variant for multiple tensors. An empty multi variant can be used to represent a lack of data. Once a the depth of a multi variant is enough for an operation to take full effect, further nesting should result in the same as applying separately
pub enum Value<B:Backend>{B1(Tensor<B,1,Bool>),B2(Tensor<B,2,Bool>),B3(Tensor<B,3,Bool>),B4(Tensor<B,4,Bool>),B5(Tensor<B,5,Bool>),B6(Tensor<B,6,Bool>),B7(Tensor<B,7,Bool>),B8(Tensor<B,8,Bool>),F1(Tensor<B,1,Float>),F2(Tensor<B,2,Float>),F3(Tensor<B,3,Float>),F4(Tensor<B,4,Float>),F5(Tensor<B,5,Float>),F6(Tensor<B,6,Float>),F7(Tensor<B,7,Float>),F8(Tensor<B,8,Float>),I1(Tensor<B,1,Int>),I2(Tensor<B,2,Int>),I3(Tensor<B,3,Int>),I4(Tensor<B,4,Int>),I5(Tensor<B,5,Int>),I6(Tensor<B,6,Int>),I7(Tensor<B,7,Int>),I8(Tensor<B,8,Int>),Incompatible(String),Multi(Vec<Self>)}
#[derive(Clone,Debug,Deserialize,Serialize)]
/// burn tensors as tensor data for serialization
pub enum ValueData{BX(TensorData),FX(TensorData),IX(TensorData),Incompatible(String),Multi(Vec<ValueData>)}
#[derive(Clone,Debug,Deserialize,Serialize)]
#[serde(bound="")]
/// general loss output for being converted into other loss outputs
pub struct LossOutput<B:Backend>{loss:Value<B>,output:Value<B>,target:Value<B>}
use {bicop_num,try_unwrap};
use Bound::{Excluded,Included,Unbounded};
use Shape::{X1,X2,X3,X4,X5,X6,X7,X8};
use Value::{B1,B2,B3,B4,B5,B6,B7,B8,F1,F2,F3,F4,F5,F6,F7,F8,I1,I2,I3,I4,I5,I6,I7,I8};
use ValueData::{BX,FX,IX};
use burn::{
	module::{AutodiffModule,ConstantRecord,Content,DisplaySettings,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor,Quantizer},
	nn::{
		BatchNorm,Dropout,Embedding,LayerNorm,Linear,Relu,RotaryEncoding,Tanh,conv::Conv2d,loss::{CrossEntropyLoss,MseLoss},pool::MaxPool2d
	},
	prelude::{Backend,Bool,Float,Int,Module,Tensor,TensorData},
	record::{FileRecorder,RecorderError},
	tensor::{
		BasicOps,ElementConversion,Numeric,TensorKind,activation::{log_softmax,softmax},backend::AutodiffBackend,cast::ToElement
	}
};
use crate::{
	AI,Decompose,Merge,Op,
	builtin::{
		Alignment,ReductionMode,math::{MeanLayer,SquaredErrorLayer,SumLayer},reinforcement::AccQLayer,soft::{ChooseLayer,CrossEntropyLayer,SoftmaxLayer},structural::CatLayer
	},
	ops::Abs
};
use rand::random;
use serde::{Deserialize,Deserializer,Serialize,Serializer};
use std::{
	any::TypeId,fmt::{Display,Result as FmtResult},iter::{FromIterator,once},mem,ops::{Add,Bound,Div,Mul,RangeBounds,Range,Rem,Sub},path::PathBuf,slice::{Iter as SliceIter,self},vec::IntoIter as VecIntoIter
};
