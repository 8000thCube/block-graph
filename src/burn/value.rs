fn slice_slice<B:Backend,K:BasicOps<B>+TensorKind<B>,const N:usize>(ranges:&[Range<usize>],tensor:Tensor<B,N,K>)->Tensor<B,N,K>{
	let mut n=0;
	let mut acc=||{
		n+=1;
		n-1
	};

	match ranges.len(){0=>tensor,1=>tensor.slice([0;1].map(|_|ranges[acc()].clone())),2=>tensor.slice([0;2].map(|_|ranges[acc()].clone())),3=>tensor.slice([0;3].map(|_|ranges[acc()].clone())),4=>tensor.slice([0;4].map(|_|ranges[acc()].clone())),5=>tensor.slice([0;5].map(|_|ranges[acc()].clone())),6=>tensor.slice([0;6].map(|_|ranges[acc()].clone())),7=>tensor.slice([0;7].map(|_|ranges[acc()].clone())),8=>tensor.slice([0;8].map(|_|ranges[acc()].clone())),_=>panic!("too many ranges for current max 8 dims")}
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
impl AsRef<Self> for Shape{//TODO more reref stuff
	fn as_ref(&self)->&Self{self}
}
impl Shape{
	/// converts to the eight dimensional array type by extending with ones. The original data will be placed according to the alignment. Multi and incompatible types will be all ones
	pub fn to_x8(self,alignment:Alignment)->[usize;8]{
		let mut result=[1;8];
		let slice=match &self{Shape::Incompatible(_e)=>return result,Shape::Multi(_v)=>return result,Shape::Recursive(_r)=>return result,X1(x)=>x.as_slice(),X2(x)=>x.as_slice(),X3(x)=>x.as_slice(),X4(x)=>x.as_slice(),X5(x)=>x.as_slice(),X6(x)=>x.as_slice(),X7(x)=>x.as_slice(),X8(x)=>x.as_slice()};
		let l=slice.len();
		match alignment{Alignment::Center=>result[4-l/2..][..l].copy_from_slice(slice),Alignment::Left=>result[..l].copy_from_slice(slice),Alignment::Right=>result[8-l..].copy_from_slice(slice)}
		result
	}
}
impl<A:Into<Value<B>>,B:Backend> FromIterator<A> for Value<B>{
	fn from_iter<I:IntoIterator<Item=A>>(iter:I)->Self{
		let v:Vec<Value<B>>=iter.into_iter().map(Into::into).collect();
		if v.len()==1{v.into_iter().next().unwrap()}else{v.into()}
	}
}
impl<B:Backend,D:WhichDims> AI<Value<B>,Value<B>> for TruncateToMatch<(),D>{//TODO test this
	fn forward(&self,input:Value<B>)->Value<B>{
		if let Value::Incompatible(e)=input{return e.into()}
		let alignment=self.alignment();
		let dims=self.dims();
		let strict=dims.is_strict();
		let variant=mem::discriminant(&input);
		let mut maxdim=0;
		let mut shouldtruncate=[false;8];

		for dim in dims.which_dims(8){
			if dim>=8{
				if strict{return "Cannot truncate dims above index 7 because more than 8 dims is currently not supported".into()}
			}else{
				maxdim=dim.max(maxdim);
				shouldtruncate[dim]=true;
			}
		}
		let mut sizes=[usize::MAX;8];
		let mut v=if let Value::Multi(v)=input{
			v
		}else{
			let rank=input.rank().unwrap();
			if maxdim>=rank{return format!("Highest dim to truncate must be less than tensor rank. dim: {maxdim} rank: {rank}").into()}
			return input
		};
		let incompatible=mem::discriminant(&Value::Incompatible(String::new()));
		let multi=variant;
		let mut shouldmulti=None;

		for x in v.iter(){
			let variant=mem::discriminant(x);
			if variant==incompatible{continue}
			if let Some(shouldmulti)=shouldmulti{
				if (variant==multi)!=shouldmulti{return "Truncate to match doesn't support mixed multi/single variants".into()}
			}else{
				shouldmulti=Some(variant==multi);
			}
			if variant==multi{continue}

			let rank=x.rank().unwrap();
			if maxdim>=rank{return format!("Highest dim to truncate must be less than tensor rank. dim: {maxdim} rank: {rank}").into()}
			shouldtruncate.iter().zip(sizes.iter_mut()).zip(x.shape().to_x8(Alignment::Left)).for_each(|((&t,s),d)|if t{*s=d.min(*s)});
		}
		for x in v.iter_mut(){
			let dims=x.shape().to_x8(Alignment::Left);
			let mut ranges=sizes.map(|s|0..s);
			let rank=if let Some(r)=x.rank(){r}else{continue};

			match alignment{
				Alignment::Center=>{
					dims.iter().zip(ranges.iter_mut()).take(rank).for_each(|(&d,r)|if r.end==usize::MAX{*r=0..d}else{*r=d/2-r.end/2..d/2+(r.end+1)/2});
				},
				Alignment::Left=>{
					dims.iter().zip(ranges.iter_mut()).take(rank).for_each(|(&d,r)|if r.end==usize::MAX{*r=0..d});
				},
				Alignment::Right=>{
					dims.iter().zip(ranges.iter_mut()).take(rank).for_each(|(&d,r)|if r.end==usize::MAX{*r=0..d}else{*r=d-r.end..d});
				}
			}
			*x=mem::take(x).slice(&ranges[..rank]);
		}
		v.into()
	}
}
impl<B:Backend,S:?Sized+AsRef<str>> From<&S> for Value<B>{
	fn from(value:&S)->Self{Self::Incompatible(value.as_ref().to_string())}
}
impl<B:Backend> AI<(Value<B>,Value<B>),Value<B>> for CrossEntropy<()>{// TODO float float version, separate logits and normal (CrossEntropy, SoftEntropy), make smoothing and such work on burn specific one
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Value<B>{
		fn cross_entropy<B:Backend,const N:usize>(y:Tensor<B,N>,t:Tensor<B,N,Int>)->Value<B>{
			if y.dims()==t.dims(){F1(y.log().gather(N-1,t).mean().neg())}else{"incompatible".into()}
		}

		match (output,target){(F2(y),I1(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F3(y),I2(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F4(y),I3(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F5(y),I4(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F6(y),I5(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F7(y),I6(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F8(y),I7(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(Value::Incompatible(y),_)=>y.into(),(_,Value::Incompatible(t))=>t.into(),(Value::Multi(y),Value::Multi(t))=>Value::Multi(y.into_iter().zip(t).map(|x|self.forward_typed::<_,Value<B>>(x)).collect()),_=>"incompatible".into()}
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),Value<B>> for CrossEntropyLoss<B>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Value<B>{
		if self.logits{ai::new().fix_type::<Value<B>>().soft_entropy().forward((output,target))}else{ai::new().fix_type::<Value<B>>().cross_entropy().forward((output,target))}
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),LossOutput<B>> for SquaredError<()>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->LossOutput<B>{
		let loss=self.forward((output.clone(),target.clone()));
		LossOutput::new(loss,output,target)
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),Value<B>> for SquaredError<()>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Value<B>{
		match (output.float(),target.float()){(F1(y),F1(t))=>MseLoss.forward_no_reduction(y,t).into(),(F2(y),F2(t))=>MseLoss.forward_no_reduction(y,t).into(),(F3(y),F3(t))=>MseLoss.forward_no_reduction(y,t).into(),(F4(y),F4(t))=>MseLoss.forward_no_reduction(y,t).into(),(F5(y),F5(t))=>MseLoss.forward_no_reduction(y,t).into(),(F6(y),F6(t))=>MseLoss.forward_no_reduction(y,t).into(),(F7(y),F7(t))=>MseLoss.forward_no_reduction(y,t).into(),(F8(y),F8(t))=>MseLoss.forward_no_reduction(y,t).into(),(Value::Incompatible(y),_)=>y.into(),(_,Value::Incompatible(t))=>t.into(),(Value::Multi(y),Value::Multi(t))=>Value::Multi(y.into_iter().zip(t).map(|x|self.forward_typed::<_,Value<B>>(x)).collect()),_=>"compatible inputs for squared error are float tensors of the same shape".into()}
	}
}
impl<B:Backend> AI<(Value<B>,Value<B>),Vec<f32>> for SquaredError<()>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Vec<f32>{
		let error:Value<B>=self.forward((output,target));
		error.into_float_vec()
	}
}

/*
impl<B:Backend> AI<(Value<B>,Value<B>),Value<B>> for MSE<()>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Value<B>{
		fn mse<B:Backend,const N:usize>(y:Tensor<B,N>,t:Tensor<B,N>)->Value<B>{
			if y.dims()==t.dims(){F1(MseLoss.forward_no_reduction(y,t).mean())}else{"compatible inputs for squared error are tensors of the same shape".into()}
		}

		match (output.float(),target.float()){(F1(y),F1(t))=>mse(y,t),(F2(y),F2(t))=>mse(y,t),(F3(y),F3(t))=>mse(y,t),(F4(y),F4(t))=>mse(y,t),(F5(y),F5(t))=>mse(y,t),(F6(y),F6(t))=>mse(y,t),(F7(y),F7(t))=>mse(y,t),(F8(y),F8(t))=>mse(y,t),(Value::Incompatible(y),_)=>y.into(),(_,Value::Incompatible(t))=>t.into(),(Value::Multi(y),Value::Multi(t))=>Value::Multi(y.into_iter().zip(t).map(|x|self.forward_typed::<_,Value<B>>(x)).collect()),_=>"compatible inputs for squared error are float tensors of the same shape".into()}
	}
}*/
/*
impl<B:Backend> AI<(Value<B>,Value<B>),Value<B>> for MseLoss{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Value<B>{ai::new().fix_type::<Value<B>>().mse().forward((output,target))}
}*/
impl<B:Backend> AI<(Value<B>,Value<B>),Value<B>> for SoftEntropy<()>{// TODO float float version, separate logits and normal (CrossEntropy, SoftEntropy), make smoothing and such work on burn specific one
	fn forward(&self,(output,target):(Value<B>,Value<B>))->Value<B>{
		fn cross_entropy<B:Backend,const N:usize>(y:Tensor<B,N>,t:Tensor<B,N,Int>)->Value<B>{
			if y.dims()==t.dims(){F1(log_softmax(y,N-1).gather(N-1,t).mean().neg())}else{"incompatible".into()}
		}

		match (output,target){(F2(y),I1(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F3(y),I2(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F4(y),I3(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F5(y),I4(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F6(y),I5(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F7(y),I6(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(F8(y),I7(t))=>cross_entropy(y,t.unsqueeze_dim(1)),(Value::Incompatible(y),_)=>y.into(),(_,Value::Incompatible(t))=>t.into(),(Value::Multi(y),Value::Multi(t))=>Value::Multi(y.into_iter().zip(t).map(|x|self.forward_typed::<_,Value<B>>(x)).collect()),_=>"incompatible".into()}
	}
}

impl<B:Backend> AI<(Value<B>,Value<B>),f32> for SoftEntropy<()>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->f32{ai::new().fix_type::<Value<B>>().soft_entropy().mean().forward((output,target))}
}
impl<B:Backend> AI<(Value<B>,Value<B>),f32> for CrossEntropy<()>{
	fn forward(&self,(output,target):(Value<B>,Value<B>))->f32{ai::new().fix_type::<Value<B>>().cross_entropy().mean().forward((output,target))}
}
impl<B:Backend> AI<Value<B>,Tensor<B,1>> for Mean<()>{
	fn forward(&self,input:Value<B>)->Tensor<B,1>{
		fn avg<B:Backend,const N:usize>(x:Tensor<B,N>)->Tensor<B,1>{x.mean()}
		let l=input.len();

		if l==0{return Tensor::from_data(TensorData::new(vec![f32::NAN],[1]),&Default::default())}
		match input.float(){F1(x)=>avg(x),F2(x)=>avg(x),F3(x)=>avg(x),F4(x)=>avg(x),F5(x)=>avg(x),F6(x)=>avg(x),F7(x)=>avg(x),F8(x)=>avg(x),Value::Incompatible(e)=>panic!("Could not reduce to a scalar due to incompatibility: {e}"),Value::Multi(v)=>v.into_iter().map(|x|self.forward_typed::<_,Tensor<B,1>>(x)).reduce(|x,y|x+y).unwrap()/l as f32,_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Mean<()>{
	fn forward(&self,input:Value<B>)->Value<B>{
		//TODO tensor mean version
		F1(self.forward(input))
	}
}
impl<B:Backend> AI<Value<B>,f32> for Mean<()>{
	fn forward(&self,input:Value<B>)->f32{
		let y:Tensor<B,1>=self.forward(input);
		y.into_scalar().to_f32()
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for AccQ<()>{
	fn forward(&self,input:Value<B>)->Value<B>{
		fn acc_q<B:Backend,const N:usize>(dim:usize,gamma:f32,i:Tensor<B,N>)->Tensor<B,N>{
			let mut q=i.split(1,dim);
			q.iter_mut().rev().fold(None,|future,present|{
				if let Some(f)=future{*present=f*gamma+present.clone()}
				Some(present.clone())
			});
			Tensor::cat(q,dim)
		}
		let (dim,gamma)=(self.dim(),self.gamma());

		match input.float(){F1(x)=>F1(acc_q(dim,gamma,x)),F2(x)=>F2(acc_q(dim,gamma,x)),F3(x)=>F3(acc_q(dim,gamma,x)),F4(x)=>F4(acc_q(dim,gamma,x)),F5(x)=>F5(acc_q(dim,gamma,x)),F6(x)=>F6(acc_q(dim,gamma,x)),F7(x)=>F7(acc_q(dim,gamma,x)),F8(x)=>F8(acc_q(dim,gamma,x)),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>Value::Multi(x.into_iter().map(|x|self.forward(x)).collect()),_=>panic!("unexpected non float value")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Cat<()>{
	fn forward(&self,input:Value<B>)->Value<B>{input.cat(self.dim())}
}
impl<B:Backend> AI<Value<B>,u32> for SoftChoose<()>{
	fn forward(&self,input:Value<B>)->u32{//TODO convert multi on this to single
		let (dim,temperature)=(self.dim(),self.temperature());

		match input.float(){F1(x)=>soft_choose_burn_1(dim,x,temperature),F2(x)=>soft_choose_burn_1(dim,x,temperature),F3(x)=>soft_choose_burn_1(dim,x,temperature),F4(x)=>soft_choose_burn_1(dim,x,temperature),F5(x)=>soft_choose_burn_1(dim,x,temperature),F6(x)=>soft_choose_burn_1(dim,x,temperature),F7(x)=>soft_choose_burn_1(dim,x,temperature),F8(x)=>soft_choose_burn_1(dim,x,temperature),Value::Incompatible(e)=>panic!("Could not create scalar due to incompatibility: {e}"),Value::Multi(_v)=>panic!("Cannot soft choose one scalar from multiple values"),_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Vec<u32>> for SoftChoose<()>{
	fn forward(&self,input:Value<B>)->Vec<u32>{
		let (dim,temperature)=(self.dim(),self.temperature());

		match input.float(){F1(x)=>soft_choose_burn_multi(dim,x,temperature),F2(x)=>soft_choose_burn_multi(dim,x,temperature),F3(x)=>soft_choose_burn_multi(dim,x,temperature),F4(x)=>soft_choose_burn_multi(dim,x,temperature),F5(x)=>soft_choose_burn_multi(dim,x,temperature),F6(x)=>soft_choose_burn_multi(dim,x,temperature),F7(x)=>soft_choose_burn_multi(dim,x,temperature),F8(x)=>soft_choose_burn_multi(dim,x,temperature),Value::Incompatible(e)=>panic!("Could not create vector due to incompatibility: {e}"),Value::Multi(v)=>v.into_iter().flat_map(|x|self.forward_typed::<_,Vec<u32>>(x)).collect(),_=>panic!("internal error")}
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
		match input.float(){F1(x)=>F1(self.forward(x)),F2(x)=>F2(self.forward(x)),F3(x)=>F3(self.forward(x)),F4(x)=>F4(self.forward(x)),F5(x)=>F5(self.forward(x)),F6(x)=>F6(self.forward(x)),F7(x)=>F7(self.forward(x)),F8(x)=>F8(self.forward(x)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(|x|AI::forward(self,x)).collect()),_=>panic!("internal error")}
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
impl<B:Backend> AI<Value<B>,Value<B>> for SoftChoose<()>{
	fn forward(&self,input:Value<B>)->Value<B>{
		let (dim,temperature)=(self.dim(),self.temperature());

		match input.float(){F1(x)=>I1(soft_choose_burn_tensor(dim,x,temperature)),F2(x)=>I1(soft_choose_burn_tensor(dim,x,temperature).squeeze(1)),F3(x)=>I2(soft_choose_burn_tensor(dim,x,temperature).squeeze(2)),F4(x)=>I3(soft_choose_burn_tensor(dim,x,temperature).squeeze(3)),F5(x)=>I4(soft_choose_burn_tensor(dim,x,temperature).squeeze(4)),F6(x)=>I5(soft_choose_burn_tensor(dim,x,temperature).squeeze(5)),F7(x)=>I6(soft_choose_burn_tensor(dim,x,temperature).squeeze(6)),F8(x)=>I7(soft_choose_burn_tensor(dim,x,temperature).squeeze(7)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(|v|self.forward_typed::<_,Value<B>>(v)).collect()),_=>panic!("internal error")}
	}
}
impl<B:Backend> AI<Value<B>,Value<B>> for Tanh{
	fn forward(&self,input:Value<B>)->Value<B>{
		match input.float(){F1(x)=>F1(self.forward(x)),F2(x)=>F2(self.forward(x)),F3(x)=>F3(self.forward(x)),F4(x)=>F4(self.forward(x)),F5(x)=>F5(self.forward(x)),F6(x)=>F6(self.forward(x)),F7(x)=>F7(self.forward(x)),F8(x)=>F8(self.forward(x)),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(|x|AI::forward(self,x)).collect()),_=>panic!("internal error")}
	}
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
				v.push(u);
				*self=v.into();
			},
			(u,v)=>*self=vec![u,v].into()
		}
	}
}
impl<B:Backend> Value<B>{// TODO more builtin functions // TODO be more decisive about whether multi 1 and single are equivalent //TODO stack errors if possible
	/// concatenates the multi tensor along dimension d.
	pub fn cat(self,d:usize)->Self{
		if d>7{return format!("since the max rank is 8, the max dimension index is 7. d: {d}").into()}
		if let Some(r)=self.rank(){
			if d>=r{return format!("rank {r} is too low to cat along dimension {d}").into()}
		}
		let v=if let Value::Multi(v)=self{v}else{return self};
		let (shape,variant)=if let Some(t)=v.iter().filter(|x|!(x.is_incompatible()||x.is_multi())).next(){(t.shape().to_x8(Alignment::Left),mem::discriminant(t))}else{return Value::Multi(v.into_iter().map(|x|x.cat(d)).collect())};
		if !v.iter().all(|x|mem::discriminant(x)==variant&&x.shape().to_x8(Alignment::Left).iter().enumerate().zip(shape.iter()).all(|((n,s),z)|d==n||s==z)){return "incompatible shapes for concatenation".into()}
		let mut v=v.into_iter();

		match v.next().unwrap(){B1(x0)=>B1(Tensor::cat(once(B1(x0)).chain(v).map(|x|x.unwrap_b1()).collect(),d)),B2(x0)=>B2(Tensor::cat(once(B2(x0)).chain(v).map(|x|x.unwrap_b2()).collect(),d)),B3(x0)=>B3(Tensor::cat(once(B3(x0)).chain(v).map(|x|x.unwrap_b3()).collect(),d)),B4(x0)=>B4(Tensor::cat(once(B4(x0)).chain(v).map(|x|x.unwrap_b4()).collect(),d)),B5(x0)=>B5(Tensor::cat(once(B5(x0)).chain(v).map(|x|x.unwrap_b5()).collect(),d)),B6(x0)=>B6(Tensor::cat(once(B6(x0)).chain(v).map(|x|x.unwrap_b6()).collect(),d)),B7(x0)=>B7(Tensor::cat(once(B7(x0)).chain(v).map(|x|x.unwrap_b7()).collect(),d)),B8(x0)=>B8(Tensor::cat(once(B8(x0)).chain(v).map(|x|x.unwrap_b8()).collect(),d)),F1(x0)=>F1(Tensor::cat(once(F1(x0)).chain(v).map(|x|x.unwrap_f1()).collect(),d)),F2(x0)=>F2(Tensor::cat(once(F2(x0)).chain(v).map(|x|x.unwrap_f2()).collect(),d)),F3(x0)=>F3(Tensor::cat(once(F3(x0)).chain(v).map(|x|x.unwrap_f3()).collect(),d)),F4(x0)=>F4(Tensor::cat(once(F4(x0)).chain(v).map(|x|x.unwrap_f4()).collect(),d)),F5(x0)=>F5(Tensor::cat(once(F5(x0)).chain(v).map(|x|x.unwrap_f5()).collect(),d)),F6(x0)=>F6(Tensor::cat(once(F6(x0)).chain(v).map(|x|x.unwrap_f6()).collect(),d)),F7(x0)=>F7(Tensor::cat(once(F7(x0)).chain(v).map(|x|x.unwrap_f7()).collect(),d)),F8(x0)=>F8(Tensor::cat(once(F8(x0)).chain(v).map(|x|x.unwrap_f8()).collect(),d)),I1(x0)=>I1(Tensor::cat(once(I1(x0)).chain(v).map(|x|x.unwrap_i1()).collect(),d)),I2(x0)=>I2(Tensor::cat(once(I2(x0)).chain(v).map(|x|x.unwrap_i2()).collect(),d)),I3(x0)=>I3(Tensor::cat(once(I3(x0)).chain(v).map(|x|x.unwrap_i3()).collect(),d)),I4(x0)=>I4(Tensor::cat(once(I4(x0)).chain(v).map(|x|x.unwrap_i4()).collect(),d)),I5(x0)=>I5(Tensor::cat(once(I5(x0)).chain(v).map(|x|x.unwrap_i5()).collect(),d)),I6(x0)=>I6(Tensor::cat(once(I6(x0)).chain(v).map(|x|x.unwrap_i6()).collect(),d)),I7(x0)=>I7(Tensor::cat(once(I7(x0)).chain(v).map(|x|x.unwrap_i7()).collect(),d)),I8(x0)=>I8(Tensor::cat(once(I8(x0)).chain(v).map(|x|x.unwrap_i8()).collect(),d)),Value::Incompatible(_x0)=>panic!("cat internal error"),Value::Multi(_x0)=>panic!("cat internal error")}
	}
	/// casts to a float tensor if not one
	pub fn float(self)->Value<B>{
		match self{B1(x)=>F1(x.float()),B2(x)=>F2(x.float()),B3(x)=>F3(x.float()),B4(x)=>F4(x.float()),B5(x)=>F5(x.float()),B6(x)=>F6(x.float()),B7(x)=>F7(x.float()),B8(x)=>F8(x.float()),F1(x)=>F1(x),F2(x)=>F2(x),F3(x)=>F3(x),F4(x)=>F4(x),F5(x)=>F5(x),F6(x)=>F6(x),F7(x)=>F7(x),F8(x)=>F8(x),I1(x)=>F1(x.float()),I2(x)=>F2(x.float()),I3(x)=>F3(x.float()),I4(x)=>F4(x.float()),I5(x)=>F5(x.float()),I6(x)=>F6(x.float()),I7(x)=>F7(x.float()),I8(x)=>F8(x.float()),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(Value::float).collect())}
	}
	/// creates a new empty value
	pub fn empty()->Self{Self::Multi(Vec::new())}
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
	/// returns a shallow count the number of values directly within this one. 1 if not multi, otherwise the len of the vec inside.
	pub fn len(&self)->usize{
		if let Value::Multi(v)=self{v.len()}else{1}
	}
	/// recursively counts the number of tensors within this value, including multi tensors within multi tensors
	pub fn len_recursive(&self)->usize{
		if let Value::Multi(v)=self{v.iter().map(Value::len_recursive).sum()}else{1}
	}
	/// casts to a multiple tensor if not one
	pub fn multi(self)->Self{
		if let Value::Multi(v)=self{v.into()}else{vec![self].into()}
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
	/// returns a value containing the elements selected from the given ranges. If this is a multi tensor the slice will be applied to each sub tensor
	pub fn slice<A:AsRef<[R]>,R:RangeBounds<usize>>(self,ranges:A)->Self{
		let ranges=ranges.as_ref();
		let len=ranges.len();
		if let Value::Incompatible(x)=self{return x.into()}
		let rank=self.rank().unwrap_or(len);
		let shape=self.shape();

		let mut normalizedranges=[0;8].map(|_|0..0);
		for ((d,n),r) in shape.clone().to_x8(Alignment::Left).into_iter().zip(normalizedranges.iter_mut()).zip(ranges){
			n.start=match r.start_bound(){Excluded(&x)=>x+1,Included(&x)=>x,Unbounded=>0};
			n.end=match r.end_bound(){Excluded(&x)=>x,Included(&x)=>x-1,Unbounded=>d};
		}
		if len>rank{return format!("Length of ranges argument must be less than the the value's rank. len: {len} ranges: {normalizedranges:?} rank: {rank} shape: {shape:?}").into()}
		for (d,n) in shape.clone().to_x8(Alignment::Left).into_iter().zip(normalizedranges.iter()).take(len){
			if n.start>=n.end{return format!("Empty or reverse ranges are currently not supported. ranges: {normalizedranges:?}").into()}
			if d<n.end{return format!("Cannot index beyond the end of a dimension. ranges: {normalizedranges:?} shape: {shape:?}").into()}
		}
		let ranges=&normalizedranges[..len];

		match self{B1(x)=>B1(slice_slice(ranges,x)),B2(x)=>B2(slice_slice(ranges,x)),B3(x)=>B3(slice_slice(ranges,x)),B4(x)=>B4(slice_slice(ranges,x)),B5(x)=>B5(slice_slice(ranges,x)),B6(x)=>B6(slice_slice(ranges,x)),B7(x)=>B7(slice_slice(ranges,x)),B8(x)=>B8(slice_slice(ranges,x)),F1(x)=>F1(slice_slice(ranges,x)),F2(x)=>F2(slice_slice(ranges,x)),F3(x)=>F3(slice_slice(ranges,x)),F4(x)=>F4(slice_slice(ranges,x)),F5(x)=>F5(slice_slice(ranges,x)),F6(x)=>F6(slice_slice(ranges,x)),F7(x)=>F7(slice_slice(ranges,x)),F8(x)=>F8(slice_slice(ranges,x)),I1(x)=>I1(slice_slice(ranges,x)),I2(x)=>I2(slice_slice(ranges,x)),I3(x)=>I3(slice_slice(ranges,x)),I4(x)=>I4(slice_slice(ranges,x)),I5(x)=>I5(slice_slice(ranges,x)),I6(x)=>I6(slice_slice(ranges,x)),I7(x)=>I7(slice_slice(ranges,x)),I8(x)=>I8(slice_slice(ranges,x)),Value::Incompatible(x)=>x.into(),Value::Multi(x)=>Value::Multi(x.into_iter().map(|x|x.slice(ranges)).collect())}
	}
	/// stacks the multi tensor, inserting a dimension at d. for singular tensors this has an unsqueezing effect
	pub fn stack(self,d:usize)->Self{self.unsqueeze_dim(d).cat(d)}
	/// attempts to unwrap the inner B1 value
	pub fn try_b1(self)->Result<Tensor<B,1,Bool>,Self>{
		if let B1(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B2 value
	pub fn try_b2(self)->Result<Tensor<B,2,Bool>,Self>{
		if let B2(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B3 value
	pub fn try_b3(self)->Result<Tensor<B,3,Bool>,Self>{
		if let B3(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B4 value
	pub fn try_b4(self)->Result<Tensor<B,4,Bool>,Self>{
		if let B4(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B5 value
	pub fn try_b5(self)->Result<Tensor<B,5,Bool>,Self>{
		if let B5(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B6 value
	pub fn try_b6(self)->Result<Tensor<B,6,Bool>,Self>{
		if let B6(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B7 value
	pub fn try_b7(self)->Result<Tensor<B,7,Bool>,Self>{
		if let B7(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner B8 value
	pub fn try_b8(self)->Result<Tensor<B,8,Bool>,Self>{
		if let B8(x)=self{Ok(x)}else{Err(self)}
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
	/// attempts to unwrap the inner incompatible value
	pub fn try_incompatible(self)->Result<String,Self>{
		if let Value::Incompatible(x)=self{Ok(x)}else{Err(self)}
	}
	/// attempts to unwrap the inner multi value
	pub fn try_multi(self)->Result<Vec<Value<B>>,Self>{
		if let Value::Multi(v)=self{Ok(v)}else{Err(self)}
	}
	/// inserts a dimension of size 1 at position d
	pub fn unsqueeze_dim(self,d:usize)->Self{
		match self{B1(x)=>B2(x.unsqueeze_dim(d)),B2(x)=>B3(x.unsqueeze_dim(d)),B3(x)=>B4(x.unsqueeze_dim(d)),B4(x)=>B5(x.unsqueeze_dim(d)),B5(x)=>B6(x.unsqueeze_dim(d)),B6(x)=>B7(x.unsqueeze_dim(d)),B7(x)=>B8(x.unsqueeze_dim(d)),B8(_x)=>"currently cannot increase number of tensor dimensions above 8".into(),F1(x)=>F2(x.unsqueeze_dim(d)),F2(x)=>F3(x.unsqueeze_dim(d)),F3(x)=>F4(x.unsqueeze_dim(d)),F4(x)=>F5(x.unsqueeze_dim(d)),F5(x)=>F6(x.unsqueeze_dim(d)),F6(x)=>F7(x.unsqueeze_dim(d)),F7(x)=>F8(x.unsqueeze_dim(d)),F8(_x)=>"currently cannot increase number of tensor dimensions above 8".into(),I1(x)=>I2(x.unsqueeze_dim(d)),I2(x)=>I3(x.unsqueeze_dim(d)),I3(x)=>I4(x.unsqueeze_dim(d)),I4(x)=>I5(x.unsqueeze_dim(d)),I5(x)=>I6(x.unsqueeze_dim(d)),I6(x)=>I7(x.unsqueeze_dim(d)),I7(x)=>I8(x.unsqueeze_dim(d)),I8(_x)=>"currently cannot increase number of tensor dimensions above 8".into(),Value::Incompatible(e)=>e.into(),Value::Multi(v)=>Value::Multi(v.into_iter().map(|x|x.unsqueeze_dim(d)).collect())}
	}
	#[track_caller]
	/// attempts to unwrap the inner b1 value
	pub fn unwrap_b1(self)->Tensor<B,1,Bool>{self.try_b1().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner b2 value
	pub fn unwrap_b2(self)->Tensor<B,2,Bool>{self.try_b2().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner b3 value
	pub fn unwrap_b3(self)->Tensor<B,3,Bool>{self.try_b3().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner b4 value
	pub fn unwrap_b4(self)->Tensor<B,4,Bool>{self.try_b4().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner b5 value
	pub fn unwrap_b5(self)->Tensor<B,5,Bool>{self.try_b5().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner b6 value
	pub fn unwrap_b6(self)->Tensor<B,6,Bool>{self.try_b6().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner b7 value
	pub fn unwrap_b7(self)->Tensor<B,7,Bool>{self.try_b7().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner b8 value
	pub fn unwrap_b8(self)->Tensor<B,8,Bool>{self.try_b8().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner f1 value
	pub fn unwrap_f1(self)->Tensor<B,1>{self.try_f1().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner f2 value
	pub fn unwrap_f2(self)->Tensor<B,2>{self.try_f2().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner f3 value
	pub fn unwrap_f3(self)->Tensor<B,3>{self.try_f3().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner f4 value
	pub fn unwrap_f4(self)->Tensor<B,4>{self.try_f4().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner f5 value
	pub fn unwrap_f5(self)->Tensor<B,5>{self.try_f5().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner f6 value
	pub fn unwrap_f6(self)->Tensor<B,6>{self.try_f6().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner f7 value
	pub fn unwrap_f7(self)->Tensor<B,7>{self.try_f7().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner f8 value
	pub fn unwrap_f8(self)->Tensor<B,8>{self.try_f8().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner i1 value
	pub fn unwrap_i1(self)->Tensor<B,1,Int>{self.try_i1().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner i2 value
	pub fn unwrap_i2(self)->Tensor<B,2,Int>{self.try_i2().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner i3 value
	pub fn unwrap_i3(self)->Tensor<B,3,Int>{self.try_i3().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner i4 value
	pub fn unwrap_i4(self)->Tensor<B,4,Int>{self.try_i4().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner i5 value
	pub fn unwrap_i5(self)->Tensor<B,5,Int>{self.try_i5().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner i6 value
	pub fn unwrap_i6(self)->Tensor<B,6,Int>{self.try_i6().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner i7 value
	pub fn unwrap_i7(self)->Tensor<B,7,Int>{self.try_i7().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner i8 value
	pub fn unwrap_i8(self)->Tensor<B,8,Int>{self.try_i8().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner incompatible value
	pub fn unwrap_incompatible(self)->String{self.try_incompatible().unwrap()}
	#[track_caller]
	/// attempts to unwrap the inner multi value
	pub fn unwrap_multi(self)->Vec<Value<B>>{self.try_multi().unwrap()}
}
#[derive(Clone,Debug)]// TODO eq that doesn't include the payload of incompatible
/// tensor shapes for Value
pub enum Shape{Incompatible(String),Multi(usize),Recursive(Vec<Shape>),X1([usize;1]),X2([usize;2]),X3([usize;3]),X4([usize;4]),X5([usize;5]),X6([usize;6]),X7([usize;7]),X8([usize;8])}
#[derive(Clone,Debug)]//TODO implement module for this
/// enumerates burn tensors up to 8 dimensions, along with a variant to represent operation compatibility errors, and a variant for multiple tensors. An empty multi variant can be used to represent a lack of data. Once a the depth of a multi variant is enough for an operation to take full effect, further nesting should result in the same as applying separately
pub enum Value<B:Backend>{B1(Tensor<B,1,Bool>),B2(Tensor<B,2,Bool>),B3(Tensor<B,3,Bool>),B4(Tensor<B,4,Bool>),B5(Tensor<B,5,Bool>),B6(Tensor<B,6,Bool>),B7(Tensor<B,7,Bool>),B8(Tensor<B,8,Bool>),F1(Tensor<B,1,Float>),F2(Tensor<B,2,Float>),F3(Tensor<B,3,Float>),F4(Tensor<B,4,Float>),F5(Tensor<B,5,Float>),F6(Tensor<B,6,Float>),F7(Tensor<B,7,Float>),F8(Tensor<B,8,Float>),I1(Tensor<B,1,Int>),I2(Tensor<B,2,Int>),I3(Tensor<B,3,Int>),I4(Tensor<B,4,Int>),I5(Tensor<B,5,Int>),I6(Tensor<B,6,Int>),I7(Tensor<B,7,Int>),I8(Tensor<B,8,Int>),Incompatible(String),Multi(Vec<Self>)}
#[derive(Clone,Debug)]
/// general loss output for being converted into other loss outputs
pub struct LossOutput<B:Backend>{loss:Value<B>,output:Value<B>,target:Value<B>}
use Bound::{Excluded,Included,Unbounded};
use Shape::{X1,X2,X3,X4,X5,X6,X7,X8};
use Value::{B1,B2,B3,B4,B5,B6,B7,B8,F1,F2,F3,F4,F5,F6,F7,F8,I1,I2,I3,I4,I5,I6,I7,I8};
use burn::{
	prelude::{Backend,Bool,Float,Int,Tensor,TensorData},
	nn::{
		Dropout,Embedding,LayerNorm,Linear,Relu,Tanh,loss::{CrossEntropyLoss,MseLoss}
	},
	tensor::{
		BasicOps,TensorKind,activation::{log_softmax,softmax},cast::ToElement
	}
};
use crate::{
	ai::{AI,AccQ,Alignment,Cat,CrossEntropy,Decompose,Mean,Op,SoftChoose,SoftEntropy,SquaredError,TruncateToMatch,WhichDims,self},graph::Merge
};
use rand::random;
use std::{
	iter::{FromIterator,once},mem,ops::{Bound,RangeBounds,Range},slice::{Iter as SliceIter,self},vec::IntoIter as VecIntoIter
};
