/// reads data from a vector
fn read_serial<D:DeserializeOwned>(v:&mut &[u8])->Result<D,DecodeError>{
	let (Compat(d),size)=decode_from_slice(*v,standard_serial_configuration())?;
	*v=&v[size..];
	Ok(d)
}
/// writes the data to a vector
fn write_serial<S:Serialize>(data:&S,v:&mut Vec<u8>)->Result<(),EncodeError>{
	impl<'a> Writer for VecWriter<'a>{
		fn write(&mut self,bytes:&[u8])->Result<(),EncodeError>{Ok(self.0.extend_from_slice(bytes))}
	}
	struct VecWriter<'a>(&'a mut Vec<u8>);
	encode_into_writer(Compat(data),VecWriter(v),standard_serial_configuration())
}
/// reads a configuration from bytes, shortening the slice after
pub fn read_config<C:Config>(v:&mut &[u8])->Result<C,Box<dyn Error>>{Ok(read_serial(v)?)}
/// reads a module from byte
pub fn read_module<B:Backend,F:FnOnce()->M,M:Module<B>>(initialize:F,v:&mut &[u8])->Result<M,Box<dyn Error>> where M::Record:DeserializeOwned{Ok(initialize().load_record(read_serial(v)?))}
/// writes the configuration as bytes
pub fn write_config<C:Config>(config:&C,v:&mut Vec<u8>)->Result<(),Box<dyn Error>>{Ok(write_serial(config,v)?)}
/// writes the module as bytes
pub fn write_module<B:Backend,M:Module<B>>(module:&M,v:&mut Vec<u8>)->Result<(),Box<dyn Error>> where M::Record:Serialize{Ok(write_serial(&module.clone().into_record(),v)?)}
use {
	bincode::{
		config::standard as standard_serial_configuration,enc::write::Writer,encode_into_writer,decode_from_slice,error::{DecodeError,EncodeError},serde::Compat
	},
	burn::prelude::*,
	serde::{Serialize,de::DeserializeOwned},
	std::error::Error
};
