#![recursion_limit = "256"]
mod ai;
/// graph stuff
mod graph;
/// builtin ai layers and value types
pub mod builtin;
/// ai layers and value types relating to the burn library
pub mod burn;
/// operator like traits for common tensor operations
pub mod ops;
pub use {
	ai::{AI,Decompose,Inner,IntoSequence,Op,UnwrapInner},graph::{ConnectionEditor,ConnectionInfo,Graph,Label,Merge,Unvec}
};
