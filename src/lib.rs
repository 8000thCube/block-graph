#![recursion_limit = "256"]
mod ai;
mod graph;
pub mod builtin;
pub mod burn;
pub use {
	ai::{AI,Decompose,IntoSequence,Op,UnwrapInner},graph::{Graph,Label,Merge,Unvec}
};
