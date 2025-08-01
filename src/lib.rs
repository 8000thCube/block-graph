#![recursion_limit = "256"]
mod ai;
mod graph;
/// builtin ai functions
pub mod builtin;
/// provides burn specific functionality
pub mod burn;
/// provides additional op traits
pub mod ops;
pub use {
	ai::{AI,Decompose,Inner,IntoSequence,Op,UnwrapInner},graph::{Graph,Label,Merge,Unvec}
};
