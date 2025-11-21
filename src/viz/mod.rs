#[cfg(feature = "viz")]
pub mod activity;
#[cfg(feature = "viz")]
pub mod collect;
#[cfg(feature = "viz")]
pub mod frame;
#[cfg(feature = "viz")]
mod imp;
#[cfg(not(feature = "viz"))]
mod noop;

#[cfg(feature = "viz")]
pub use imp::*;
#[cfg(not(feature = "viz"))]
pub use noop::*;
