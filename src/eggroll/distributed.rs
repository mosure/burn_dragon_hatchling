use burn::tensor::backend::Backend;
use burn::module::Module;

use super::model::{EggrollObjective, EggrollTrainer};

/// Lightweight wrapper to make it clear where distributed collectives will be wired.
///
/// This is intentionally minimal: once burn-collective wiring is added, replace the `()` handle
/// with a real collective client and use all-reduce to aggregate ES gradients across ranks so that
/// large populations can be sharded deterministically.
pub struct DistributedEggroll<M, B: Backend, Obj>
where
    M: Module<B> + Clone,
    Obj: EggrollObjective<M, B>,
{
    pub trainer: EggrollTrainer<M, B, Obj>,
    /// Placeholder for future collective client; kept as unit to avoid unused type noise.
    pub collective: (),
}

impl<M, B, Obj> DistributedEggroll<M, B, Obj>
where
    M: Module<B> + Clone,
    B: Backend,
    Obj: EggrollObjective<M, B>,
{
    pub fn new(trainer: EggrollTrainer<M, B, Obj>, collective: ()) -> Self {
        Self {
            trainer,
            collective,
        }
    }
}
