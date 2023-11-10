mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::*;
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BinaryAtan2KernelOp;


pub fn atan2<S: Shape, E: Dtype, D, T: Tape<E, D> + Merge<R>, R: Default>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, R>,
) -> Tensor<S, E, D, T>
where
    D: BinaryKernel<BinaryAtan2KernelOp, E>,
{
    lhs.try_atan2(rhs).unwrap()
}


// pub fn add<S: Shape, E: Dtype, D, T: Tape<E, D> + Merge<R>, R: Default>(
//     lhs: Tensor<S, E, D, T>,
//     rhs: Tensor<S, E, D, R>,
// ) -> Tensor<S, E, D, T>
// where
//     D: BinaryKernel<BinaryAddKernelOp, E>,
// {
//     lhs + rhs
// }

/// Fallible version of [std::ops::Add]. See [add]
pub trait TryAtan2<Rhs = Self> {
    type Output;

    fn try_atan2(self, rhs: Rhs) -> Result<Self::Output, Error>;
}

impl<S: Shape, E: Dtype, D, LhsTape: Tape<E, D>, R> TryAtan2<Tensor<S, E, D, R>>
    for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<BinaryAtan2KernelOp, E>,
    LhsTape: Merge<R>,
{
    type Output = Self;

    /// See [add]
    fn try_atan2(self, rhs: Tensor<S, E, D, R>) -> Result<Self, Error> {
        try_binary_op(BinaryAtan2KernelOp, self, rhs)
    }
}