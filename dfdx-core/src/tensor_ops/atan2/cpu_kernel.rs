use crate::tensor_ops::cpu_kernels::{BinaryDerivative};
use num_traits::Float;

impl<F: Float> BinaryDerivative<F> for super::BinaryAtan2KernelOp {
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, &x: &F, &y: &F) -> F {
        x.atan2(y)
    }
    #[inline(always)]
    fn dfdx(&self, x: &F, y: &F) -> F {
        -(*x)/((*x)*(*x) + (*y)*(*y))
    }

    #[inline(always)]
    fn dfdy(&self, x: &F, y: &F) -> F {
        *y/((*x)*(*x) + (*y)*(*y))
    }
}