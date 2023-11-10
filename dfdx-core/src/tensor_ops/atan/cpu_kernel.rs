use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float + std::ops::Mul<Output=F>> UnaryDerivative<F> for super::AtanKernelOp {
    const DF_USES_FX: bool = false;
    const HAS_CONST_DF: bool = false;
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.atan()
    }
    #[inline(always)]
    fn df(&self, x: &F) -> F {
        let one = F::from(1.0).unwrap();
        one / (one + (*x)*(*x))
    }
}
