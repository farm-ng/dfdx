mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct AtanKernelOp;

///
/// It's derivative is `1 / (1 + x^2)`
///
/// Examples:
/// ```rust
/// ```
pub fn atan<S: Shape, E: Dtype, D: UnaryKernel<AtanKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.atan()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<AtanKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [atan]
    pub fn atan(self) -> Self {
        self.try_atan().unwrap()
    }
    /// See [atan]
    pub fn try_atan(self) -> Result<Self, Error> {
        try_unary_op(AtanKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::storage_traits::AsArray;
    use crate::tests::*;
    use crate::{tensor::*, tensor_ops::*};

    #[test]
    fn test_atan() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([-2.0, -1.0]).to_dtype::<TestDtype>();

        let expected_dx_atanx = [0.2, 0.5];
        let mut a = [0.0, 0.0];
        for i in 0..2 {
            let r = x.clone().leaky_trace().atan();
            let g = r.select(dev.tensor(i)).backward();
            let rr = g.get(&x);
            a[i] = rr[[i]];
        }
        assert_close_to_literal!(dev.tensor(a), expected_dx_atanx);
    }
}
