use crate::tensor::Tensor;

pub trait TensorMul<Rhs = Self> {
    type Output;

    fn tensor_mul(self, rhs: Rhs) -> Self::Output;
}

impl TensorMul for f64 {
    type Output = f64;

    fn tensor_mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
}

impl TensorMul for &f64 {
    type Output = f64;

    fn tensor_mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
}

impl<T, const N: usize, const O: usize> TensorMul<Tensor<T, N, O>> for f64
where
    f64: TensorMul<T, Output = T>,
{
    type Output = Tensor<<f64 as TensorMul<T>>::Output, N, O>;

    fn tensor_mul(self, rhs: Tensor<T, N, O>) -> Self::Output {
        Tensor(rhs.0.map(|v| self.tensor_mul(v)))
    }
}

impl<T, const N: usize, const O: usize> TensorMul<f64> for Tensor<T, N, O>
where
    T: TensorMul<f64, Output = T>,
{
    type Output = Tensor<<T as TensorMul<f64>>::Output, N, O>;

    fn tensor_mul(self, rhs: f64) -> Self::Output {
        Tensor(self.0.map(|v| v.tensor_mul(rhs)))
    }
}

impl<T, U, const N: usize, const M: usize, const O1: usize, const O2: usize>
    TensorMul<Tensor<U, M, O2>> for Tensor<T, N, O1>
where
    T: TensorMul<Tensor<U, M, O2>>,
    U: Clone,
    [(); O1 + O2]:,
{
    type Output = Tensor<<T as TensorMul<Tensor<U, M, O2>>>::Output, N, { O1 + O2 }>;

    fn tensor_mul(self, rhs: Tensor<U, M, O2>) -> Self::Output {
        Tensor(self.0.map(|v| v.tensor_mul(rhs.clone())))
    }
}
