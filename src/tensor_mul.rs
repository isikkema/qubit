pub trait TensorMul<RHS> {
    type Output;

    fn tensor_mul(self, rhs: RHS) -> Self::Output;
}
