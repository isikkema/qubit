use crate::tensor::Tensor;

pub trait InnerMul<Rhs = Self> {
    type Output;

    fn inner_mul(&self, rhs: &Rhs) -> Self::Output;
}

impl InnerMul for f64 {
    type Output = f64;

    fn inner_mul(&self, rhs: &Self) -> Self::Output {
        self * rhs
    }
}

impl<T, const N: usize> InnerMul<Tensor<T, N>> for f64
where
    f64: InnerMul<T>,
{
    type Output = Tensor<<f64 as InnerMul<T>>::Output, N>;

    fn inner_mul(&self, rhs: &Tensor<T, N>) -> Self::Output {
        Tensor(rhs.0.each_ref().map(|n| self.inner_mul(n)))
    }
}

impl<T, U, const N: usize> InnerMul<Tensor<U, N>> for Tensor<T, N>
where
    T: InnerMul<U>,
{
    type Output = Tensor<<T as InnerMul<U>>::Output, N>;

    fn inner_mul(&self, rhs: &Tensor<U, N>) -> Self::Output {
        let mut idx = 0;
        Tensor(self.0.each_ref().map(|t| {
            let rv = t.inner_mul(&rhs[idx]);
            idx += 1;

            rv
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3_inner_mul_3() {
        let actual = Tensor([1.0, 2.0, 3.0]).inner_mul(&Tensor([3.0, 4.0, 5.0]));

        let expected = Tensor([3.0, 8.0, 15.0]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_2x3_inner_mul_2x3() {
        let actual =
            Tensor([Tensor([1.0, 2.0, 3.0]), Tensor([4.0, 5.0, 6.0])]).inner_mul(&Tensor([
                Tensor([7.0, 8.0, 9.0]),
                Tensor([10.0, 11.0, 12.0]),
            ]));

        let expected = Tensor([Tensor([7.0, 16.0, 27.0]), Tensor([40.0, 55.0, 72.0])]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_0_inner_mul_0() {
        let actual = Tensor::<f64, _>([]).inner_mul(&Tensor::<f64, _>([]));

        let expected = Tensor([]);

        assert_eq!(expected, actual);
    }
}
