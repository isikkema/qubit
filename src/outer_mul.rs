use crate::tensor::Tensor;

pub trait OuterMul<Rhs = Self> {
    type Output;

    fn outer_mul(&self, rhs: &Rhs) -> Self::Output;
}

impl OuterMul for f64 {
    type Output = f64;

    fn outer_mul(&self, rhs: &Self) -> Self::Output {
        self * rhs
    }
}

impl<T, const N: usize> OuterMul<Tensor<T, N>> for f64
where
    f64: OuterMul<T>,
{
    type Output = Tensor<<f64 as OuterMul<T>>::Output, N>;

    fn outer_mul(&self, rhs: &Tensor<T, N>) -> Self::Output {
        Tensor(rhs.0.each_ref().map(|n| self.outer_mul(n)))
    }
}

impl<T, U, const N: usize, const M: usize> OuterMul<Tensor<U, M>> for Tensor<T, N>
where
    T: OuterMul<Tensor<U, M>>,
{
    type Output = Tensor<<T as OuterMul<Tensor<U, M>>>::Output, N>;

    fn outer_mul(&self, rhs: &Tensor<U, M>) -> Self::Output {
        Tensor(self.0.each_ref().map(|t| t.outer_mul(rhs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2_outer_mul_3() {
        let actual = Tensor([1.0, 2.0]).outer_mul(&Tensor([3.0, 4.0, 5.0]));

        let expected = Tensor([Tensor([3.0, 4.0, 5.0]), Tensor([6.0, 8.0, 10.0])]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_3_outer_mul_2() {
        let actual = Tensor([1.0, 2.0, 3.0]).outer_mul(&Tensor([3.0, 4.0]));

        let expected = Tensor([Tensor([3.0, 4.0]), Tensor([6.0, 8.0]), Tensor([9.0, 12.0])]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_2x3_outer_mul_2() {
        let actual = Tensor([Tensor([1.0, 2.0, 3.0]), Tensor([4.0, 5.0, 6.0])])
            .outer_mul(&Tensor([7.0, 8.0]));

        let expected = Tensor([
            Tensor([
                Tensor([7.0, 8.0]),
                Tensor([14.0, 16.0]),
                Tensor([21.0, 24.0]),
            ]),
            Tensor([
                Tensor([28.0, 32.0]),
                Tensor([35.0, 40.0]),
                Tensor([42.0, 48.0]),
            ]),
        ]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_2_outer_mul_2x3() {
        let actual = Tensor([1.0, 2.0])
            .outer_mul(&Tensor([Tensor([3.0, 4.0, 5.0]), Tensor([6.0, 7.0, 8.0])]));

        let expected = Tensor([
            Tensor([Tensor([3.0, 4.0, 5.0]), Tensor([6.0, 7.0, 8.0])]),
            Tensor([Tensor([6.0, 8.0, 10.0]), Tensor([12.0, 14.0, 16.0])]),
        ]);

        assert_eq!(expected, actual);
    }
}
