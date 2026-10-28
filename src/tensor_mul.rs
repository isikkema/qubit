use std::gca;

use crate::tensor::Tensor;

pub trait TensorMul<Rhs> {
    type Output;

    fn tensor_mul(&self, rhs: &Rhs) -> Self::Output;
}

impl TensorMul<f64> for f64 {
    type Output = f64;

    fn tensor_mul(&self, rhs: &Self) -> Self::Output {
        self * rhs
    }
}

pub const LEN<const N: usize, const M: usize>: usize = const { N * M };

impl<T, U, const N: usize, const M: usize> TensorMul<Tensor<U, M>> for Tensor<T, N>
where
    T: TensorMul<U>,
{
    type Output = Tensor<<T as TensorMul<U>>::Output, gca!(LEN::<N, M>)>;

    fn tensor_mul(&self, rhs: &Tensor<U, M>) -> Self::Output {
        Tensor(std::array::from_fn(|idx| {
            self.0[idx / M].tensor_mul(&rhs.0[idx % M])
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2_tensor_mul_3() {
        let actual = Tensor([1.0, 2.0]).tensor_mul(&Tensor([3.0, 4.0, 5.0]));

        let expected = Tensor([3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_3_tensor_mul_2() {
        let actual = Tensor([1.0, 2.0, 3.0]).tensor_mul(&Tensor([4.0, 5.0]));

        let expected = Tensor([4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_2x2_tensor_mul_2x2() {
        let actual = Tensor([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])])
            .tensor_mul(&Tensor([Tensor([5.0, 6.0]), Tensor([7.0, 8.0])]));

        let expected = Tensor([
            Tensor([5.0, 6.0, 10.0, 12.0]),
            Tensor([7.0, 8.0, 14.0, 16.0]),
            Tensor([15.0, 18.0, 20.0, 24.0]),
            Tensor([21.0, 24.0, 28.0, 32.0]),
        ]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_2x2_tensor_mul_2x1() {
        let actual = Tensor([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])])
            .tensor_mul(&Tensor([Tensor([5.0]), Tensor([6.0])]));

        let expected = Tensor([
            Tensor([5.0, 10.0]),
            Tensor([6.0, 12.0]),
            Tensor([15.0, 20.0]),
            Tensor([18.0, 24.0]),
        ]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_2x1_tensor_mul_2x2() {
        let actual = Tensor([Tensor([1.0]), Tensor([2.0])])
            .tensor_mul(&Tensor([Tensor([3.0, 4.0]), Tensor([5.0, 6.0])]));

        let expected = Tensor([
            Tensor([3.0, 4.0]),
            Tensor([5.0, 6.0]),
            Tensor([6.0, 8.0]),
            Tensor([10.0, 12.0]),
        ]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_1x2x2_tensor_mul_2x1x2() {
        let actual =
            Tensor([Tensor([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])])]).tensor_mul(&Tensor([
                Tensor([Tensor([5.0, 6.0])]),
                Tensor([Tensor([7.0, 8.0])]),
            ]));

        let expected = Tensor([
            Tensor([
                Tensor([5.0, 6.0, 10.0, 12.0]),
                Tensor([15.0, 18.0, 20.0, 24.0]),
            ]),
            Tensor([
                Tensor([7.0, 8.0, 14.0, 16.0]),
                Tensor([21.0, 24.0, 28.0, 32.0]),
            ]),
        ]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_2x1x2_tensor_mul_1x2x1() {
        let actual = Tensor([Tensor([Tensor([1.0, 2.0])]), Tensor([Tensor([3.0, 4.0])])])
            .tensor_mul(&Tensor([Tensor([Tensor([5.0]), Tensor([6.0])])]));

        let expected = Tensor([
            Tensor([Tensor([5.0, 10.0]), Tensor([6.0, 12.0])]),
            Tensor([Tensor([15.0, 20.0]), Tensor([18.0, 24.0])]),
        ]);

        assert_eq!(expected, actual);
    }
}
