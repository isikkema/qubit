use std::{array, iter::Sum, ops::Mul};

use crate::{tensor::Tensor, transposable::Transposable};

pub trait Dot<Rhs> {
    type Output;

    fn dot(&self, other: &Rhs) -> Self::Output;
}

impl<T, U, const N: usize> Dot<Tensor<U, N>> for Tensor<T, N>
where
    T: Copy + Mul<U>,
    U: Copy,
    <T as Mul<U>>::Output: Sum,
{
    type Output = <T as Mul<U>>::Output;

    fn dot(&self, rhs: &Tensor<U, N>) -> Self::Output {
        self.0.iter().zip(rhs.0.iter()).map(|(a, b)| *a * *b).sum()
    }
}

impl<T, U, const N: usize, const M: usize> Dot<Tensor<U, M>> for Tensor<Tensor<T, M>, N>
where
    T: Copy + Mul<U>,
    U: Copy,
    <T as Mul<U>>::Output: Sum,
{
    type Output = Tensor<<T as Mul<U>>::Output, N>;

    fn dot(&self, other: &Tensor<U, M>) -> Self::Output {
        Tensor(array::from_fn(|idx| self[idx].dot(other)))
    }
}

impl<T, U, const N: usize, const M: usize, const O: usize> Dot<Tensor<Tensor<U, O>, M>>
    for Tensor<Tensor<T, M>, N>
where
    T: Copy + Mul<U>,
    U: Copy + Transposable<Output = U>,
    <T as Mul<U>>::Output: Sum,
{
    type Output = Tensor<Tensor<<T as Mul<U>>::Output, O>, N>;

    fn dot(&self, other: &Tensor<Tensor<U, O>, M>) -> Self::Output {
        let rhs = other.T();

        Tensor(array::from_fn(|n| {
            Tensor(array::from_fn(|o| self[n].dot(&rhs[o])))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3_dot_3() {
        assert_eq!(Tensor([1.0, 2.0, 3.0]).dot(&Tensor([4.0, 5.0, 6.0])), 32.0);
    }

    #[test]
    fn test_0_dot_0() {
        assert_eq!(Tensor::<f64, _>([]).dot(&Tensor::<f64, _>([])), -0.0);
    }

    #[test]
    fn test_2x2_dot_2() {
        let actual = Tensor([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])]).dot(&Tensor([5.0, 6.0]));

        let expected = Tensor([17.0, 39.0]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_2x2_dot_2x2() {
        let actual = Tensor([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])])
            .dot(&Tensor([Tensor([5.0, 6.0]), Tensor([7.0, 8.0])]));

        let expected = Tensor([Tensor([19.0, 22.0]), Tensor([43.0, 50.0])]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_2x3_dot_3x2() {
        let actual = Tensor([Tensor([1.0, 2.0, 3.0]), Tensor([4.0, 5.0, 6.0])]).dot(&Tensor([
            Tensor([7.0, 8.0]),
            Tensor([9.0, 10.0]),
            Tensor([11.0, 12.0]),
        ]));

        let expected = Tensor([Tensor([58.0, 64.0]), Tensor([139.0, 154.0])]);

        assert_eq!(expected, actual);
    }
}
