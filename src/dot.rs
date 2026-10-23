use std::{iter::Sum, ops::Mul};

use crate::tensor::Tensor;

pub trait Dot {
    type Output;

    fn dot(&self, other: &Self) -> Self::Output;
}

impl<T, const N: usize> Dot for Tensor<T, N>
where
    for<'a> &'a T: Mul<Output = T>,
    T: Sum,
{
    type Output = T;

    fn dot(&self, other: &Self) -> Self::Output {
        let mut idx = 0;
        self.0
            .each_ref()
            .map(|a| {
                let rv = a * &other[idx];
                idx += 1;

                rv
            })
            .into_iter()
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3_dot_3() {
        assert_eq!(Tensor([1.0, 2.0, 3.0]).dot(&Tensor([3.0, 4.0, 5.0])), 26.0);
    }

    #[test]
    fn test_0_dot_0() {
        assert_eq!(Tensor::<f64, _>([]).dot(&Tensor([])), -0.0);
    }
}
