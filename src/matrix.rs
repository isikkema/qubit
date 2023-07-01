use std::ops::Mul;

use crate::tensor_mul::TensorMul;

#[derive(Debug, PartialEq)]
pub struct Matrix<T, const N: usize, const M: usize>(pub [[T; M]; N]);

impl<T, U, const N: usize, const M: usize, const X: usize, const Y: usize>
    TensorMul<Matrix<U, X, Y>> for Matrix<T, N, M>
where
    T: Mul<U>,
    [(); { N * X }]:,
    [(); { M * Y }]:,
{
    type Output = Matrix<<T as Mul<U>>::Output, { N * X }, { M * Y }>;

    fn tensor_mul(self, rhs: Matrix<U, X, Y>) -> Self::Output {
        
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis() {
        let a = Matrix([[1, 0], [0, 1]]);
        let b = Matrix([[1, 0], [0, 1]]);

        let result = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);

        assert_eq!(a.tensor_mul(b), result);
    }

    #[test]
    fn test_simple() {
        let a = Matrix([[1], [2]]);
        let b = Matrix([[3], [4]]);

        let result = Matrix([[3], [4], [6], [8]]);

        assert_eq!(a.tensor_mul(b), result);
    }
}
