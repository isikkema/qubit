use crate::tensor::Tensor;

pub trait Transposable {
    type Output: Clone + Transposable;

    #[allow(non_snake_case)]
    fn T(&self) -> Self::Output;
}

impl Transposable for f64 {
    type Output = f64;

    fn T(&self) -> Self::Output {
        self.clone()
    }
}

impl<T, const N: usize, const O: usize> Transposable for Tensor<T, N, O>
where
    T: Clone + Transposable,
{
    type Output = Tensor<<T as Transposable>::Output, N, O>;

    fn T(&self) -> Self::Output {
        // TODO: This is wrong
        let t_values: [<T as Transposable>::Output; N] = self.0.clone().map(|it| it.T());

        Tensor::<<T as Transposable>::Output, N, O>(t_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tensor_n3_r2_transpose() {
        let t1: Tensor<Tensor<f64, 3, 2>, 2, 1> =
            Tensor([Tensor([1.0, 2.0, 3.0]), Tensor([4.0, 5.0, 6.0])]);

        let t2: Tensor<Tensor<f64, 2, 2>, 3, 1> =
            Tensor([Tensor([1.0, 4.0]), Tensor([2.0, 5.0]), Tensor([3.0, 6.0])]);

        assert_eq!(t1.T(), t2);
    }
}
