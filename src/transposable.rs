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

impl<T, const N: usize> Transposable for Tensor<T, N, 1>
where
    T: Transposable + Clone,
{
    type Output = Tensor<Tensor<<T as Transposable>::Output, 1, 1>, N, 2>;

    fn T(&self) -> Self::Output {
        Tensor(
            self.as_array()
                .clone()
                .map(|t| Tensor::<<T as Transposable>::Output, 1, 1>([t.T()])),
        )
    }
}

impl<T, const N: usize, const M: usize> Transposable for Tensor<Tensor<T, N, 1>, M, 2>
where
    T: Transposable,
{
    type Output = Tensor<Tensor<<T as Transposable>::Output, M, 1>, N, 2>;

    fn T(&self) -> Self::Output {
        let mut outer_idx = 0;

        Tensor([(); N].map(|_| {
            let mut inner_idx = 0;

            let outer_rv = [(); M].map(|_| {
                let inner_rv = self.as_array()[inner_idx].as_array()[outer_idx].T();

                inner_idx += 1;

                inner_rv
            });

            outer_idx += 1;

            Tensor(outer_rv)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_n3_r1_transpose() {
        let t1 = Tensor([1.0, 2.0, 3.0]);

        let t2 = Tensor([Tensor([1.0]), Tensor([2.0]), Tensor([3.0])]);

        assert_eq!(t1.T(), t2);
    }

    #[test]
    fn test_tensor_n3_r2_transpose() {
        let t1: Tensor<Tensor<f64, 3, 1>, 2, 2> =
            Tensor([Tensor([1.0, 2.0, 3.0]), Tensor([4.0, 5.0, 6.0])]);

        let t2: Tensor<Tensor<f64, 2, 1>, 3, 2> =
            Tensor([Tensor([1.0, 4.0]), Tensor([2.0, 5.0]), Tensor([3.0, 6.0])]);

        assert_eq!(t1.T(), t2);
    }
}
