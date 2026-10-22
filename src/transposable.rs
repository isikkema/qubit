use crate::tensor::Tensor;

pub trait Transposable {
    type Output: Clone + Transposable;

    #[allow(non_snake_case)]
    fn T(&self) -> Self::Output;
}

impl Transposable for f64 {
    type Output = f64;

    fn T(&self) -> Self::Output {
        *self
    }
}

impl<const N: usize> Transposable for Tensor<f64, N> {
    type Output = Tensor<Tensor<f64, 1>, N>;

    fn T(&self) -> Self::Output {
        Tensor((*self.as_array()).map(|t| Tensor::<f64, 1>([t])))
    }
}

impl<T, const N: usize, const M: usize> Transposable for Tensor<Tensor<T, N>, M>
where
    T: Transposable,
{
    type Output = Tensor<Tensor<<T as Transposable>::Output, M>, N>;

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
        let t1: Tensor<Tensor<f64, 3>, 2> =
            Tensor([Tensor([1.0, 2.0, 3.0]), Tensor([4.0, 5.0, 6.0])]);

        let t2: Tensor<Tensor<f64, 2>, 3> =
            Tensor([Tensor([1.0, 4.0]), Tensor([2.0, 5.0]), Tensor([3.0, 6.0])]);

        assert_eq!(t1.T(), t2);
    }
}
