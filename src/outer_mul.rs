use crate::tensor::Tensor;

pub trait OuterMul<Rhs = Self> {
    type Output;

    fn outer_mul(self, rhs: Rhs) -> Self::Output;
}

impl OuterMul for f64 {
    type Output = f64;

    fn outer_mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
}

impl OuterMul for &f64 {
    type Output = f64;

    fn outer_mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
}

impl<T, const N: usize, const O: usize> OuterMul<Tensor<T, N, O>> for f64
where
    f64: OuterMul<T, Output = T>,
{
    type Output = Tensor<<f64 as OuterMul<T>>::Output, N, O>;

    fn outer_mul(self, rhs: Tensor<T, N, O>) -> Self::Output {
        Tensor(rhs.0.map(|v| self.outer_mul(v)))
    }
}

impl<T, U, const N: usize, const M: usize, const O1: usize, const O2: usize>
    OuterMul<Tensor<U, M, O2>> for Tensor<T, N, O1>
where
    T: OuterMul<Tensor<U, M, O2>>,
    U: Clone,
    [(); O1 + O2]:,
{
    type Output = Tensor<<T as OuterMul<Tensor<U, M, O2>>>::Output, N, { O1 + O2 }>;

    fn outer_mul(self, rhs: Tensor<U, M, O2>) -> Self::Output {
        Tensor(self.0.map(|v| v.outer_mul(rhs.clone())))
    }
}

impl<T, const N: usize, const O: usize> OuterMul<f64> for Tensor<T, N, O>
where
    T: OuterMul<f64, Output = T>,
{
    type Output = Tensor<<T as OuterMul<f64>>::Output, N, O>;

    fn outer_mul(self, rhs: f64) -> Self::Output {
        Tensor(self.0.map(|v| v.outer_mul(rhs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_r2_basis_r2_basis() {
        let r2: Tensor<Tensor<f64, 2, 1>, 2, 2> = Tensor([
            Tensor([1.0, 0.0]),
            Tensor([0.0, 1.0]),
        ]);

        let r4: Tensor<Tensor<f64, 4, 1>, 4, 2> = Tensor([
            Tensor([1.0, 0.0, 0.0, 0.0]),
            Tensor([0.0, 1.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 1.0, 0.0]),
            Tensor([0.0, 0.0, 0.0, 1.0]),
        ]);

        println!("{:?}", r2.clone().outer_mul(r2));

        assert!(false);
    }

    #[test]
    fn test_f64_f64() {
        assert_eq!(2.0.outer_mul(3.0), 6.0);
    }

    #[test]
    fn test_f64_tensor_n1_r1() {
        assert_eq!(2.0.outer_mul(Tensor::<_, 1, 1>([1.0])), Tensor([2.0]));
    }

    #[test]
    fn test_tensor_n1_r1_f64() {
        assert_eq!(Tensor::<_, 1, 1>([1.0]).outer_mul(2.0), Tensor([2.0]));
    }

    #[test]
    fn test_f64_tensor_n2_r1() {
        assert_eq!(
            2.0.outer_mul(Tensor::<_, 2, 1>([1.0, 2.0])),
            Tensor([2.0, 4.0])
        );
    }

    #[test]
    fn test_f64_tensor_n3_r1() {
        assert_eq!(
            2.0.outer_mul(Tensor::<_, 3, 1>([1.0, 2.0, 3.0])),
            Tensor([2.0, 4.0, 6.0])
        );
    }

    #[test]
    fn test_f64_tensor_n3_r2() {
        let t1 = 2.0;

        let t2: Tensor<Tensor<f64, 3, 1>, 3, 2> = Tensor([
            Tensor::<_, 3, 1>([1.0, 2.0, 3.0]),
            Tensor::<_, 3, 1>([4.0, 5.0, 6.0]),
            Tensor::<_, 3, 1>([7.0, 8.0, 9.0]),
        ]);

        let result: Tensor<Tensor<f64, 3, 1>, 3, 2> = Tensor([
            Tensor::<_, 3, 1>([2.0, 4.0, 6.0]),
            Tensor::<_, 3, 1>([8.0, 10.0, 12.0]),
            Tensor::<_, 3, 1>([14.0, 16.0, 18.0]),
        ]);

        assert_eq!(t1.outer_mul(t2), result);
    }

    #[test]
    fn test_f64_tensor_n3_r3() {
        let t1 = 2.0;

        let t2: Tensor<Tensor<Tensor<f64, 3, 1>, 3, 2>, 3, 3> = Tensor([
            Tensor([
                Tensor::<_, 3, 1>([1.0, 2.0, 3.0]),
                Tensor::<_, 3, 1>([4.0, 5.0, 6.0]),
                Tensor::<_, 3, 1>([7.0, 8.0, 9.0]),
            ]),
            Tensor([
                Tensor::<_, 3, 1>([10.0, 11.0, 12.0]),
                Tensor::<_, 3, 1>([13.0, 14.0, 15.0]),
                Tensor::<_, 3, 1>([16.0, 17.0, 18.0]),
            ]),
            Tensor([
                Tensor::<_, 3, 1>([19.0, 20.0, 21.0]),
                Tensor::<_, 3, 1>([22.0, 23.0, 24.0]),
                Tensor::<_, 3, 1>([25.0, 26.0, 27.0]),
            ]),
        ]);

        let result = Tensor([
            Tensor([
                Tensor::<_, 3, 1>([2.0, 4.0, 6.0]),
                Tensor::<_, 3, 1>([8.0, 10.0, 12.0]),
                Tensor::<_, 3, 1>([14.0, 16.0, 18.0]),
            ]),
            Tensor([
                Tensor::<_, 3, 1>([20.0, 22.0, 24.0]),
                Tensor::<_, 3, 1>([26.0, 28.0, 30.0]),
                Tensor::<_, 3, 1>([32.0, 34.0, 36.0]),
            ]),
            Tensor([
                Tensor::<_, 3, 1>([38.0, 40.0, 42.0]),
                Tensor::<_, 3, 1>([44.0, 46.0, 48.0]),
                Tensor::<_, 3, 1>([50.0, 52.0, 54.0]),
            ]),
        ]);

        assert_eq!(t1.outer_mul(t2), result);
    }

    #[test]
    fn test_tensor_n3_r3_f64() {
        let t2 = 2.0;

        let t1: Tensor<Tensor<Tensor<f64, 3, 1>, 3, 2>, 3, 3> = Tensor([
            Tensor([
                Tensor::<_, 3, 1>([1.0, 2.0, 3.0]),
                Tensor::<_, 3, 1>([4.0, 5.0, 6.0]),
                Tensor::<_, 3, 1>([7.0, 8.0, 9.0]),
            ]),
            Tensor([
                Tensor::<_, 3, 1>([10.0, 11.0, 12.0]),
                Tensor::<_, 3, 1>([13.0, 14.0, 15.0]),
                Tensor::<_, 3, 1>([16.0, 17.0, 18.0]),
            ]),
            Tensor([
                Tensor::<_, 3, 1>([19.0, 20.0, 21.0]),
                Tensor::<_, 3, 1>([22.0, 23.0, 24.0]),
                Tensor::<_, 3, 1>([25.0, 26.0, 27.0]),
            ]),
        ]);

        let result = Tensor([
            Tensor([
                Tensor::<_, 3, 1>([2.0, 4.0, 6.0]),
                Tensor::<_, 3, 1>([8.0, 10.0, 12.0]),
                Tensor::<_, 3, 1>([14.0, 16.0, 18.0]),
            ]),
            Tensor([
                Tensor::<_, 3, 1>([20.0, 22.0, 24.0]),
                Tensor::<_, 3, 1>([26.0, 28.0, 30.0]),
                Tensor::<_, 3, 1>([32.0, 34.0, 36.0]),
            ]),
            Tensor([
                Tensor::<_, 3, 1>([38.0, 40.0, 42.0]),
                Tensor::<_, 3, 1>([44.0, 46.0, 48.0]),
                Tensor::<_, 3, 1>([50.0, 52.0, 54.0]),
            ]),
        ]);

        assert_eq!(t1.outer_mul(t2), result);
    }

    #[test]
    fn test_tensor_n3_r1_tensor_n3_r1() {
        let t1: Tensor<f64, 3, 1> = Tensor([1.0, 2.0, 3.0]);

        let t2: Tensor<f64, 3, 1> = Tensor([4.0, 5.0, 6.0]);

        let result = Tensor([
            Tensor([4.0, 5.0, 6.0]),
            Tensor([8.0, 10.0, 12.0]),
            Tensor([12.0, 15.0, 18.0]),
        ]);

        assert_eq!(t1.outer_mul(t2), result);
    }

    #[test]
    fn test_tensor_n2_r1_tensor_n3_r2() {
        let t1: Tensor<f64, 2, 1> = Tensor([2.0, 3.0]);

        let t2: Tensor<Tensor<f64, 3, 1>, 3, 2> = Tensor([
            Tensor::<_, 3, 1>([1.0, 2.0, 3.0]),
            Tensor::<_, 3, 1>([4.0, 5.0, 6.0]),
            Tensor::<_, 3, 1>([7.0, 8.0, 9.0]),
        ]);

        let result = Tensor([
            Tensor([
                Tensor([2.0, 4.0, 6.0]),
                Tensor([8.0, 10.0, 12.0]),
                Tensor([14.0, 16.0, 18.0]),
            ]),
            Tensor([
                Tensor([3.0, 6.0, 9.0]),
                Tensor([12.0, 15.0, 18.0]),
                Tensor([21.0, 24.0, 27.0]),
            ]),
        ]);

        assert_eq!(t1.outer_mul(t2), result);
    }

    #[test]
    fn test_tensor_n3_r2_tensor_n2_r1() {
        let t1: Tensor<Tensor<f64, 3, 1>, 3, 2> = Tensor([
            Tensor::<_, 3, 1>([1.0, 2.0, 3.0]),
            Tensor::<_, 3, 1>([4.0, 5.0, 6.0]),
            Tensor::<_, 3, 1>([7.0, 8.0, 9.0]),
        ]);

        let t2: Tensor<f64, 2, 1> = Tensor([2.0, 3.0]);

        let result = Tensor([
            Tensor([Tensor([2.0, 3.0]), Tensor([4.0, 6.0]), Tensor([6.0, 9.0])]),
            Tensor([
                Tensor([8.0, 12.0]),
                Tensor([10.0, 15.0]),
                Tensor([12.0, 18.0]),
            ]),
            Tensor([
                Tensor([14.0, 21.0]),
                Tensor([16.0, 24.0]),
                Tensor([18.0, 27.0]),
            ]),
        ]);

        assert_eq!(t1.outer_mul(t2), result);
    }
}
