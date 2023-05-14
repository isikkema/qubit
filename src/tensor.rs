use std::{
    iter::Sum,
    ops::{Add, Mul},
};

use crate::transposable::Transposable;

#[derive(Clone, PartialEq, Debug)]
pub struct Tensor<T, const N: usize, const O: usize>(pub [T; N]);
impl<T, const N: usize, const O: usize> Tensor<T, N, O> {
    pub fn as_array(&self) -> &[T; N] {
        &self.0
    }

    // This O may not be correct
    pub fn map<F, U>(self, f: F) -> Tensor<U, N, O>
    where
        F: FnMut(T) -> U,
    {
        Tensor(self.0.map(f))
    }
}

impl<T, const N: usize, const O: usize> Default for Tensor<T, N, O>
where
    T: Default,
{
    fn default() -> Self {
        Tensor([(); N].map(|_| T::default()))
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

impl<T, U, const N: usize, const O: usize> Add<Tensor<U, N, O>> for Tensor<T, N, O>
where
    T: Add<U>,
{
    type Output = Tensor<<T as Add<U>>::Output, N, O>;

    fn add(self, rhs: Tensor<U, N, O>) -> Self::Output {
        let mut rhs_it = rhs.0.into_iter();

        Tensor(self.0.map(|it| it + rhs_it.next().unwrap()))
    }
}

impl<T, const N: usize, const O: usize> Sum for Tensor<T, N, O>
where
    T: Add<T, Output = T> + Default,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Tensor::default(), |a, b| a + b)
    }
}

impl<T, U, const N: usize, const O1: usize, const O2: usize> Mul<Tensor<U, N, O2>>
    for Tensor<T, N, O1>
where
    T: Clone + Mul<U>,
    <T as Mul<U>>::Output: Sum,
{
    type Output = <T as Mul<U>>::Output;

    fn mul(self, rhs: Tensor<U, N, O2>) -> Self::Output {
        self.0
            .clone()
            .into_iter()
            .zip(rhs.0)
            .map(|(a, b)| a * b)
            .sum()
    }
}

impl<T, const N: usize, const O: usize> Mul<f64> for Tensor<T, N, O>
where
    T: Clone + Mul<f64>,
    <T as Mul<f64>>::Output: Sum,
{
    type Output = Tensor<<T as Mul<f64>>::Output, N, O>;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor(self.0.clone().map(|it| it * rhs))
    }
}

impl<T, const N: usize, const O: usize> Mul<Tensor<T, N, O>> for f64
where
    T: Clone,
    f64: Mul<T>,
    <f64 as Mul<T>>::Output: Sum,
{
    type Output = Tensor<<f64 as Mul<T>>::Output, N, O>;

    fn mul(self, rhs: Tensor<T, N, O>) -> Self::Output {
        Tensor(rhs.0.clone().map(|it| self * it))
    }
}
