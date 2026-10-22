use std::{
    array,
    iter::Sum,
    ops::{Add, Index, IndexMut, Mul},
};

#[derive(Clone, PartialEq, Debug)]
pub struct Tensor<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> Tensor<T, N> {
    pub fn as_array(&self) -> &[T; N] {
        &self.0
    }

    pub fn map<F, U>(self, f: F) -> Tensor<U, N>
    where
        F: FnMut(T) -> U,
    {
        Tensor(self.0.map(f))
    }
}

impl<const N: usize> Tensor<f64, N> {
    pub fn tensor_product(&self, rhs: &Self) -> Tensor<f64, { N * N }> {
        let mut iter = self.0.iter().flat_map(|n| *n * rhs.clone());

        Tensor(array::from_fn(|_| iter.next().unwrap()))
    }
}

impl<T, const N: usize> Default for Tensor<T, N>
where
    T: Default,
{
    fn default() -> Self {
        Tensor([(); N].map(|_| T::default()))
    }
}

impl<T, const N: usize> IntoIterator for Tensor<T, N> {
    type Item = T;

    type IntoIter = array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, const N: usize> IndexMut<usize> for Tensor<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<T, const N: usize> Index<usize> for Tensor<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, U, const N: usize> Add<Tensor<U, N>> for Tensor<T, N>
where
    T: Add<U>,
{
    type Output = Tensor<<T as Add<U>>::Output, N>;

    fn add(self, rhs: Tensor<U, N>) -> Self::Output {
        let mut rhs_it = rhs.0.into_iter();

        Tensor(self.0.map(|it| it + rhs_it.next().unwrap()))
    }
}

impl<T, const N: usize> Sum for Tensor<T, N>
where
    T: Add<T, Output = T> + Default,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Tensor::default(), |a, b| a + b)
    }
}

impl<T, U, const N: usize> Mul<Tensor<U, N>> for Tensor<T, N>
where
    T: Clone + Mul<U>,
    <T as Mul<U>>::Output: Sum,
{
    type Output = <T as Mul<U>>::Output;

    fn mul(self, rhs: Tensor<U, N>) -> Self::Output {
        self.0
            .clone()
            .into_iter()
            .zip(rhs.0)
            .map(|(a, b)| a * b)
            .sum()
    }
}

impl<T, const N: usize> Mul<f64> for Tensor<T, N>
where
    T: Clone + Mul<f64>,
    <T as Mul<f64>>::Output: Sum,
{
    type Output = Tensor<<T as Mul<f64>>::Output, N>;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor(self.0.clone().map(|it| it * rhs))
    }
}

impl<T, const N: usize> Mul<Tensor<T, N>> for f64
where
    T: Clone,
    f64: Mul<T>,
    <f64 as Mul<T>>::Output: Sum,
{
    type Output = Tensor<<f64 as Mul<T>>::Output, N>;

    fn mul(self, rhs: Tensor<T, N>) -> Self::Output {
        Tensor(rhs.0.clone().map(|it| self * it))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_product() {
        let t1: Tensor<f64, 2> = Tensor([1.0, 0.0]);
        let t2: Tensor<f64, 2> = Tensor([1.0, 0.0]);

        let result = Tensor([1.0, 0.0, 0.0, 0.0]);

        assert_eq!(t1.tensor_product(&t2), result)
    }
}
