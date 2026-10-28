use std::{
    array,
    ops::{Index, IndexMut},
};

#[derive(Clone, PartialEq, Debug)]
pub struct Tensor<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> Tensor<T, N> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        N == 0
    }

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

pub fn mostly_eq<const N: usize>(a: &Tensor<f64, N>, b: &Tensor<f64, N>) -> bool {
    a.0.iter()
        .zip(b.0.iter())
        .all(|(a, b)| (a - b).abs() < 0.00001)
}
