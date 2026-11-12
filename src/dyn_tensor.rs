use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

use crate::tensor::Tensor;

#[derive(Clone, PartialEq, Debug)]
pub struct DynTensor<T>(pub Box<[T]>);

impl<T> DynTensor<T> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_array(&self) -> &[T] {
        &self.0
    }
}

impl<T> IntoIterator for DynTensor<T> {
    type Item = T;

    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> IndexMut<usize> for DynTensor<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<T> Index<usize> for DynTensor<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, const N: usize> From<Tensor<T, N>> for DynTensor<T> {
    fn from(value: Tensor<T, N>) -> Self {
        Self(value.0.into())
    }
}

impl<T, const N: usize, const M: usize> From<Tensor<Tensor<T, M>, N>> for DynTensor<DynTensor<T>> {
    fn from(value: Tensor<Tensor<T, M>, N>) -> Self {
        Self(value.0.map(DynTensor::from).into())
    }
}

impl<T, const N: usize> TryFrom<DynTensor<T>> for Tensor<T, N> {
    type Error = DynTensor<T>;

    fn try_from(value: DynTensor<T>) -> Result<Self, Self::Error> {
        let sized: Box<[T; N]> = value.0.try_into().map_err(|e| DynTensor(e))?;
        Ok(Tensor(*sized))
    }
}

impl<T: Debug, const N: usize, const M: usize> TryFrom<DynTensor<DynTensor<T>>>
    for Tensor<Tensor<T, M>, N>
{
    type Error = DynTensor<DynTensor<T>>;

    fn try_from(value: DynTensor<DynTensor<T>>) -> Result<Self, Self::Error> {
        if value.len() != N || value.0.iter().any(|row| row.len() != M) {
            return Err(value);
        }

        let sized: Box<[Tensor<T, M>]> = value
            .0
            .into_iter()
            .map(|row| Tensor::<T, M>::try_from(row).unwrap())
            .collect();

        Ok(Tensor(*Box::<[Tensor<T, M>; N]>::try_from(sized).unwrap()))
    }
}

pub fn mostly_eq<const N: usize>(a: &DynTensor<f64>, b: &DynTensor<f64>) -> bool {
    a.len() == b.len()
        && a.0
            .iter()
            .zip(b.0.iter())
            .all(|(a, b)| (a - b).abs() < 0.00001)
}

#[cfg(test)]
mod tests {
    use crate::tensor_mul::TensorMul;

    use super::*;

    #[test]
    fn test_dyn_tensor_mul() {
        let t: DynTensor<DynTensor<f64>> = Tensor([Tensor([1.0, 0.0]), Tensor([0.0, 1.0])]).into();

        let mut product = t.clone();
        for _ in 0..2 {
            product = product.tensor_mul(&t);
        }

        let expected = Tensor([
            Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ]);

        let actual: Tensor<Tensor<f64, 8>, 8> = product.try_into().unwrap();

        assert_eq!(expected, actual);
    }
}
