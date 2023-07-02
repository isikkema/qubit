use std::{
    iter::Sum,
    mem,
    ops::{Add, Mul},
};

use crate::{tensor_mul::TensorMul, transposable::Transposable, util::flatten_arrays};

#[derive(Clone, Debug, PartialEq)]
pub struct Bra<T, const N: usize>(pub [T; N])
where
    T: Clone + Transposable;

impl<T, const N: usize> Bra<T, N>
where
    T: Clone + Transposable,
{
    pub fn as_array(&self) -> &[T; N] {
        &self.0
    }

    pub fn map<U, F>(self, f: F) -> Bra<U, N>
    where
        U: Clone + Transposable,
        F: FnMut(T) -> U,
    {
        Bra(self.0.map(f))
    }
}

impl<T, const N: usize> Transposable for Bra<T, N>
where
    T: Clone + Transposable,
{
    type Output = Ket<<T as Transposable>::Output, N>;

    fn T(&self) -> Self::Output {
        let t_values: [<T as Transposable>::Output; N] = self.0.clone().map(|it| it.T());

        Ket::<<T as Transposable>::Output, N>(t_values)
    }
}

impl<U, T, const N: usize> Mul<Ket<U, N>> for Bra<T, N>
where
    <T as Mul<U>>::Output: Sum,
    T: Mul<U>,
    U: Clone + Transposable,
    T: Clone + Transposable + Mul<U>,
{
    type Output = <T as Mul<U>>::Output;

    fn mul(self, rhs: Ket<U, N>) -> Self::Output {
        self.0
            .clone()
            .into_iter()
            .zip(rhs.0)
            .map(|(a, b)| a * b)
            .sum()
    }
}

impl<U, T, const N: usize> Add<Bra<U, N>> for Bra<T, N>
where
    <T as Add<U>>::Output: Clone + Transposable,
    U: Clone + Transposable,
    T: Clone + Transposable + Add<U>,
{
    type Output = Bra<<T as Add<U>>::Output, N>;

    fn add(self, rhs: Bra<U, N>) -> Self::Output {
        let mut rhs_it = rhs.0.into_iter();

        Bra(self.0.map(|it| it + rhs_it.next().unwrap()))
    }
}

impl<U, T, const N: usize, const M: usize> Mul<Bra<T, M>> for Bra<T, N>
where
    <T as Mul<T>>::Output: Clone + Transposable,
    <T as Mul<Bra<T, M>>>::Output: Clone + Transposable,
    [(); N * M]:,
    U: Clone + Transposable,
    T: Clone + Transposable + Mul<T> + Mul<Bra<T, M>, Output = Bra<U, M>>,
{
    type Output = Bra<U, { N * M }>;

    fn mul(self, rhs: Bra<T, M>) -> Self::Output {
        let bra_arrays = self.0.map(|it| (it * rhs.clone()).0);
        let array = unsafe {
            let bra_vec = bra_arrays.concat();
            mem::transmute::<_, &[U; N * M]>(bra_vec.as_slice() as *const [U] as *const U).clone()
        };

        Bra(array)
    }
}

impl<V, U, T, const N: usize, const M: usize> TensorMul<Bra<U, M>> for Bra<T, N>
where
    <T as Mul<U>>::Output: Clone + Transposable,
    <T as Mul<Bra<U, M>>>::Output: Clone + Transposable,
    [(); N * M]:,
    V: Clone + Transposable,
    U: Clone + Transposable,
    T: Clone + Transposable + Mul<U> + Mul<Bra<U, M>, Output = Bra<V, M>>,
{
    type Output = Bra<V, { N * M }>;

    fn tensor_mul(self, rhs: Bra<U, M>) -> Self::Output {
        Bra(flatten_arrays(self.0.map(|it| (it * rhs.clone()).0)))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ket<T: Clone + Transposable, const N: usize>(pub [T; N]);

impl<T: Clone + Transposable, const N: usize> Ket<T, N> {
    pub fn as_array(&self) -> &[T; N] {
        &self.0
    }

    pub fn map<U, F>(self, f: F) -> Ket<U, N>
    where
        U: Clone + Transposable,
        F: FnMut(T) -> U,
    {
        Ket(self.0.map(f))
    }
}

impl<T: Clone + Transposable, const N: usize> Transposable for Ket<T, N> {
    type Output = Bra<<T as Transposable>::Output, N>;

    fn T(&self) -> Self::Output {
        let t_values: [<T as Transposable>::Output; N] = self.0.clone().map(|it| it.T());

        Bra::<<T as Transposable>::Output, N>(t_values)
    }
}

impl<U: Clone + Transposable, T: Clone + Transposable + Mul<U>, const N: usize> Mul<U> for Ket<T, N>
where
    <T as Mul<U>>::Output: Clone + Transposable + Sum,
{
    type Output = Ket<<T as Mul<U>>::Output, N>;

    fn mul(self, rhs: U) -> Self::Output {
        Ket(self.0.map(|bra| bra * rhs.clone()))
    }
}

// impl<U: Clone + Transposable, T: Clone + Transposable + Mul<U>, const N: usize, const M: usize>
//     Mul<Ket<U, M>> for Ket<Bra<T, M>, N>
// where
//     <T as Mul<U>>::Output: Clone + Transposable + Sum,
// {
//     type Output = Ket<<Bra<T, M> as Mul<Ket<U, M>>>::Output, N>;

//     fn mul(self, rhs: Ket<U, M>) -> Self::Output {
//         Ket(self.0.map(|bra| bra * rhs.clone()))
//     }
// }

impl<U: Clone + Transposable, const N: usize> Mul<Ket<U, N>> for f64
where
    <f64 as Mul<U>>::Output: Clone + Transposable,
    f64: Mul<U>,
{
    type Output = Ket<<f64 as Mul<U>>::Output, N>;

    fn mul(self, rhs: Ket<U, N>) -> Self::Output {
        Ket(rhs.0.map(|it| self * it))
    }
}

impl<U: Clone + Transposable, T: Clone + Transposable + Add<U>, const N: usize> Add<Ket<U, N>>
    for Ket<T, N>
where
    <T as Add<U>>::Output: Clone + Transposable,
{
    type Output = Ket<<T as Add<U>>::Output, N>;

    fn add(self, rhs: Ket<U, N>) -> Self::Output {
        let mut rhs_it = rhs.0.into_iter();

        Ket(self.0.map(|it| it + rhs_it.next().unwrap()))
    }
}

impl<T, const N: usize> Sum for Ket<T, N>
where
    T: Clone + Transposable + Add<Output = T>,
    <T as Add>::Output: Clone + Transposable,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, e| acc + e).unwrap()
    }
}

impl<V, U, T, const N: usize, const M: usize> TensorMul<Ket<U, M>> for Ket<T, N>
where
    <T as Mul<U>>::Output: Clone + Transposable,
    <T as Mul<Ket<U, M>>>::Output: Clone + Transposable,
    [(); N * M]:,
    V: Clone + Transposable,
    U: Clone + Transposable,
    T: Clone + Transposable + Mul<U> + Mul<Ket<U, M>, Output = Ket<V, M>>,
{
    type Output = Ket<V, { N * M }>;

    fn tensor_mul(self, rhs: Ket<U, M>) -> Self::Output {
        Ket(flatten_arrays(self.0.map(|it| (it * rhs.clone()).0)))
    }
}

impl<V, U, T, const N: usize, const X: usize, const Y: usize> TensorMul<Bra<Ket<U, Y>, X>>
    for Ket<T, N>
where
    <T as Mul<U>>::Output: Clone + Transposable,
    <T as Mul<Ket<U, Y>>>::Output: Clone + Transposable,
    [(); N * Y]:,
    V: Clone + Transposable,
    U: Clone + Transposable,
    T: Clone + Transposable + Mul<U> + Mul<Ket<U, Y>, Output = Ket<V, Y>>,
{
    type Output = Bra<Ket<V, { N * Y }>, X>;

    fn tensor_mul(self, rhs: Bra<Ket<U, Y>, X>) -> Self::Output {
        rhs.map(|rhs_ket| self.clone().tensor_mul(rhs_ket))
    }
}

impl<V, U, T, const N: usize, const M: usize, const X: usize, const Y: usize>
    TensorMul<Bra<Ket<U, Y>, X>> for Bra<Ket<T, M>, N>
where
    <T as Mul<U>>::Output: Clone + Transposable,
    <T as Mul<Ket<U, Y>>>::Output: Clone + Transposable,
    [(); M * Y]:,
    [(); N * X]:,
    V: Clone + Transposable,
    U: Clone + Transposable,
    T: Clone + Transposable + Mul<U> + Mul<Ket<U, Y>, Output = Ket<V, Y>>,
{
    type Output = Bra<Ket<V, { M * Y }>, { N * X }>;

    fn tensor_mul(self, rhs: Bra<Ket<U, Y>, X>) -> Self::Output {
        Bra(flatten_arrays(
            self.0.map(|ket| (ket.tensor_mul(rhs.clone())).0),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_mul() {
        let a = Ket([1.0, 2.0]);
        let b = Ket([3.0, 4.0]);

        assert_eq!(a.tensor_mul(b), Ket([3.0, 4.0, 6.0, 8.0]));
    }

    #[test]
    fn test_tensors_mul() {
        let a = Bra([Ket([1.0, 0.0]), Ket([0.0, 1.0])]);
        let b = Bra([Ket([1.0, 0.0]), Ket([0.0, 1.0])]);

        assert_eq!(
            a.tensor_mul(b),
            Bra([
                Ket([1.0, 0.0, 0.0, 0.0]),
                Ket([0.0, 1.0, 0.0, 0.0]),
                Ket([0.0, 0.0, 1.0, 0.0]),
                Ket([0.0, 0.0, 0.0, 1.0]),
            ])
        );
    }
}
