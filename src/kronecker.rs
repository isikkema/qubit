use std::ops::Mul;

use crate::{
    braket::{Bra, Ket},
    transposable::Transposable,
};

pub trait Kronecker<RHS> {
    type Output;

    fn kron(&self, rhs: &RHS) -> Self::Output;
}

impl<T, const N: usize, const M: usize, const P: usize, const Q: usize> Kronecker<Bra<Ket<T, Q>, P>>
    for Bra<Ket<T, M>, N>
where
    T: Clone + Transposable + Mul<T>,
    <T as Mul<T>>::Output: Clone + Transposable,
    [(); M * Q]:,
    [(); N * P]:,
{
    type Output = Bra<Ket<<T as Mul<T>>::Output, { M * Q }>, { N * P }>;

    fn kron(&self, rhs: &Bra<Ket<T, Q>, P>) -> Self::Output {
        let mut bra: Vec<Ket<<T as Mul<T>>::Output, { M * Q }>> = Vec::with_capacity(N * P);

        for ai in 0..N {
            for bi in 0..P {
                let mut ket = Vec::with_capacity(M * Q);
                for aj in 0..M {
                    for bj in 0..Q {
                        ket.push(self.0[ai].0[aj].clone() * rhs.0[bi].0[bj].clone());
                    }
                }

                bra.push(Ket(ket.try_into().unwrap_or_else(|_| unreachable!())));
            }
        }

        Bra(bra.try_into().unwrap_or_else(|_| unreachable!()))
    }
}

#[cfg(test)]
mod tests {
    use crate::braket::{Bra, Ket};

    use super::Kronecker;

    #[test]
    fn test_identity() {
        let identity = Bra([Ket([1.0, 0.0]), Ket([0.0, 1.0])]);

        let result = Bra([
            Ket([1.0, 0.0, 0.0, 0.0]),
            Ket([0.0, 1.0, 0.0, 0.0]),
            Ket([0.0, 0.0, 1.0, 0.0]),
            Ket([0.0, 0.0, 0.0, 1.0]),
        ]);

        assert_eq!(identity.kron(&identity), result);
    }

    #[test]
    fn test_wiki_example() {
        let a = Bra([Ket([1.0, 3.0]), Ket([2.0, 4.0])]);

        let b = Bra([Ket([0.0, 6.0]), Ket([5.0, 7.0])]);

        let result = Bra([
            Ket([0.0, 6.0, 0.0, 18.0]),
            Ket([5.0, 7.0, 15.0, 21.0]),
            Ket([0.0, 12.0, 0.0, 24.0]),
            Ket([10.0, 14.0, 20.0, 28.0]),
        ]);

        assert_eq!(a.kron(&b), result);
    }
}
