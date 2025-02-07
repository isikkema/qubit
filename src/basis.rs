use std::{f64::consts::FRAC_1_SQRT_2, mem};

use crate::{
    braket::{Bra, Ket},
    kronecker::Kronecker,
};

pub const DEG_0: Basis<2> = Basis(Bra([Ket([1.0, 0.0]), Ket([0.0, 1.0])]));
pub const DEG_45: Basis<2> = Basis(Bra([
    Ket([FRAC_1_SQRT_2, -FRAC_1_SQRT_2]),
    Ket([FRAC_1_SQRT_2, FRAC_1_SQRT_2]),
]));
pub const DEG_90: Basis<2> = Basis(Bra([Ket([0.0, 1.0]), Ket([1.0, 0.0])]));

#[derive(Debug, Clone)]
pub struct Basis<const N: usize>(Bra<Ket<f64, N>, N>);

impl<const N: usize> Basis<N> {
    pub fn as_bra(&self) -> &Bra<Ket<f64, N>, N> {
        &self.0
    }

    pub fn into_bra(self) -> Bra<Ket<f64, N>, N> {
        self.0
    }
}

impl Basis<2> {
    pub const fn new(x: Ket<f64, 2>, y: Ket<f64, 2>) -> Self {
        Basis(Bra([x, y]))
    }

    pub fn from_radians(radians: f64) -> Self {
        Basis(Bra([
            Ket([radians.cos(), -radians.sin()]),
            Ket([radians.sin(), radians.cos()]),
        ]))
    }

    // pub fn scale_to_n<const N: usize, const M: usize, const P: usize, const Q: usize>(&self, index: u32) -> Basis<N>
    // where
    //     [(); N >> 1]:,
    //     [(); N * P]:,
    //     [(); M * Q]:,
    // {
    //     if N == 2 {
    //         let scaled_basis = if index == 0 {
    //             self.clone()
    //         } else {
    //             Basis(Bra([Ket([1.0, 0.0]), Ket([0.0, 1.0])]))
    //         };

    //         return unsafe { *(&scaled_basis as *const Basis<2> as *const Basis<N>) };
    //     }

    //     let num_qubits = N.trailing_zeros();

    //     let scaling_factor = if index == num_qubits {
    //         self.as_bra().clone()
    //     } else {
    //         Bra([Ket([1.0, 0.0]), Ket([0.0, 1.0])])
    //     };

    //     let scaled_basis = Basis(
    //         self.scale_to_n::<{ N >> 1 }>(index)
    //             .as_bra()
    //             .kron(&scaling_factor),
    //     );

    //     unsafe { *(&scaled_basis as *const Basis<N>) }
    // }
}
