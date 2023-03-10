use std::f64::consts::FRAC_1_SQRT_2;

use crate::braket::{Bra, Ket};

pub const DEG_0: Basis = Basis(Bra([Ket([1.0, 0.0]), Ket([0.0, 1.0])]));
pub const DEG_45: Basis = Basis(Bra([
    Ket([FRAC_1_SQRT_2, -FRAC_1_SQRT_2]),
    Ket([FRAC_1_SQRT_2, FRAC_1_SQRT_2]),
]));
pub const DEG_90: Basis = Basis(Bra([Ket([0.0, 1.0]), Ket([1.0, 0.0])]));

#[derive(Clone)]
pub struct Basis(Bra<Ket<f64, 2>, 2>);

impl Basis {
    pub const fn new(x: Ket<f64, 2>, y: Ket<f64, 2>) -> Self {
        Basis(Bra([x, y]))
    }

    pub fn from_radians(radians: f64) -> Self {
        Basis(Bra([
            Ket([radians.cos(), -radians.sin()]),
            Ket([radians.sin(), radians.cos()]),
        ]))
    }

    pub fn as_bra(&self) -> &Bra<Ket<f64, 2>, 2> {
        &self.0
    }

    pub fn into_bra(self) -> Bra<Ket<f64, 2>, 2> {
        self.0
    }
}
