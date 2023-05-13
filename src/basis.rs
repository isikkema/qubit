use std::f64::consts::FRAC_1_SQRT_2;

use crate::tensor::Tensor;

pub const DEG_0: Basis = Basis(Tensor([Tensor([1.0, 0.0]), Tensor([0.0, 1.0])]));
pub const DEG_45: Basis = Basis(Tensor([
    Tensor([FRAC_1_SQRT_2, -FRAC_1_SQRT_2]),
    Tensor([FRAC_1_SQRT_2, FRAC_1_SQRT_2]),
]));
pub const DEG_90: Basis = Basis(Tensor([Tensor([0.0, 1.0]), Tensor([1.0, 0.0])]));

#[derive(Clone)]
pub struct Basis(Tensor<Tensor<f64, 2, 1>, 2, 2>);

impl Basis {
    pub const fn new(x: Tensor<f64, 2, 1>, y: Tensor<f64, 2, 1>) -> Self {
        Basis(Tensor([x, y]))
    }

    pub fn from_radians(radians: f64) -> Self {
        Basis(Tensor([
            Tensor([radians.cos(), -radians.sin()]),
            Tensor([radians.sin(), radians.cos()]),
        ]))
    }

    pub fn as_tensor(&self) -> &Tensor<Tensor<f64, 2, 1>, 2, 2> {
        &self.0
    }
}
