use rand::{RngExt, rng};

use crate::{
    basis::Basis, dot::Dot, tensor::Tensor, tensor_mul::TensorMul, transposable::Transposable,
};

pub struct System {
    state: Tensor<f64, 4>,
}

impl System {
    pub fn new(r: f64, s: f64, t: f64, u: f64) -> Self {
        Self {
            state: Tensor([r, s, t, u]),
        }
    }

    pub fn measure(&mut self, qubit_idx: u8, basis: &Basis) -> Option<bool> {
        if qubit_idx > 1 {
            return None;
        }

        let full_basis = Tensor([
            basis.as_tensor()[0].tensor_mul(&basis.as_tensor()[0]),
            basis.as_tensor()[0].tensor_mul(&basis.as_tensor()[1]),
            basis.as_tensor()[1].tensor_mul(&basis.as_tensor()[0]),
            basis.as_tensor()[1].tensor_mul(&basis.as_tensor()[1]),
        ]);

        self.state = full_basis.T().dot(&self.state);

        let is_on = match qubit_idx {
            0 => {
                let p_off: f64 = self.state.as_array()[..2].iter().map(|n| n.powi(2)).sum();
                let p_on: f64 = self.state.as_array()[2..].iter().map(|n| n.powi(2)).sum();

                debug_assert!(1.0 - (p_off + p_on) < 0.0001);

                let is_on = rng().random_bool(p_on);

                if !is_on {
                    self.state.0[0] /= p_off.sqrt();
                    self.state.0[1] /= p_off.sqrt();
                    self.state.0[2..].fill(0.0);
                } else {
                    self.state.0[..2].fill(0.0);
                    self.state.0[2] /= p_on.sqrt();
                    self.state.0[3] /= p_on.sqrt();
                }

                is_on
            }
            1 => {
                let p_off: f64 = self.state[0].powi(2) + self.state[2].powi(2);
                let p_on: f64 = self.state[1].powi(2) + self.state[3].powi(2);

                debug_assert!(1.0 - (p_off + p_on) < 0.0001);

                let is_on = rng().random_bool(p_on);

                if !is_on {
                    self.state.0[0] /= p_off.sqrt();
                    self.state.0[1] = 0.0;
                    self.state.0[2] /= p_off.sqrt();
                    self.state.0[3] = 0.0;
                } else {
                    self.state.0[0] = 0.0;
                    self.state.0[1] /= p_on.sqrt();
                    self.state.0[2] = 0.0;
                    self.state.0[3] /= p_on.sqrt();
                }

                is_on
            }
            _ => return None,
        };

        Some(is_on)
    }
}
