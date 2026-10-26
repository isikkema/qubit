use std::gca;

use rand::{RngExt, rng};

use crate::{dot::Dot, tensor::Tensor, transposable::Transposable};

const DIM<const Q: u8>: usize = const { 1 << Q };

pub struct System<const Q: u8> {
    state: Tensor<f64, gca!(DIM::<Q>)>,
}

impl<const Q: u8> System<Q> {
    pub fn new(state: Tensor<f64, gca!(DIM::<Q>)>) -> Self {
        Self { state }
    }

    pub fn measure(
        &mut self,
        qubit_idx: u8,
        basis: &Tensor<Tensor<f64, gca!(DIM::<Q>)>, gca!(DIM::<Q>)>,
    ) -> Option<bool> {
        if qubit_idx > 1 {
            return None;
        }

        self.state = basis.T().dot(&self.state);

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
