use rand::{thread_rng, Rng};

use crate::{basis::Basis, tensor::Tensor, transposable::Transposable};

#[derive(Debug)]
pub struct Qubit {
    state: Tensor<f64, 2, 1>,
}

impl Qubit {
    pub fn new(a: f64, b: f64) -> Self {
        Qubit {
            state: Tensor([a, b]),
        }
    }

    pub fn random() -> Self {
        let mut rng = thread_rng();

        let a: f64 = rng.gen_range(-1.0..=1.0);
        let b: f64 = rng.gen_range(-1.0..=1.0);
        let c = (a * a + b * b).sqrt();

        Qubit {
            state: Tensor([a / c, b / c]),
        }
    }

    pub fn get_state(&self) -> &Tensor<f64, 2, 1> {
        &self.state
    }

    pub fn get_probability_amplitudes(&self, basis: Basis) -> Tensor<f64, 2, 1> {
        basis.as_tensor().T() * self.state.clone()
    }

    pub fn get_probabilities(&self, basis: Basis) -> Tensor<f64, 2, 1> {
        self.get_probability_amplitudes(basis).map(|n| n.powi(2))
    }

    pub fn measure(&mut self, basis: Basis) -> bool {
        let [off_state, on_state] = basis.as_tensor().as_array().clone();

        let &[off_p, on_p] = self.get_probabilities(basis).as_array();

        debug_assert!((1.0 - (off_p + on_p)).abs() < 0.0001);

        let is_on = thread_rng().gen_bool(on_p);

        self.state = if is_on { on_state } else { off_state };

        is_on
    }
}
