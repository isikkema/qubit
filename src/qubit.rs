use crate::braket::Ket;
use crate::{basis::Basis, transposable::Transposable};

use rand::{thread_rng, Rng};

#[derive(Debug)]
pub struct Qubit {
    state: Ket<f64, 2>,
}

impl Qubit {
    pub fn new(a: f64, b: f64) -> Self {
        Qubit { state: Ket([a, b]) }
    }

    pub fn random() -> Self {
        let mut rng = thread_rng();

        let a: f64 = rng.gen_range(-1.0..=1.0);
        let b: f64 = rng.gen_range(-1.0..=1.0);
        let c = (a * a + b * b).sqrt();

        Qubit {
            state: Ket([a / c, b / c]),
        }
    }

    pub fn get_state(&self) -> &Ket<f64, 2> {
        &self.state
    }

    pub fn get_probability_amplitudes(&self, basis: Basis) -> Ket<f64, 2> {
        basis.into_bra().T() * self.state.clone()
    }

    pub fn get_probabilities(&self, basis: Basis) -> Ket<f64, 2> {
        self.get_probability_amplitudes(basis).map(|n| n.powi(2))
    }

    pub fn measure(&mut self, basis: Basis) -> bool {
        let [off_state, on_state] = basis.as_bra().as_array().clone();

        let &[off_p, on_p] = self.get_probabilities(basis).as_array();

        debug_assert!((1.0 - (off_p + on_p)).abs() < 0.0001);

        let is_on = thread_rng().gen_bool(on_p);

        self.state = if is_on { on_state } else { off_state };

        is_on
    }
}
