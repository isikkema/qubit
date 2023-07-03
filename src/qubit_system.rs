use rand::{distributions::WeightedIndex, prelude::Distribution, thread_rng};

use crate::{
    basis::Basis, braket::Ket, qubit::Qubit, tensor_mul::TensorMul, transposable::Transposable,
};

#[derive(Debug)]
pub struct QubitSystem<const N: usize> {
    pub(crate) state: Ket<f64, N>,
}

impl<const N: usize> QubitSystem<N> {
    pub fn add_system<const M: usize>(self, other: QubitSystem<M>) -> QubitSystem<{ N * M }> {
        QubitSystem {
            state: self.state.tensor_mul(other.state),
        }
    }

    pub fn is_entangled(&self) -> bool {
        todo!()
    }

    pub fn get_probability_amplitudes(&self, basis: Basis<N>) -> Ket<f64, N> {
        basis.into_bra().T() * self.state.clone()
    }

    pub fn get_probabilities(&self, basis: Basis<N>) -> Ket<f64, N> {
        self.get_probability_amplitudes(basis).map(|n| n.powi(2))
    }

    /*
    Measure only 1 qubit at a time.
    Sum up probabilities for 0 vs 1, choose, then collapse inner state in half.
    Only elements where the qubit was {result} remain.
    How to normalize amplitudes afterwards?
     */
    pub fn measure(
        self,
        qubit: usize, /* how to specify qbit to measure??? */
        basis: Basis<2>,
    ) -> (QubitSystem<{ N / 2 }>, Ket<f64, 2>) {
        todo!()
        // let states = basis.as_bra().as_array().clone();
        // let probabilities = self.get_probabilities(basis);

        // let dist = WeightedIndex::new(probabilities.as_array()).unwrap();
        // let idx = dist.sample(&mut thread_rng());

        // self.state = states[idx].clone();

        // &self.state
    }

    pub(crate) fn from_ket(ket: Ket<f64, N>) -> Self {
        QubitSystem { state: ket }
    }
}

impl From<Qubit> for QubitSystem<2> {
    fn from(value: Qubit) -> Self {
        QubitSystem { state: value.state }
    }
}
