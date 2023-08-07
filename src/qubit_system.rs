use rand::{thread_rng, Rng};

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

    pub fn sum_abs_probability_amplitudes(&self, qubit: u32) -> Ket<f64, 2> {
        // Cute
        // log_2(N) = N.trailing_zeros()

        if qubit >= N.trailing_zeros() {
            panic!(
                "Qubit index ({qubit}) exceeds max index of {}",
                N.trailing_zeros() - 1
            );
        }

        let mut p_zero = 0.0;
        let mut p_one = 0.0;

        // I am simply having too much fun
        let step = 2usize.pow(qubit);
        let offset = 2usize.pow(N.trailing_zeros() - qubit - 1);

        for i in (0..(N - offset)).step_by(step) {
            p_zero += self.state.0[i].abs();
            p_one += self.state.0[i + offset].abs();
        }

        let magnitude = (p_zero.powi(2) + p_one.powi(2)).sqrt();
        Ket([p_zero / magnitude, p_one / magnitude])
    }

    pub fn get_probability_amplitudes(&self, qubit: u32, basis: &Basis<2>) -> Ket<f64, 2> {
        basis.as_bra().T() * self.sum_abs_probability_amplitudes(qubit)
    }

    pub fn get_probabilities(&self, qubit: u32, basis: &Basis<2>) -> Ket<f64, 2> {
        self.get_probability_amplitudes(qubit, basis)
            .map(|amp| amp.powi(2))
    }

    pub(crate) fn from_ket(ket: Ket<f64, N>) -> Self {
        QubitSystem { state: ket }
    }

    pub(crate) fn from_array(array: [f64; N]) -> Self {
        QubitSystem { state: Ket(array) }
    }
}

impl QubitSystem<4> {
    pub fn measure(self, qubit: u32, basis: &Basis<2>) -> (QubitSystem<{ 4 / 2 }>, Qubit) {
        let [off_state, on_state] = basis.as_bra().as_array().clone();

        let &[off_p, on_p] = self.get_probabilities(qubit, basis).as_array();

        debug_assert!((1.0 - (off_p + on_p)).abs() < 0.0001);

        let is_on = thread_rng().gen_bool(on_p);

        let new_qubit_state = if is_on { on_state } else { off_state };
        let new_qubit = Qubit {
            state: new_qubit_state,
        };

        let step = 2usize.pow(qubit);
        let offset = if is_on {
            2usize.pow(4usize.trailing_zeros() - qubit - 1)
        } else {
            0
        };

        let mut i = 0;
        let unnormalized_system_state = [(); 4 / 2].map(|_| {
            let rv = self.state.0[i + offset];
            i += step;
            rv
        });

        println!("uns: {unnormalized_system_state:?}");

        let magnitude = unnormalized_system_state
            .iter()
            .fold(0.0, |acc, amp| acc + amp.powi(2))
            .sqrt();

        println!("mag: {magnitude}");

        let normalized_system_state = Ket(unnormalized_system_state.map(|amp| amp / magnitude));

        println!("ns: {normalized_system_state:?}");

        let new_qubit_system = QubitSystem::from_ket(basis.as_bra().T() * normalized_system_state);

        (new_qubit_system, new_qubit)
    }
}

impl From<Qubit> for QubitSystem<2> {
    fn from(value: Qubit) -> Self {
        QubitSystem { state: value.state }
    }
}
