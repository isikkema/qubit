use crate::{braket::Ket, qubit::Qubit, tensor_mul::TensorMul, basis::Basis};

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

    pub fn measure(&mut self, basis: Basis<N>) -> Ket<f64, N> {
        todo!()
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
