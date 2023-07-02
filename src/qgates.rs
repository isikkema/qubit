use crate::{
    braket::{Bra, Ket},
    qubit_system::QubitSystem,
};

pub fn cnot(system: QubitSystem<4>) -> QubitSystem<4> {
    QubitSystem::from_ket(
        Bra([
            Ket([1.0, 0.0, 0.0, 0.0]),
            Ket([0.0, 1.0, 0.0, 0.0]),
            Ket([0.0, 0.0, 0.0, 1.0]),
            Ket([0.0, 0.0, 1.0, 0.0]),
        ]) * system.state,
    )
}
