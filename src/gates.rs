use std::{f64::consts::FRAC_1_SQRT_2, gca};

use crate::{system::DIM, tensor::Tensor};

pub const I: Gate<1> = Gate {
    matrix: Tensor([Tensor([1.0, 0.0]), Tensor([0.0, 1.0])]),
};

pub const Z: Gate<1> = Gate {
    matrix: Tensor([Tensor([1.0, 0.0]), Tensor([0.0, -1.0])]),
};

pub const X: Gate<1> = Gate {
    matrix: Tensor([Tensor([0.0, 1.0]), Tensor([1.0, 0.0])]),
};

pub const Y: Gate<1> = Gate {
    matrix: Tensor([Tensor([0.0, 1.0]), Tensor([-1.0, 0.0])]),
};

pub const HADAMARD: Gate<1> = Gate {
    matrix: Tensor([
        Tensor([FRAC_1_SQRT_2, FRAC_1_SQRT_2]),
        Tensor([FRAC_1_SQRT_2, -FRAC_1_SQRT_2]),
    ]),
};

pub const CNOT: Gate<2> = Gate {
    matrix: Tensor([
        Tensor([1.0, 0.0, 0.0, 0.0]),
        Tensor([0.0, 1.0, 0.0, 0.0]),
        Tensor([0.0, 0.0, 0.0, 1.0]),
        Tensor([0.0, 0.0, 1.0, 0.0]),
    ]),
};

pub struct Gate<const Q: u8> {
    pub(crate) matrix: Tensor<Tensor<f64, gca!(DIM::<Q>)>, gca!(DIM::<Q>)>,
}

#[cfg(test)]
mod tests {
    use crate::{qubit::Qubit, system::System, tensor::mostly_eq};

    use super::*;

    #[test]
    fn test_hadamard() {
        let (mut system, mut ids) = System::<1>::from_state(Tensor([1.0, 0.0]));
        system.apply(&HADAMARD, &[&mut ids.pop().unwrap()]).unwrap();
        assert_eq!(system.state, Tensor([FRAC_1_SQRT_2, FRAC_1_SQRT_2]));

        let (mut system, mut ids) = System::<1>::from_state(Tensor([0.0, 1.0]));
        system.apply(&HADAMARD, &[&mut ids.pop().unwrap()]).unwrap();
        assert_eq!(system.state, Tensor([FRAC_1_SQRT_2, -FRAC_1_SQRT_2]));

        let expected = Qubit::random().get_state().clone();
        let (mut system, mut ids) = System::<1>::from_state(expected.clone());
        let mut id = ids.pop().unwrap();
        system.apply(&HADAMARD, &[&mut id]).unwrap();
        system.apply(&HADAMARD, &[&mut id]).unwrap();
        assert!(mostly_eq(&system.state, &expected));
    }

    #[test]
    fn test_cnot() {
        let input = Qubit::new(1.0, 0.0);

        let (mut system, mut id1) = System::from_qubit(input);
        system.apply(&HADAMARD, &[&mut id1]).unwrap();

        let (mut id2, mut system) = system.push_back(Qubit::new(1.0, 0.0));
        system.apply(&CNOT, &[&mut id1, &mut id2]).unwrap();

        assert_eq!(
            system.state,
            Tensor([FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2])
        );
    }
}
