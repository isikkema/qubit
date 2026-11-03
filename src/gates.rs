use std::f64::consts::FRAC_1_SQRT_2;

use crate::{dot::Dot, system::System, tensor::Tensor, transposable::Transposable};

pub fn hadamard(input: System<1>) -> System<1> {
    let System { state, id_map } = input;

    let state = Tensor([
        Tensor([FRAC_1_SQRT_2, FRAC_1_SQRT_2]),
        Tensor([FRAC_1_SQRT_2, -FRAC_1_SQRT_2]),
    ])
    .T()
    .dot(&state);

    System { state, id_map }
}

pub fn cnot(input: System<2>) -> System<2> {
    let System { state, id_map } = input;

    let state = Tensor([
        Tensor([1.0, 0.0, 0.0, 0.0]),
        Tensor([0.0, 1.0, 0.0, 0.0]),
        Tensor([0.0, 0.0, 0.0, 1.0]),
        Tensor([0.0, 0.0, 1.0, 0.0]),
    ])
    .T()
    .dot(&state);

    System { state, id_map }
}

#[cfg(test)]
mod tests {
    use crate::{qubit::Qubit, tensor::mostly_eq};

    use super::*;

    #[test]
    fn test_hadamard() {
        assert_eq!(
            hadamard(System::from_state(Tensor([1.0, 0.0])).0).state,
            Tensor([FRAC_1_SQRT_2, FRAC_1_SQRT_2])
        );
        assert_eq!(
            hadamard(System::from_state(Tensor([0.0, 1.0])).0).state,
            Tensor([FRAC_1_SQRT_2, -FRAC_1_SQRT_2])
        );

        let expected = Qubit::random().get_state().clone();
        assert!(mostly_eq(
            &hadamard(hadamard(System::from_state(expected.clone()).0)).state,
            &expected
        ));
    }

    #[test]
    fn test_cnot() {
        let input = Qubit::new(1.0, 0.0);

        let input = hadamard(System::from_qubit(input).0);
        let input = input.insert(Qubit::new(1.0, 0.0)).1;

        assert_eq!(
            cnot(input).state,
            Tensor([FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2])
        );
    }
}
