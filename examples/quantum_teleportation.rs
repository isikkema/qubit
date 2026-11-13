use qubit::{
    basis::DEG_0,
    gates::{CNOT, HADAMARD, X, Y, Z},
    qubit::Qubit,
    system::System,
    tensor::mostly_eq_phase_shifted,
    tensor_mul::TensorMul,
};
use rand::rng;

fn main() {
    // Alice entangles two qubits
    let (mut system, mut id_alice) = System::from_qubit(Qubit::new(1.0, 0.0));
    system.apply(&HADAMARD, &[&mut id_alice]).unwrap();
    let (mut id_bob, mut system) = system.push_back(Qubit::new(1.0, 0.0));
    system.apply(&CNOT, &[&mut id_alice, &mut id_bob]).unwrap();

    // And sends one to Bob
    let expected;
    let (bits, mut system) = {
        let id_bob = ();
        let _ = id_bob;

        // Alice adds the qubit she wants to teleport to the system
        let teleporting_qubit = Qubit::random();
        println!("Teleporting qubit: {:?}", teleporting_qubit.get_state());
        expected = teleporting_qubit.get_state().clone();
        let (mut id_teleport, mut system) = system.push_front(teleporting_qubit);

        system
            .apply(&CNOT, &[&mut id_teleport, &mut id_alice])
            .unwrap();
        system.apply(&HADAMARD, &[&mut id_teleport]).unwrap();

        let deg0 = DEG_0.as_tensor();
        let mut rng = rng();
        let (bit0, system, _) = system
            .measure(
                id_teleport,
                &deg0.tensor_mul(deg0).tensor_mul(deg0),
                &mut rng,
            )
            .unwrap();
        let (bit1, system, _) = system
            .measure(id_alice, &deg0.tensor_mul(deg0), &mut rng)
            .unwrap();

        let bits = (bit0 as u8) << 1 | bit1 as u8;

        (bits, system)
    };

    match bits {
        0 => (),
        1 => system.apply(&X, &[&mut id_bob]).unwrap(),
        2 => system.apply(&Z, &[&mut id_bob]).unwrap(),
        3 => system.apply(&Y, &[&mut id_bob]).unwrap(),
        _ => unreachable!(),
    }

    println!("Bob's qubit: {:?}", system.state);
    assert!(mostly_eq_phase_shifted(&system.state, &expected));
}
