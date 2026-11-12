use qubit::{
    basis::DEG_0,
    gates::{CNOT, HADAMARD, I, X, Y, Z},
    qubit::Qubit,
    system::System,
    tensor_mul::TensorMul,
};
use rand::rng;

fn main() {
    let (mut system, mut id_alice) = System::from_qubit(Qubit::new(1.0, 0.0));
    system.apply(&HADAMARD, &[&mut id_alice]).unwrap();
    let (mut id_bob, mut system) = system.insert(Qubit::new(1.0, 0.0));
    system.apply(&CNOT, &[&mut id_alice, &mut id_bob]).unwrap();

    // Alice does her work inside this block.
    let sent = {
        // We've sent Bob's qubit over now. This is to ensure that Alice can't touch that qubit again.
        let id_bob = ();
        let _ = id_bob;

        let bits: u8 = rand::random_range(0..4);
        match bits {
            0 => {
                system.apply(&I, &[&mut id_alice]).unwrap();
            }
            1 => {
                system.apply(&X, &[&mut id_alice]).unwrap();
            }
            2 => {
                system.apply(&Z, &[&mut id_alice]).unwrap();
            }
            3 => {
                system.apply(&Y, &[&mut id_alice]).unwrap();
            }
            _ => unreachable!(),
        }

        bits
    };

    println!("Sent: {sent}");

    system.apply(&CNOT, &[&mut id_alice, &mut id_bob]).unwrap();
    system.apply(&HADAMARD, &[&mut id_alice]).unwrap();

    let mut rng = rng();
    let (bit0, system, _) = system
        .measure(
            id_alice,
            &DEG_0.as_tensor().tensor_mul(&DEG_0.as_tensor()),
            &mut rng,
        )
        .unwrap();
    let (bit1, _, _) = system
        .measure(id_bob, &DEG_0.as_tensor(), &mut rng)
        .unwrap();

    let received = u8::from(bit0) << 1 | u8::from(bit1);
    println!("Received: {received}");

    assert_eq!(sent, received);
}
