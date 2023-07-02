#![feature(generic_const_exprs)]

use std::{env, num::ParseIntError};

use qubit::{qgates, qubit::Qubit, qubit_system::QubitSystem};

fn main() -> Result<(), ParseIntError> {
    let num_qubits: u32 = match env::args().nth(1) {
        Some(s) => s.parse()?,
        None => 100_000,
    };

    // for _ in 0..num_qubits {
    let qbs: QubitSystem<2> = Qubit::deg45().into();
    println!("{qbs:?}");

    let qbs = qbs.add_system(Qubit::zero().into());
    println!("{qbs:?}");

    let qbs = qgates::cnot(qbs);

    println!("{qbs:?}");
    // }

    Ok(())
}
