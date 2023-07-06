#![feature(generic_const_exprs)]

use std::{env, num::ParseIntError};

use qubit::{basis::DEG_0, qgates, qubit::Qubit, qubit_system::QubitSystem};
use rand::{distributions::Standard, prelude::Distribution, thread_rng};

#[derive(Debug)]
enum MeasureDirection {
    Deg0,
    Deg120,
    Deg240,
}

impl Distribution<MeasureDirection> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> MeasureDirection {
        match rng.gen_range(0..=2) {
            0 => MeasureDirection::Deg0,
            1 => MeasureDirection::Deg120,
            _ => MeasureDirection::Deg240,
        }
    }
}

fn main() -> Result<(), ParseIntError> {
    let num_qubits: u32 = match env::args().nth(1) {
        Some(s) => s.parse()?,
        None => 100_000,
    };

    let mut rng = thread_rng();

    let qbs: QubitSystem<2> = Qubit::deg45().into();
    println!("{qbs:?}");

    let qbs = qbs.add_system(Qubit::zero().into());
    println!("{qbs:?}");

    let qbs = qgates::cnot(qbs);

    println!("{qbs:?}");

    let (qbs, qb) = qbs.measure(1, DEG_0);

    println!("{qb:?}");
    println!("{qbs:?}");

    // let md: MeasureDirection = rng.gen();

    // println!("{md:?}");

    // We need to measure one at a time
    // let measure_basis = match md {
    //     MeasureDirection::Deg0 => todo!(),
    //     MeasureDirection::Deg120 => todo!(),
    //     MeasureDirection::Deg240 => todo!(),
    // };

    Ok(())
}
