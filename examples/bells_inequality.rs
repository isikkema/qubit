#![feature(generic_const_exprs)]

use std::{env, f64::consts::FRAC_PI_3, num::ParseIntError};

use qubit::{
    basis::{Basis, DEG_0},
    qgates,
    qubit::Qubit,
    qubit_system::QubitSystem,
};
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};

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

    let (qbs, alice_on) = {
        let md: MeasureDirection = rng.gen();

        println!("{md:?}");

        let measure_basis = match md {
            MeasureDirection::Deg0 => DEG_0,
            MeasureDirection::Deg120 => Basis::from_radians(2.0 * FRAC_PI_3),
            MeasureDirection::Deg240 => Basis::from_radians(4.0 * FRAC_PI_3),
        };

        println!("{measure_basis:?}");

        let (qbs, mut qb) = qbs.measure(0, &measure_basis);

        println!("{qb:?}");
        println!("{qbs:?}");

        (qbs, qb.measure(&measure_basis))
    };

    // let (qbs, bob_on) = {
    //     let md: MeasureDirection = rng.gen();

    //     println!("{md:?}");

    //     let measure_basis = match md {
    //         MeasureDirection::Deg0 => DEG_0,
    //         MeasureDirection::Deg120 => Basis::from_radians(2.0 * FRAC_PI_3),
    //         MeasureDirection::Deg240 => Basis::from_radians(4.0 * FRAC_PI_3),
    //     };

    //     println!("{measure_basis:?}");

    //     let (qbs, mut qb) = qbs.measure(0, &measure_basis);

    //     println!("{qb:?}");
    //     println!("{qbs:?}");

    //     (qbs, qb.measure(&measure_basis))
    // };

    Ok(())
}
