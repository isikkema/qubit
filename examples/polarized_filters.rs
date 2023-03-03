use std::{env, num::ParseIntError};

use qubit::{
    basis::{DEG_0, DEG_45, DEG_90},
    filter::Filter,
    qubit::Qubit,
};

fn main() -> Result<(), ParseIntError> {
    let num_qubits: u32 = match env::args().nth(1) {
        Some(s) => s.parse()?,
        None => 100_000,
    };

    let mut filter_0 = Filter::new(DEG_0, true);
    let mut filter_45 = Filter::new(DEG_45, true);
    let mut filter_90 = Filter::new(DEG_90, true);

    let mut qb;

    for _ in 0..num_qubits {
        qb = Qubit::random();

        qb = if let Some(qb) = filter_0.filter(qb) {
            qb
        } else {
            continue;
        };

        filter_90.filter(qb);
    }

    println!(
        "[0 - 90] Passed: {:.3}%",
        (filter_90.num_passed * 100) as f64 / filter_0.num_total as f64
    );

    filter_0.reset();
    filter_45.reset();
    filter_90.reset();

    for _ in 0..num_qubits {
        qb = Qubit::random();

        qb = if let Some(qb) = filter_0.filter(qb) {
            qb
        } else {
            continue;
        };

        qb = if let Some(qb) = filter_45.filter(qb) {
            qb
        } else {
            continue;
        };

        filter_90.filter(qb);
    }

    println!(
        "[0 - 45 - 90] Passed: {:.3}%",
        (filter_90.num_passed * 100) as f64 / filter_0.num_total as f64
    );

    Ok(())
}
