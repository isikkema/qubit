use std::{env, num::ParseIntError};

fn main() -> Result<(), ParseIntError> {
    let _num_qubits: u32 = match env::args().nth(1) {
        Some(s) => s.parse()?,
        None => 100_000,
    };

    Ok(())
}
