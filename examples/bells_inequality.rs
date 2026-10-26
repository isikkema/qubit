use std::{env, f64::consts::FRAC_1_SQRT_2, num::ParseIntError};

use qubit::{
    basis::{DEG_0, DEG_120, DEG_240},
    system::System,
    tensor::Tensor,
    tensor_mul::TensorMul,
};
use rand::{
    rng,
    seq::{IndexedRandom, SliceRandom},
};

fn main() -> Result<(), ParseIntError> {
    let num_runs: u32 = match env::args().nth(1) {
        Some(s) => s.parse()?,
        None => 100_000,
    };

    let measurement_angles = [DEG_0, DEG_120, DEG_240];
    let mut rng = rng();
    let mut agree_count: u32 = 0;
    for _ in 0..num_runs {
        let mut pair = System::<2>::new(Tensor([FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2]));

        let mut order = [0, 1];
        order.shuffle(&mut rng);

        let angle1 = measurement_angles.choose(&mut rng).unwrap();
        let angle2 = measurement_angles.choose(&mut rng).unwrap();

        let a = pair
            .measure(order[0], &angle1.as_tensor().tensor_mul(angle1.as_tensor()))
            .unwrap();
        let b = pair
            .measure(order[1], &angle2.as_tensor().tensor_mul(angle2.as_tensor()))
            .unwrap();

        if a == b {
            agree_count += 1;
        }
    }

    println!("Agree ratio: {}", agree_count as f64 / num_runs as f64);

    Ok(())
}
