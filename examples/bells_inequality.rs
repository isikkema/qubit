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
        let (system, mut ids) =
            System::<2>::from_state(Tensor([FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2]));

        ids.shuffle(&mut rng);

        let id1 = ids.pop().unwrap();
        let id2 = ids.pop().unwrap();

        let angle1 = measurement_angles.choose(&mut rng).unwrap();
        let angle2 = measurement_angles.choose(&mut rng).unwrap();

        let (value1, system, _) = system
            .measure(
                id1,
                &angle1.as_tensor().tensor_mul(angle1.as_tensor()),
                &mut rng,
            )
            .unwrap();
        let (value2, _, _) = system.measure(id2, &angle2.as_tensor(), &mut rng).unwrap();

        if value1 == value2 {
            agree_count += 1;
        }
    }

    println!("Agree ratio: {}", agree_count as f64 / num_runs as f64);

    Ok(())
}
