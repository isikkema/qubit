use crate::{basis::Basis, qubit::Qubit};

pub struct Filter {
    basis: Basis,
    allow: bool,
    pub num_passed: u64,
    pub num_total: u64,
}

impl Filter {
    pub fn new(basis: Basis, allow: bool) -> Self {
        Filter {
            basis,
            allow,
            num_passed: 0,
            num_total: 0,
        }
    }

    pub fn filter(&mut self, mut qb: Qubit) -> Option<Qubit> {
        self.num_total += 1;

        if qb.measure(self.basis.clone()) == self.allow {
            self.num_passed += 1;

            Some(qb)
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.num_passed = 0;
        self.num_total = 0;
    }
}
