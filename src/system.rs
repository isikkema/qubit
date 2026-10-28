use std::{array, collections::HashMap, gca};

use rand::{RngExt, rng};

use crate::{
    dot::Dot, inner_mul::InnerMul, qubit::Qubit, random_bool::RandomBool, tensor::Tensor,
    tensor_mul::TensorMul, transposable::Transposable,
};

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct QubitId(u128);

impl QubitId {
    fn new() -> Self {
        Self(rng().random())
    }

    fn clone(&self) -> Self {
        Self(self.0)
    }
}

const DIM<const Q: u8>: usize = const { 1 << Q };
const R<const Q: u8>: u8 = const { Q - 1 };
const P<const Q: u8>: u8 = const { Q + 1 };

pub struct System<const Q: u8> {
    pub state: Tensor<f64, gca!(DIM::<Q>)>,
    id_map: HashMap<QubitId, usize>,
}

impl System<0> {
    pub fn new() -> System<0> {
        System::from_state(Tensor([1.0])).0
    }
}

impl Default for System<0> {
    fn default() -> Self {
        Self::new()
    }
}

impl System<1> {
    pub fn from_qubit(qubit: Qubit) -> (Self, QubitId) {
        let (system, mut ids) = Self::from_state(qubit.get_state().clone());
        (system, ids.pop().unwrap())
    }
}

impl<const Q: u8> System<Q> {
    pub fn from_state(state: Tensor<f64, gca!(DIM::<Q>)>) -> (Self, Vec<QubitId>) {
        let mut id_map = HashMap::with_capacity(Q.into());
        let mut ids = Vec::with_capacity(Q.into());
        for idx in 0..Q.into() {
            let id = QubitId::new();
            id_map.insert(id.clone(), idx);
            ids.push(id);
        }

        (Self { state, id_map }, ids)
    }

    pub fn get_ids(&self) -> &HashMap<QubitId, usize> {
        &self.id_map
    }

    pub fn measure<R: RandomBool>(
        mut self,
        id: QubitId,
        basis: &Tensor<Tensor<f64, gca!(DIM::<Q>)>, gca!(DIM::<Q>)>,
        rng: &mut R,
    ) -> Option<(bool, System<gca!(R::<Q>)>, (f64, f64))> {
        let qubit_idx = self.id_map.remove(&id)?;
        self.id_map.iter_mut().for_each(|(_, v)| {
            if *v > qubit_idx {
                *v -= 1;
            }
        });

        self.state = basis.T().dot(&self.state);

        let off_state = Tensor(array::from_fn(|idx| self.state[idx << qubit_idx]));
        let on_state = Tensor(array::from_fn(|idx| {
            self.state[(self.state.len() >> (qubit_idx + 1)) + (idx << qubit_idx)]
        }));

        debug_assert!(self.state.len() % 2 == 0);

        let p_off: f64 = off_state.as_array().iter().map(|n| n.powi(2)).sum();
        let p_on: f64 = on_state.as_array().iter().map(|n| n.powi(2)).sum();

        let is_on = rng.random_bool(p_on);

        let new_system = if !is_on {
            self.measure_off(p_off, off_state)
        } else {
            self.measure_on(p_on, on_state)
        };

        Some((is_on, new_system, (p_off, p_on)))
    }

    pub fn insert(self, qubit: Qubit) -> (QubitId, System<gca!(P::<Q>)>) {
        let Self { state, mut id_map } = self;

        let mut new_state = Tensor::default();
        new_state
            .0
            .copy_from_slice(state.tensor_mul(qubit.get_state()).as_array());

        let id = QubitId::new();
        id_map.insert(id.clone(), Q as usize);

        (
            id,
            System {
                state: new_state,
                id_map,
            },
        )
    }

    fn measure_off(
        self,
        p: f64,
        off_state: Tensor<f64, gca!(DIM::<gca!(R::<Q>)>)>,
    ) -> System<gca!(R::<Q>)> {
        self.collapse_state(p, off_state)
    }

    fn measure_on(
        self,
        p: f64,
        on_state: Tensor<f64, gca!(DIM::<gca!(R::<Q>)>)>,
    ) -> System<gca!(R::<Q>)> {
        self.collapse_state(p, on_state)
    }

    fn collapse_state(
        self,
        p: f64,
        relevant_state: Tensor<f64, gca!(DIM::<gca!(R::<Q>)>)>,
    ) -> System<gca!(R::<Q>)> {
        System {
            state: (1.0 / p.sqrt()).inner_mul(&relevant_state),
            id_map: self.id_map,
        }
    }
}

impl From<Qubit> for (System<1>, QubitId) {
    fn from(value: Qubit) -> Self {
        let (system, mut ids) = System::from_state(value.get_state().clone());
        (system, ids.pop().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::{basis::DEG_0, random_bool::StaticBool, tensor::mostly_eq};

    use super::*;

    #[test]
    fn test_from_qubit() {
        let qubit = Qubit::new(1.0 / 4.0f64.sqrt(), 3.0f64.sqrt() / 4.0f64.sqrt());

        let (system, _) = System::from_qubit(qubit);

        assert_eq!(
            system.state,
            Tensor([1.0 / 4.0f64.sqrt(), 3.0f64.sqrt() / 4.0f64.sqrt()])
        );

        assert_eq!(system.get_ids().values().collect::<Vec<_>>(), &[&0]);
    }

    #[test]
    fn test_insert() {
        let qubit1 = Qubit::new(1.0 / 4f64.sqrt(), 3f64.sqrt() / 4f64.sqrt());
        let qubit2 = Qubit::new(1.0 / 3f64.sqrt(), 2f64.sqrt() / 3f64.sqrt());

        let system = System::new();
        let (_, system) = system.insert(qubit1);
        let (_, system) = system.insert(qubit2);

        assert!(mostly_eq(
            &system.state,
            &Tensor([
                1.0 / 12f64.sqrt(),
                1.0 / 6f64.sqrt(),
                1.0 / 4f64.sqrt(),
                1.0 / 2f64.sqrt()
            ])
        ));

        assert_eq!(
            system.get_ids().values().collect::<HashSet<_>>(),
            HashSet::from([&0, &1])
        );
    }

    #[test]
    fn test_measure_unentangled() {
        // Measure 1 first as false
        let qubit1 = Qubit::new(1.0 / 4f64.sqrt(), 3f64.sqrt() / 4f64.sqrt());
        let qubit2 = Qubit::new(1.0 / 3f64.sqrt(), 2f64.sqrt() / 3f64.sqrt());

        let expected = qubit2.get_state().clone();

        let system = System::new();
        let (id1, system) = system.insert(qubit1);
        let (id2, system) = system.insert(qubit2);

        let basis = DEG_0.as_tensor().tensor_mul(DEG_0.as_tensor());

        let (value, system, (p_off, p_on)) =
            system.measure(id1, &basis, &mut StaticBool(false)).unwrap();
        assert_eq!(value, false);
        assert!(mostly_eq(&system.state, &expected));
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 1.0 / 4.0 < 0.00001);
        assert!(p_on - 3.0 / 4.0 < 0.00001);

        let (value, _, (p_off, p_on)) = system
            .measure(id2, &DEG_0.as_tensor(), &mut StaticBool(false))
            .unwrap();
        assert_eq!(value, false);
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 1.0 / 3.0 < 0.00001);
        assert!(p_on - 2.0 / 3.0 < 0.00001);

        // Measure 1 first as true
        let qubit1 = Qubit::new(1.0 / 4f64.sqrt(), 3f64.sqrt() / 4f64.sqrt());
        let qubit2 = Qubit::new(1.0 / 3f64.sqrt(), 2f64.sqrt() / 3f64.sqrt());

        let system = System::new();
        let (id1, system) = system.insert(qubit1);
        let (id2, system) = system.insert(qubit2);

        let (value, system, (p_off, p_on)) =
            system.measure(id1, &basis, &mut StaticBool(true)).unwrap();
        assert_eq!(value, true);
        assert!(mostly_eq(&system.state, &expected));
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 1.0 / 4.0 < 0.00001);
        assert!(p_on - 3.0 / 4.0 < 0.00001);

        let (value, _, (p_off, p_on)) = system
            .measure(id2, &DEG_0.as_tensor(), &mut StaticBool(true))
            .unwrap();
        assert_eq!(value, true);
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 1.0 / 3.0 < 0.00001);
        assert!(p_on - 2.0 / 3.0 < 0.00001);

        // Measure 2 first as false
        let qubit1 = Qubit::new(1.0 / 4f64.sqrt(), 3f64.sqrt() / 4f64.sqrt());
        let qubit2 = Qubit::new(1.0 / 3f64.sqrt(), 2f64.sqrt() / 3f64.sqrt());

        let expected = qubit1.get_state().clone();

        let system = System::new();
        let (id1, system) = system.insert(qubit1);
        let (id2, system) = system.insert(qubit2);

        let (value, system, (p_off, p_on)) =
            system.measure(id2, &basis, &mut StaticBool(false)).unwrap();
        assert_eq!(value, false);
        assert!(mostly_eq(&system.state, &expected));
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 1.0 / 3.0 < 0.00001);
        assert!(p_on - 2.0 / 3.0 < 0.00001);

        let (value, _, (p_off, p_on)) = system
            .measure(id1, &DEG_0.as_tensor(), &mut StaticBool(false))
            .unwrap();
        assert_eq!(value, false);
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 1.0 / 4.0 < 0.00001);
        assert!(p_on - 3.0 / 4.0 < 0.00001);

        // Measure 2 first as true
        let qubit1 = Qubit::new(1.0 / 4f64.sqrt(), 3f64.sqrt() / 4f64.sqrt());
        let qubit2 = Qubit::new(1.0 / 3f64.sqrt(), 2f64.sqrt() / 3f64.sqrt());

        let system = System::new();
        let (id1, system) = system.insert(qubit1);
        let (id2, system) = system.insert(qubit2);

        let (value, system, (p_off, p_on)) =
            system.measure(id2, &basis, &mut StaticBool(true)).unwrap();
        assert_eq!(value, true);
        assert!(mostly_eq(&system.state, &expected));
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 1.0 / 3.0 < 0.00001);
        assert!(p_on - 2.0 / 3.0 < 0.00001);

        let (value, _, (p_off, p_on)) = system
            .measure(id1, &DEG_0.as_tensor(), &mut StaticBool(true))
            .unwrap();
        assert_eq!(value, true);
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 1.0 / 4.0 < 0.00001);
        assert!(p_on - 3.0 / 4.0 < 0.00001);
    }

    #[test]
    fn test_measure_entangled() {
        // Measure 1 first as false
        let (system, mut ids) = System::<2>::from_state(Tensor([
            1.0 / 2f64.sqrt(),
            1.0 / 4f64.sqrt(),
            1.0 / 8f64.sqrt(),
            1.0 / 8f64.sqrt(),
        ]));

        let id2 = ids.pop().unwrap();
        let id1 = ids.pop().unwrap();

        let basis = DEG_0.as_tensor().tensor_mul(DEG_0.as_tensor());

        let (value, system, (p_off, p_on)) =
            system.measure(id1, &basis, &mut StaticBool(false)).unwrap();
        assert_eq!(value, false);
        assert!(mostly_eq(
            &system.state,
            &Tensor([2f64.sqrt() / 3f64.sqrt(), 1.0 / 3f64.sqrt()])
        ));
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 3.0 / 4.0 < 0.00001);
        assert!(p_on - 1.0 / 4.0 < 0.00001);

        let (value, _, (p_off, p_on)) = system
            .measure(id2, &DEG_0.as_tensor(), &mut StaticBool(false))
            .unwrap();
        assert_eq!(value, false);
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 2.0 / 3.0 < 0.00001);
        assert!(p_on - 1.0 / 3.0 < 0.00001);

        // Measure 1 first as true
        let (system, mut ids) = System::<2>::from_state(Tensor([
            1.0 / 2f64.sqrt(),
            1.0 / 4f64.sqrt(),
            1.0 / 8f64.sqrt(),
            1.0 / 8f64.sqrt(),
        ]));

        let id2 = ids.pop().unwrap();
        let id1 = ids.pop().unwrap();

        let (value, system, (p_off, p_on)) =
            system.measure(id1, &basis, &mut StaticBool(true)).unwrap();
        assert_eq!(value, true);
        assert!(mostly_eq(
            &system.state,
            &Tensor([1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt()])
        ));
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 3.0 / 4.0 < 0.00001);
        assert!(p_on - 1.0 / 4.0 < 0.00001);

        let (value, _, (p_off, p_on)) = system
            .measure(id2, &DEG_0.as_tensor(), &mut StaticBool(true))
            .unwrap();
        assert_eq!(value, true);
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 1.0 / 2.0 < 0.00001);
        assert!(p_on - 1.0 / 2.0 < 0.00001);

        // Measure 2 first as false
        let (system, mut ids) = System::<2>::from_state(Tensor([
            1.0 / 2f64.sqrt(),
            1.0 / 4f64.sqrt(),
            1.0 / 8f64.sqrt(),
            1.0 / 8f64.sqrt(),
        ]));

        let id2 = ids.pop().unwrap();
        let id1 = ids.pop().unwrap();

        let basis = DEG_0.as_tensor().tensor_mul(DEG_0.as_tensor());

        let (value, system, (p_off, p_on)) =
            system.measure(id2, &basis, &mut StaticBool(false)).unwrap();
        assert_eq!(value, false);
        assert!(mostly_eq(
            &system.state,
            &Tensor([4f64.sqrt() / 5f64.sqrt(), 1.0 / 5f64.sqrt()])
        ));
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 5.0 / 8.0 < 0.00001);
        assert!(p_on - 3.0 / 8.0 < 0.00001);

        let (value, _, (p_off, p_on)) = system
            .measure(id1, &DEG_0.as_tensor(), &mut StaticBool(false))
            .unwrap();
        assert_eq!(value, false);
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 4.0 / 5.0 < 0.00001);
        assert!(p_on - 1.0 / 5.0 < 0.00001);

        // Measure 2 first as true
        let (system, mut ids) = System::<2>::from_state(Tensor([
            1.0 / 2f64.sqrt(),
            1.0 / 4f64.sqrt(),
            1.0 / 8f64.sqrt(),
            1.0 / 8f64.sqrt(),
        ]));

        let id2 = ids.pop().unwrap();
        let id1 = ids.pop().unwrap();

        let (value, system, (p_off, p_on)) =
            system.measure(id2, &basis, &mut StaticBool(true)).unwrap();
        assert_eq!(value, true);
        assert!(mostly_eq(
            &system.state,
            &Tensor([2f64.sqrt() / 3f64.sqrt(), 1.0 / 3f64.sqrt()])
        ));
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 5.0 / 8.0 < 0.00001);
        assert!(p_on - 3.0 / 8.0 < 0.00001);

        let (value, _, (p_off, p_on)) = system
            .measure(id1, &DEG_0.as_tensor(), &mut StaticBool(true))
            .unwrap();
        assert_eq!(value, true);
        assert!(1.0 - (p_off + p_on) < 0.00001);
        assert!(p_off - 2.0 / 3.0 < 0.00001);
        assert!(p_on - 1.0 / 3.0 < 0.00001);
    }
}
