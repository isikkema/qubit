use rand::RngExt;

pub trait RandomBool {
    fn random_bool(&mut self, p: f64) -> bool;
}

impl<T: RngExt> RandomBool for T {
    fn random_bool(&mut self, p: f64) -> bool {
        <T as RngExt>::random_bool(self, p)
    }
}

#[cfg(test)]
pub struct StaticBool(pub bool);

#[cfg(test)]
impl RandomBool for StaticBool {
    fn random_bool(&mut self, _p: f64) -> bool {
        self.0
    }
}
