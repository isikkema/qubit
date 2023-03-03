pub trait Transposable {
    type Output: Clone + Transposable;

    #[allow(non_snake_case)]
    fn T(&self) -> Self::Output;
}

impl Transposable for f64 {
    type Output = f64;

    fn T(&self) -> Self::Output {
        self.clone()
    }
}
