pub fn flatten_arrays<T: Clone, const N: usize, const M: usize>(arrays: [[T; M]; N]) -> [T; N * M] {
    if let Ok(flat_array) = arrays.concat().try_into() {
        flat_array
    } else {
        unreachable!()
    }
}
