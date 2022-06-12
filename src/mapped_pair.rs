#[derive(Hash, Copy, Clone)]
pub struct  MappedPair <T> {
    value1: T,
    value2: T,
}


impl <T: PartialEq> MappedPair <T> {
    pub fn new(value1: T, value2: T) -> Self {
        Self {value1, value2}
    }

    pub fn first(&self) -> &T {&self.value1}
    pub fn second(&self) -> &T {&self.value2}
}

impl <T: PartialEq> PartialEq for MappedPair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value1 == other.value1 && self.value2 == other.value2 ||
            (self.value1 == other.value2 && self.value2 == other.value1)

    }
}

impl <T: PartialEq> Eq for MappedPair<T> {}
