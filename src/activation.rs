use serde::{Deserialize, Serialize};

pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Serialize, Deserialize)]
pub enum ActivationFn {
    Sigmoid,
}
