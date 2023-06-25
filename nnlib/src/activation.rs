use serde::{Deserialize, Serialize};

pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn leaky_relu(x: &f64) -> f64 {
    if *x >= 0.0 {
        *x
    } else {
        x * 0.1
    }
}

#[derive(Serialize, Deserialize)]
pub enum ActivationFn {
    Sigmoid,
    LeakyRELU,
}
