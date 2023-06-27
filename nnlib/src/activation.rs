use serde::{Deserialize, Serialize};

pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: &f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn leaky_relu(x: &f64) -> f64 {
    if *x >= 0.0 {
        *x
    } else {
        x * 0.1
    }
}

pub fn no_derivative(_x: &f64) -> f64 {
    1.0
}

#[derive(Serialize, Deserialize)]
pub enum ActivationFn {
    Sigmoid,
    LeakyRELU,
}
