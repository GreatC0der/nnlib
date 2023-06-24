pub mod activation;
pub mod math;

#[cfg(test)]
pub mod tests;

use std::{io::Write, path::Path};

use serde::{Deserialize, Serialize};

use math::{Matrix, Vector};

use activation::*;

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    weights: Vec<Matrix>,
    values: Vec<Vector>,
    errors: Vec<Vector>,
    learning_coefficient: f64,
    activation_fn: ActivationFn,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>, learning_coefficient: f64, activation_fn: ActivationFn) -> Self {
        // creating weights
        let mut weights = Vec::new();
        for index in 0..layers.len() - 1 {
            weights.push(Matrix::new(layers[index + 1], layers[index]));
        }
        // defining values and errors
        let mut values = Vec::new();
        let mut errors = Vec::new();
        for layer_size in layers {
            values.push(Vector::new(layer_size));
            errors.push(Vector::new(layer_size));
        }

        NeuralNetwork {
            weights,
            values,
            errors,
            learning_coefficient,
            activation_fn,
        }
    }

    pub fn run(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.values[0] = Vector(input);

        let mut activation_fn = match self.activation_fn {
            ActivationFn::Sigmoid => sigmoid,
        };

        let layers = self.weights.len();
        for layer in 0..layers {
            self.values[layer].activate(&mut activation_fn);
            self.values[layer + 1] = self.values[layer].multiply(&self.weights[layer]);
        }
        self.values[layers].0.clone()
    }

    pub fn teach(&mut self, input: Vec<f64>, expected_output: Vec<f64>) -> f64 {
        let output = self.run(input);
        let mut overall_error = 0.0;

        // Calculate errors for the output layer
        let last_layer = self.errors.len() - 1;
        for node in 0..output.len() {
            self.errors[last_layer].0[node] = output[node] - expected_output[node];
            overall_error += self.errors[last_layer].0[node].abs();
        }

        // Calculate errors for the all layers
        let layers = self.errors.len();
        for layer in 1..layers {
            self.errors[layer - 1] = self.errors[layer].invert_multiply(&self.weights[layer - 1]);
        }

        // Change weights
        self.change_weights();
        overall_error
    }

    fn change_weights(&mut self) {
        let layers = self.weights.len();

        for layer in 0..layers {
            let rows = self.weights[layer].0.len();
            let cols = self.weights[layer].0[0].len();

            for x in 0..rows {
                for y in 0..cols {
                    self.weights[layer].0[x][y] -= self.errors[layer].0[y]
                        * self.values[layer].0[y]
                        * self.learning_coefficient;
                }
            }
        }
    }

    pub fn save(&self, path: &Path) {
        let data = serde_json::to_string(&self).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(data.as_bytes()).unwrap();
    }

    pub fn load(path: &Path) -> Self {
        let data = std::fs::read_to_string(path).unwrap();
        serde_json::from_str(&data).unwrap()
    }
}
