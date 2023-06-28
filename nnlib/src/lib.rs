pub mod activation;
pub mod math;

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
    /// Create a neural network.
    /// layers - vec![2, 3, 4] would mean 2 nodes in input layer, 3 nodes in hidden layer and 4 nodes in output layer.
    /// Bias nodes are added automaticly.
    pub fn new(layers: Vec<usize>, learning_coefficient: f64, activation_fn: ActivationFn) -> Self {
        // array `layers` but with bias node (+1)
        let layers_with_bias = {
            let mut temp = layers.clone();

            for i in 0..temp.len() - 1 {
                // Don't need a bias node on the last layer
                temp[i] += 1;
            }

            temp
        };

        // creating weights
        let mut weights = Vec::new();
        for index in 0..layers.len() - 1 {
            weights.push(Matrix::new(layers[index + 1], layers_with_bias[index]));
        }
        // defining values(to access values after forward propagation to complete backward propagation )
        // and defining errors
        let mut values = Vec::new();
        let mut errors = Vec::new();
        for layer in 0..layers.len() {
            values.push(Vector::new(layers[layer]));
            errors.push(Vector::new(layers_with_bias[layer]));
        }

        NeuralNetwork {
            weights,
            values,
            errors,
            learning_coefficient,
            activation_fn,
        }
    }

    /// Runs neural network, returns output.
    pub fn run(&mut self, input: Vec<f64>) -> Vec<f64> {
        // Setting input nodes to input
        self.values[0] = Vector(input);

        // Figuring out what activation function to use
        let mut activation_fn = match self.activation_fn {
            ActivationFn::Sigmoid => sigmoid,
            ActivationFn::LeakyRELU => leaky_relu,
        };

        // Forward propagation.
        let layers = self.weights.len();
        for layer in 0..layers {
            self.values[layer + 1] = self.values[layer].forwards(&self.weights[layer]);
            self.values[layer + 1].activate(&mut activation_fn);
        }

        // returning last layer as output.
        self.values[layers].0.clone()
    }

    // Train neural network. Sizes of `input` and `expected_output` are not checked! Return error(mean).
    pub fn teach(&mut self, input: Vec<f64>, expected_output: Vec<f64>) -> f64 {
        let output = self.run(input);
        let mut overall_error = 0.0;

        // Calculate errors for the output layer
        let last_layer = self.errors.len() - 1;
        for node in 0..output.len() {
            self.errors[last_layer].0[node] = output[node] - expected_output[node];
            overall_error += self.errors[last_layer].0[node].abs();
        }

        overall_error /= self.errors[last_layer].0.len() as f64;

        // Calculate errors for all layers
        let layers = self.errors.len();
        for layer in 1..layers {
            self.errors[layer - 1] = self.errors[layer].backwards(&self.weights[layer - 1]);
        }

        self.change_weights();

        overall_error
    }

    fn change_weights(&mut self) {
        let layers = self.weights.len();

        let derivative = match self.activation_fn {
            ActivationFn::Sigmoid => sigmoid_derivative,
            _ => no_derivative,
        };

        for layer in 0..layers {
            let rows = self.weights[layer].0.len();
            let cols = self.weights[layer].0[0].len() - 1;

            for x in 0..rows {
                for y in 0..cols {
                    self.weights[layer].0[x][y] -= self.errors[layer].0[y]
                        * self.values[layer].0[y]
                        * derivative(&self.values[layer].0[y])
                        * self.learning_coefficient;
                }
                // Don't forget about bias node.
                self.weights[layer].0[x][cols] -= self.errors[layer].0[cols]
                    * 1.0
                    * self.learning_coefficient
                    * sigmoid_derivative(&1.0);
            }
        }
    }

    /// Saves neural network to a file.
    pub fn save(&self, path: &Path) {
        let data = serde_json::to_string(&self).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(data.as_bytes()).unwrap();
    }

    /// Loads neural network from the file.
    pub fn load(path: &Path) -> Self {
        let data = std::fs::read_to_string(path).unwrap();
        serde_json::from_str(&data).unwrap()
    }
}
