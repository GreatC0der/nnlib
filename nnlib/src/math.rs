use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Matrix(pub Vec<Vec<f64>>);

#[derive(Serialize, Deserialize)]
pub struct Vector(pub Vec<f64>);

impl Vector {
    /// Creates vector and fills it with 0s.
    pub fn new(size: usize) -> Self {
        let mut vector = Vec::new();
        vector.resize(size, 0.0);
        Vector(vector)
    }

    /// Multiplies the matrix by the vector. Cares about bias node.
    pub fn forwards(&self, matrix: &Matrix) -> Vector {
        let matrix_cols = matrix.0[0].len() - 1;
        let vector_length = matrix.0.len();
        let mut result = Vector::new(vector_length);

        for j in 0..vector_length {
            for c in 0..matrix_cols {
                result.0[j] += self.0[c] * matrix.0[j][c];
            }

            // Adding bias node.
            result.0[j] += 1.0 * matrix.0[j][matrix_cols];
        }

        result
    }

    /// Inverts and multiplies the matrix by the vector. Cares about bias node.
    pub fn backwards(&self, matrix: &Matrix) -> Vector {
        let matrix_cols = matrix.0.len() - 1;
        let vector_length = matrix.0[0].len();
        let mut result = Vector::new(vector_length);

        for j in 0..vector_length {
            for c in 0..matrix_cols {
                result.0[j] += self.0[c] * matrix.0[c][j];
            }
        }

        result
    }

    /// Applies activation_fn to every value in vector.
    pub fn activate(&mut self, activation_fn: &mut dyn FnMut(&f64) -> f64) {
        let size = self.0.len();
        for i in 0..size {
            self.0[i] = activation_fn(&self.0[i]);
        }
    }
}

impl Matrix {
    /// Creates matrix and fills it with random numbers.
    pub fn new(rows: usize, columns: usize) -> Self {
        let mut column = Vec::new();
        column.resize(columns, 0.0);
        let mut matrix = Vec::new();
        matrix.resize(rows, column);

        let mut rng = rand::thread_rng();
        for x in 0..rows {
            for y in 0..columns {
                matrix[x][y] = rng.gen::<f64>();
            }
        }
        Matrix(matrix)
    }
}
