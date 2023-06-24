use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Matrix(pub Vec<Vec<f64>>);

#[derive(Serialize, Deserialize)]
pub struct Vector(pub Vec<f64>);

impl Vector {
    pub fn new(size: usize) -> Self {
        let mut vector = Vec::new();
        vector.resize(size, 0.0);
        Vector(vector)
    }
    pub fn multiply(&self, matrix: &Matrix) -> Vector {
        let matrix_cols = matrix.0[0].len();
        let vector_length = matrix.0.len();
        let mut result = Vector::new(vector_length);

        for j in 0..vector_length {
            for c in 0..matrix_cols {
                result.0[j] += self.0[c] * matrix.0[j][c];
            }
        }
        result
    }
    pub fn invert_multiply(&self, matrix: &Matrix) -> Vector {
        let matrix_cols = matrix.0[0].len();
        let vector_length = matrix.0.len();
        let mut result = Vector::new(vector_length);

        for j in 0..vector_length {
            for c in 0..matrix_cols {
                result.0[j] += self.0[c] * matrix.0[c][j];
            }
        }
        result
    }

    pub fn activate(&mut self, activation_fn: &mut dyn FnMut(&f64) -> f64) {
        let size = self.0.len();
        for i in 0..size {
            self.0[i] = activation_fn(&self.0[i]);
        }
    }
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Self {
        let mut column = Vec::new();
        column.resize(columns, 0.0);
        let mut matrix = Vec::new();
        matrix.resize(rows, column);

        let mut rng = rand::thread_rng();
        for x in 0..rows {
            for y in 0..columns {
                matrix[x][y] = rng.gen();
            }
        }
        Matrix(matrix)
    }
}
