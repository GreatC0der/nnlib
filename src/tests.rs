use crate::math::{Matrix, Vector};

#[test]
fn multiplication() {
    let expected_result = vec![0.0, -3.0, -6.0, -9.0];
    let vector = Vector(vec![-2.0, 1.0, 0.0]);
    let matrix = Matrix(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![10.0, 11.0, 12.0],
    ]);
    assert_eq!(expected_result, vector.multiply(&matrix).0)
}

#[test]
fn inverted_multiplication() {
    let vector = Vector(vec![1.0, 2.0, 3.0, 4.0]);
    let matrix = Matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let inverted_matrix = Matrix(vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    assert_eq!(
        vector.multiply(&inverted_matrix).0,
        vector.invert_multiply(&matrix).0
    )
}
