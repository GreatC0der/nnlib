use std::path::Path;

use nnlib::NeuralNetwork;

fn main() {
    let path = Path::new("./nn.nnlib");
    let mut nn = NeuralNetwork::load(path);
    let inputs = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
    let expected_outputs = vec![vec![0.1, 0.9], vec![0.9, 0.1]];

    for _generations in 0..100 {
        for data_set in 0..2 {
            let error: f64 = nn.teach(inputs[data_set].clone(), expected_outputs[data_set].clone());
            println!("Error: {:.3}", error);
        }
    }

    println!("Input: 0.9, 0.1");
    let output = nn.run(vec![0.9, 0.1]);
    println!("Output: {:.3}, {:.3}", output[0], output[1]);

    println!("Input: 0.1, 0.9");
    let output = nn.run(vec![0.1, 0.9]);
    println!("Output: {:.3}, {:.3}", output[0], output[1]);
    nn.save(path);
}
