mod image_utils;
use std::env;
use std::path::Path;

use nnlib::NeuralNetwork;

fn main() {
    let args: Vec<String> = env::args().collect();

    match args[1].as_str() {
        "teach" => teach(),
        "run" => run(args[2].clone()),
        _ => println!("Unknown argument. \n Args:\nteach\nrun `path to file.png`"),
    }
}

fn run(path: String) {
    let mut neural_network = NeuralNetwork::load(Path::new("neural_network"));

    let output = neural_network.run(image_utils::image_to_array(&Path::new(&path)));

    // Printing results.

    println!("Neural network's output:");
    for number in output.iter() {
        println!("{}", number);
    }

    // (Probability, number)
    let mut result: (f64, usize) = (0.0, 0);

    for number in 0..output.len() {
        if output[number] > result.0 {
            result = (output[number], number);
        }
    }

    println!("Possibly the answer is {}", result.1);
}

fn teach() {
    let data_set = image_utils::load_images(3);

    let mut neural_network = NeuralNetwork::new(
        vec![16 * 16, 8 * 8, 4 * 4, 10],
        0.1,
        nnlib::activation::ActivationFn::Sigmoid,
    );

    let expected_output = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];

    for generation in 0..10 {
        let mut error = 0.0;
        for data in data_set.iter() {
            error += neural_network.teach(data.0.clone(), expected_output[data.1.clone()].clone());
        }
        println!("Generation: {}, error: {}", generation, error);
    }

    neural_network.save(Path::new("neural_network"));
    println!("Neural network trained and saved.");
}
