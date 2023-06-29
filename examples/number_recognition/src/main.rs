mod image_utils;
use std::env;
use std::path::Path;

use nnlib::NeuralNetwork;

fn main() {
    let args: Vec<String> = env::args().collect();

    match args[1].as_str() {
        "teach" => teach(),
        "run" => run_from_disk(args[2].clone()),
        _ => println!("Unknown argument. \n Args:\nteach\nrun `path to file.png`"),
    }
}

fn run_from_disk(path: String) {
    let mut neural_network = NeuralNetwork::load(Path::new("neural_network"));
    let input = image_utils::image_to_array(&Path::new(&path));

    run_from_mem(&mut neural_network, input);
}

fn teach() {
    let data_set = image_utils::load_images(9);

    let mut neural_network = NeuralNetwork::new(
        vec![16 * 16, 8 * 8, 10],
        0.05,
        nnlib::formulas::ActivationFn::Sigmoid,
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

    'main_loop: for generation in 0..300 {
        let mut error = 0.0;
        for data in data_set.iter() {
            error += neural_network.teach(data.0.clone(), expected_output[data.1.clone()].clone());
        }
        println!(
            "Generation: {}, error: {}",
            generation,
            error / data_set.len() as f64
        );
        if generation % 10 == 0 {
            println!(" (t)est, (f)inish, (c)ontinue [default=c]");
            match read_line().as_str() {
                "t\n" => test(&mut neural_network, &data_set),
                "f\n" => break 'main_loop,
                _ => {}
            }
        }
    }

    neural_network.save(Path::new("neural_network"));
    println!("Neural network trained and saved.");
}

pub fn test(neural_network: &mut NeuralNetwork, data: &Vec<(Vec<f64>, usize)>) {
    for i in 0..10 {
        run_from_mem(neural_network, data[i].0.clone());
    }
}

pub fn run_from_mem(neural_network: &mut NeuralNetwork, input: Vec<f64>) {
    let output = neural_network.run(input);

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

pub fn read_line() -> String {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    input
}
