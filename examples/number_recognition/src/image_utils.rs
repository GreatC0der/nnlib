use std::path::Path;

/// Loads specified number of data groups(folders: 0, 1, 2.... in dataset folder) from the dataset folder, converts them into appropriate form.
/// Returns - Vector of images and numbers on this images
pub fn load_images(data_groups: usize) -> Vec<(Vec<f64>, usize)> {
    let mut result = Vec::new();
    result.reserve_exact(data_groups * 10);

    for data_group in 0..data_groups {
        for number in 0..10 {
            result.push((
                image_to_array(
                    &Path::new("dataset")
                        .join(data_group.to_string())
                        .join(number.to_string() + ".png"),
                ),
                number,
            ));
        }
    }

    result
}

pub fn image_to_array(path: &Path) -> Vec<f64> {
    let img = image::open(path).unwrap();
    img.to_rgb32f()
        .into_vec()
        .iter()
        .copied()
        .map(|x| x as f64 / 255.0)
        .collect()
}
