use pyo3::prelude::*;

use std::fs::File;
use std::io::{self, Read};
use ndarray::Array2;
use numpy::PyArray2;

use serde::{Deserialize, Serialize};
use serde_json;

#[pyclass]
pub struct FileChunkIterator {
    file: File,
    chunk_size: usize,
    num_channel: usize
}

#[pymethods]
impl FileChunkIterator {
    #[new]
    fn new(path: &str, chunk_size: usize, num_channel: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self { file, chunk_size, num_channel })
    }

    fn __iter__(slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<PyResult<Py<PyArray2<f32>>>> {
        let mut buffer = vec![0; slf.chunk_size*slf.num_channel*4]; // 4 bytes per f32

        match slf.file.read(&mut buffer) {
            Ok(0) => None, // EOF
            Ok(n) => {
                buffer.truncate(n);

                // Convert bytes to f32
                let floats: Vec<f32> = buffer
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                
                // Convert Vec<f32> to Array2
                let array = Array2::from_shape_vec((n/(4 * slf.num_channel), slf.num_channel), floats)
                    .expect("Shape mismatch or not enough data");
                let array = PyArray2::from_owned_array(slf.py(), array);
                Some(Ok(array.into()))
            }
            Err(e) => Some(Err(e.into())),
        }
    }
}

impl Iterator for FileChunkIterator {
    type Item = io::Result<Array2<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buffer = vec![0; self.chunk_size*self.num_channel*4]; // 4 bytes per f32
        
        match self.file.read(&mut buffer) {
            Ok(0) => None, // EOF
            Ok(n) => {
                buffer.truncate(n);

                // Convert bytes to f32
                let floats: Vec<f32> = buffer
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                
                // Convert Vec<f32> to Array2
                let array = Array2::from_shape_vec((n/(4 * self.num_channel), self.num_channel), floats)
                    .expect("Shape mismatch or not enough data");
                Some(Ok(array))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_chunk_iterator() -> io::Result<()> {
        if !std::path::Path::new("./data/data.bin").exists() {
            save_test_data()?;
        }
        let iterator = FileChunkIterator::new("./data/data.bin", 10, 5)?; // 10 rows, 5 columns

    for chunk in iterator {
        match chunk {
            Ok(array) => {
                println!("Read chunk with shape: {:?}", array.dim());
                println!("Array contents:\n{:?}", array);
            }
            Err(e) => eprintln!("Error reading chunk: {}", e),
        }
    }

    Ok(())
    }

    fn save_test_data() -> io::Result<()> {
    use std::io::Write;
    let mut file = File::create("./data/data.bin")?;
    for i in 0..25 {
        for j in 0..5 {
            let value = (i * j) as f32;
            file.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct Probes{
    contact_positions: Vec<Vec<f32>>
}

#[derive(Serialize, Deserialize)]
struct RootLocation{
    probes: Vec<Probes>
}

#[pyfunction]
pub fn retrieve_channel_locations(py:Python,path: &str) -> PyResult<Py<PyArray2<f32>>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let data: RootLocation = serde_json::from_reader(reader).unwrap();
    let location = &data.probes[0].contact_positions;
    let c_locations = PyArray2::from_vec2(py, location)?;
    Ok(c_locations.into())
}

#[cfg(test)]
mod test_channel_locations {
    use super::*;

    #[test]
    fn test_retrieve_channel_locations() -> io::Result<()> {
        Python::attach(|py| {
            let locations = retrieve_channel_locations(py, "./data/static/probe.json").unwrap();
            println!("Retrieved channel locations:\n{:?}", locations);
        });
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct Parameters{
    sampling_frequency: f32,
    num_channels: u64,
    file_paths: Vec<String>
}

#[derive(Serialize, Deserialize)]
struct RootParameters{
    kwargs: Parameters
}

#[pyfunction]
pub fn retrieve_parameters(path: &str) -> io::Result<(f32, u64, Vec<String>)> {
    let parameters_path = path.to_string() + "/binary.json";
    let file = std::fs::File::open(parameters_path)?;
    let reader = std::io::BufReader::new(file);
    let data: RootParameters = serde_json::from_reader(reader)?;
    Ok((data.kwargs.sampling_frequency as f32, data.kwargs.num_channels, data.kwargs.file_paths))
}