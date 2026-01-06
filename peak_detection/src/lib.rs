use pyo3::prelude::*;
mod rust_peak_detection_locally_exclusive;
mod data_file_reading;

#[pymodule]
pub fn peak_detection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_peak_detection_locally_exclusive::rust_peak_detection_locally_exclusive, m)?)?;

    m.add_function(wrap_pyfunction!(data_file_reading::retrieve_channel_locations, m)?)?;
    m.add_function(wrap_pyfunction!(data_file_reading::retrieve_parameters, m)?)?;

    m.add_class::<data_file_reading::FileChunkIterator>()?;
    Ok(())
}