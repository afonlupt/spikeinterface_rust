use pyo3::prelude::*;
mod rust_peak_detection_locally_exclusive;

#[pymodule]
pub fn peak_detection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_peak_detection_locally_exclusive::detect_peaks_rust_locally_exclusive_on_chunk, m)?)?;
    Ok(())
}