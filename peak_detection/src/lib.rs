use pyo3::prelude::*;
// mod rust_peak_detection_locally_exclusive;
mod rust_peak_detection_locally_exclusive_sliding_window;
// mod rust_peak_detection_locally_exclusive_sam;
// mod rust_peak_detection_locally_exclusive_sam2;

#[pymodule]
pub fn peak_detection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(rust_peak_detection_locally_exclusive::detect_peaks_rust_locally_exclusive_on_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(rust_peak_detection_locally_exclusive_sliding_window::detect_peaks_rust_locally_exclusive_on_chunk, m)?)?;
    // m.add_function(wrap_pyfunction!(rust_peak_detection_locally_exclusive_sam::detect_peaks_rust_locally_exclusive_on_chunk, m)?)?;
    // m.add_function(wrap_pyfunction!(rust_peak_detection_locally_exclusive_sam2::detect_peaks_rust_locally_exclusive_on_chunk, m)?)?;
    Ok(())
}