use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::time::Instant;

#[pyfunction]
pub fn detect_peaks_rust_locally_exclusive_on_chunk<'py>(py: Python<'py>, traces: PyReadonlyArray2<f32>, peak_sign: &str, abs_thresholds: PyReadonlyArray1<f32>, exclude_sweep_size: usize, neighbours_mask: PyReadonlyArray2<bool>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<usize>>) {
    assert!(["pos", "neg", "both"].contains(&peak_sign), "peak_sign must be 'pos', 'neg', or 'both'");

    let data: ArrayView2<f32> = traces.as_array();
    let abs_thresholds: ArrayView1<f32> = abs_thresholds.as_array();
    let neighbours_mask: ArrayView2<bool> = neighbours_mask.as_array();

    let peaks = detect_peaks_locally_exclusive(&data, peak_sign, &abs_thresholds, exclude_sweep_size, &neighbours_mask);

    (peaks.0.into_pyarray(py), peaks.1.into_pyarray(py))
}

fn detect_peaks_locally_exclusive(data : &ArrayView2<f32>, peak_sign: &str, abs_thresholds: &ArrayView1<f32>, exclude_sweep_size: usize, neighbours_mask: &ArrayView2<bool>) -> (Vec<usize>, Vec<usize>) {

    let n_samples = data.nrows();
    if n_samples == 0 {
        return (vec![], vec![]);
    }

    use ndarray::s;
    let data_center = data.slice(s![exclude_sweep_size..n_samples-exclude_sweep_size, ..]);
    let n_samples_center = data_center.nrows();

    let mut peak_mask : Array2<bool> = Array2::from_elem((n_samples_center, data.ncols()), false);

    if ["pos","both"].contains(&peak_sign) {
        // Create the peak mask by comparing each value to the threshold for its channel
        for ((i, j), &value) in data_center.indexed_iter() {
            if value > abs_thresholds[j] {
                peak_mask[[i, j]] = true;
            }
            else {
                peak_mask[[i, j]] = false;
            }
        }
        peak_mask = remove_neighboring_peaks(&peak_mask, &data,&data_center, &neighbours_mask, exclude_sweep_size,"pos");
    }

    if ["neg","both"].contains(&peak_sign) {
        let mut peak_mask_pos: Array2<bool> = Array2::from_elem((n_samples_center, data.ncols()), false);
        if peak_sign == "both" {
            peak_mask_pos = peak_mask.clone();
        }

        for ((i, j), &value) in data_center.indexed_iter() {
            if value < -abs_thresholds[j] {
                peak_mask[[i, j]] = true;
            }
            else {
                peak_mask[[i, j]] = false;
            }
        }
        peak_mask = remove_neighboring_peaks(&peak_mask, &data,&data_center, &neighbours_mask, exclude_sweep_size,"neg");

        if peak_sign == "both" {
            peak_mask = peak_mask | peak_mask_pos;
        }
    }

    let result: (Vec<usize>, Vec<usize>) = peak_mask.indexed_iter()
        .filter_map(|((i, j), &is_peak)| if is_peak { Some((i + exclude_sweep_size, j)) } else { None })
        .unzip();

    result
}


fn remove_neighboring_peaks(peak_mask: &Array2<bool>, data: &ArrayView2<f32>, data_center: &ArrayView2<f32>, neighbours_mask: &ArrayView2<bool>, exclude_sweep_size: usize, peak_sign: &str) -> Array2<bool> {
    assert!(["pos", "neg"].contains(&peak_sign), "peak_sign must be 'pos' or 'neg'");

    let sign:f32 = if peak_sign == "pos" { 1.0 } else { -1.0 };
    let num_channels = data.ncols();
    let num_samples = data_center.nrows();
    let mut result_peak_mask = peak_mask.clone();
    for chan_ind in 0..num_channels{
        for s in 0..num_samples{
            if !result_peak_mask[[s, chan_ind]] {
                continue;
            }
            for neighbour in 0..num_channels{
                if !neighbours_mask[[chan_ind, neighbour]]{
                    continue;
                }
                for i in 0..exclude_sweep_size{
                    if chan_ind != neighbour{
                            result_peak_mask[[s, chan_ind]] &= data_center[[s, chan_ind]]*sign >= data_center[[s, neighbour]]*sign;
                    }
                    result_peak_mask[[s, chan_ind]] &= data_center[[s, chan_ind]]*sign > data[[s + i, neighbour]]*sign;
                    result_peak_mask[[s, chan_ind]] &= data_center[[s, chan_ind]]*sign >= data[[exclude_sweep_size + s + i + 1, neighbour]]*sign;
                    if !result_peak_mask[[s, chan_ind]] {
                        break;
                    }
                }
                if !result_peak_mask[[s, chan_ind]] {
                    break;
                }
            }
        }
    }
    result_peak_mask
}