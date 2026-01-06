use ndarray::{Array1, Array2, ArrayView2};
use numpy::{PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (data, channel_locations, sampling_frequency, detect_threshold=5.0, peak_sign="neg", exclude_sweep_ms=0.1, radius=50.0))]
pub fn rust_peak_detection_locally_exclusive(data: PyReadonlyArray2<f32>, channel_locations: PyReadonlyArray2<f32>, sampling_frequency: f32, detect_threshold: f32, peak_sign: &str, exclude_sweep_ms: f32, radius: f32) -> Vec<(usize, usize)> {
    let data: ArrayView2<f32> = data.as_array();
    let channel_locations: ArrayView2<f32> = channel_locations.as_array();

    let num_channels = data.shape()[0];
    let noise_levels:Array1<f32> = Array1::from_elem(num_channels, 12.0); // We use a fixed noise level for all channels for now
    let abs_thresholds:Array1<f32> = noise_levels * detect_threshold;
    let exclude_sweep_size = (exclude_sweep_ms * sampling_frequency / 1000.0) as usize;

    let peaks = detect_peaks_locally_exclusive(&data, &channel_locations, peak_sign, &abs_thresholds, exclude_sweep_size, radius);

    peaks
}

fn detect_peaks_locally_exclusive(data : &ArrayView2<f32>, channel_locations: &ArrayView2<f32>, peak_sign: &str, abs_thresholds: &Array1<f32>, exclude_sweep_size: usize, radius : f32) -> Vec<(usize, usize)> {
    assert!(["pos", "neg", "both"].contains(&peak_sign), "peak_sign must be 'pos', 'neg', or 'both'");

    let n_samples = data.nrows();
    if n_samples == 0 {
        return vec![];
    }

    use ndarray::s;
    let data_center = data.slice(s![exclude_sweep_size..n_samples-exclude_sweep_size, ..]);
    let n_samples_center = data_center.nrows();

    let channel_distance = get_channel_distance(channel_locations);
    let neighbour_mask = channel_distance.mapv(|d| d <= radius); // We consider channels within 50 microns as neighbors
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

        peak_mask = remove_neighboring_peaks(&peak_mask, &data,&data_center, &neighbour_mask, exclude_sweep_size,"pos");
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

        peak_mask = remove_neighboring_peaks(&peak_mask, &data,&data_center, &neighbour_mask, exclude_sweep_size,"neg");

        if peak_sign == "both" {
            peak_mask = peak_mask | peak_mask_pos;
        }
    }

    let result: Vec<(usize, usize)> = peak_mask.indexed_iter()
        .filter_map(|((i, j), &is_peak)| if is_peak { Some((i + exclude_sweep_size, j)) } else { None })
        .collect();

    result
}

fn get_channel_distance(channel_locations: &ArrayView2<f32>) -> Array2<f32> {
    let n_channels = channel_locations.nrows();
    let mut distance = Array2::zeros((n_channels, n_channels));
    for i in 0..n_channels {
        for j in 0..n_channels {
            distance[[i, j]] = ((channel_locations[[i, 0]] - channel_locations[[j, 0]]).powi(2) + (channel_locations[[i, 1]] - channel_locations[[j, 1]]).powi(2)).sqrt();
        }
    }
    distance
}

fn remove_neighboring_peaks(peak_mask: &Array2<bool>, data: &ArrayView2<f32>, data_center: &ArrayView2<f32>, neighbour_mask: &Array2<bool>, exclude_sweep_size: usize, peak_sign: &str) -> Array2<bool> {
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
                if !neighbour_mask[[chan_ind, neighbour]]{
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