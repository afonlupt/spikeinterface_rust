use std::collections::HashMap;

use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn detect_peaks_rust_locally_exclusive_on_chunk<'py>(py: Python<'py>, traces: PyReadonlyArray2<f32>, peak_sign: &str, abs_thresholds: PyReadonlyArray1<f32>, exclude_sweep_size: usize, neighbours_mask: PyReadonlyArray2<bool>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<usize>>) {
    assert!(["pos", "neg", "both"].contains(&peak_sign), "peak_sign must be 'pos', 'neg', or 'both'");

    let data: ArrayView2<f32> = traces.as_array();
    let abs_thresholds: ArrayView1<f32> = abs_thresholds.as_array();
    let neighbours_mask: ArrayView2<bool> = neighbours_mask.as_array();
    let adjency_list: Vec<Vec<usize>> = neighbours_mask.axis_iter(ndarray::Axis(0))
        .map(|row| row.indexed_iter()
            .filter_map(|(j, &is_neighbor)| if is_neighbor { Some(j) } else { None })
            .collect()
        )
        .collect();

    let peaks = detect_peaks_locally_exclusive(&data, peak_sign, &abs_thresholds, exclude_sweep_size, &adjency_list);

    (peaks.0.into_pyarray(py), peaks.1.into_pyarray(py))
}

fn detect_peaks_locally_exclusive(data : &ArrayView2<f32>, peak_sign: &str, abs_thresholds: &ArrayView1<f32>, exclude_sweep_size: usize, adjency_list: &Vec<Vec<usize>>) -> (Vec<usize>, Vec<usize>) {

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
        let mut potential_peaks: HashMap<usize,(usize, f32)> = HashMap::new();
        let mut exploring_neighboors:Vec<usize>=Vec::new();
        for ((i, j), &value) in data_center.indexed_iter() {
            if value <= abs_thresholds[j] {
                continue;
            }

            for &ch in &adjency_list[j]{
                if potential_peaks.contains_key(&ch){
                    exploring_neighboors.push(ch);
                }
            }
            let mut explore = !exploring_neighboors.is_empty();

            if explore {
                let mut c =0;
                for ch in &exploring_neighboors{
                    let (n_sample, n_value) = potential_peaks[&ch];
                    if (i-n_sample) > exclude_sweep_size {
                        peak_mask[[n_sample,*ch]] = true;
                        c+=1;
                    }
                    else if value > n_value {
                        //we remove the channel from the potential peaks
                        potential_peaks.remove(ch);

                        //we replace it with the new peak
                        potential_peaks.insert(j, (i, value));
                    }
                }
                explore = c < exploring_neighboors.len();
                exploring_neighboors.clear();
            }

            if !explore && value > abs_thresholds[j]{
                potential_peaks.insert(j, (i, value));
            }
        }
    }
        
    if ["neg","both"].contains(&peak_sign) {
        let mut peak_mask_pos: Array2<bool> = Array2::from_elem((n_samples_center, data.ncols()), false);
        if peak_sign == "both" {
            peak_mask_pos = peak_mask.clone();
        }

        // Create the peak mask by comparing each value to the threshold for its channel
        let mut potential_peaks: HashMap<usize,(usize, f32)> = HashMap::new();
        let mut exploring_neighboors:Vec<usize>=Vec::new();
        for ((i, j), &value) in data_center.indexed_iter() {
            if value >= -abs_thresholds[j] {
                continue;
            }

            for &ch in &adjency_list[j]{
                if potential_peaks.contains_key(&ch){
                    exploring_neighboors.push(ch);
                }
            }
            let mut explore = !exploring_neighboors.is_empty();

            if explore {
                let mut c =0;
                for ch in &exploring_neighboors{
                    let (n_sample, n_value) = potential_peaks[&ch];
                    if (i-n_sample) > exclude_sweep_size {
                        peak_mask[[n_sample,*ch]] = true;
                        c+=1;
                    }
                    else if value < n_value {
                        //we remove the channel from the potential peaks
                        potential_peaks.remove(ch);

                        //we replace it with the new peak
                        potential_peaks.insert(j, (i, value));
                    }
                }
                explore = c < exploring_neighboors.len();
                exploring_neighboors.clear();
            }

            if !explore && value < -abs_thresholds[j]{
                potential_peaks.insert(j, (i, value));
            }
        }

        if peak_sign == "both" {
            peak_mask = peak_mask | peak_mask_pos;
        }
    }

    let result: (Vec<usize>, Vec<usize>) = peak_mask.indexed_iter()
        .filter_map(|((i, j), &is_peak)| if is_peak { Some((i + exclude_sweep_size, j)) } else { None })
        .unzip();

    result
}