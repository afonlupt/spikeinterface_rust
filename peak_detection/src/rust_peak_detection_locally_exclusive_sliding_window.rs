use std::collections::VecDeque;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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
    let n_channels = data.ncols();
    if n_samples == 0 {
        return (vec![], vec![]);
    }

    use ndarray::s;
    let data_center = data.slice(s![exclude_sweep_size..n_samples-exclude_sweep_size, ..]);
    let n_samples_center = data_center.nrows();

    let mut peak_mask : Array2<bool> = Array2::from_elem((n_samples_center, n_channels), false);

    if ["pos","both"].contains(&peak_sign) {
        // Create the peak mask by comparing each value to the threshold for its channel
        let mut current_max = Array1::from_elem(n_channels, VecDeque::with_capacity(exclude_sweep_size));
        let mut solved_neighbourhood = Array1::from_elem(n_channels, false);
        let mut possible_peak = Array1::from_elem(n_channels, true);

        for i in 0..n_samples {
            solved_neighbourhood.fill(false);
            for j in 0..n_channels {
                if solved_neighbourhood[j] || data[[i,j]] <= abs_thresholds[j] {
                    continue;
                }

                let neighbours = &adjency_list[j];

                let mut max:f32 = 0.0;
                let mut i_max :usize = usize::MAX;
                for &ch in neighbours {
                    let deque: &mut VecDeque<usize> = &mut current_max[ch];
                    let value = data[[i,ch]];

                    while !solved_neighbourhood[ch] && !deque.is_empty() && i > *deque.front().unwrap() + exclude_sweep_size {
                        if possible_peak[ch] && *deque.front().unwrap() >= exclude_sweep_size{
                            peak_mask[[*deque.front().unwrap() - exclude_sweep_size, ch]] = true;
                        }
                        possible_peak[ch] = false;
                        deque.pop_front();
                    }

                    while !deque.is_empty() && value > data[[*deque.back().unwrap(),ch]] {
                        deque.pop_back();
                    }

                    if deque.is_empty(){
                        possible_peak[ch] = true;
                    }

                    deque.push_back(i);

                    solved_neighbourhood[ch] = true;

                    if max < data[[*deque.front().unwrap(),ch]] {
                        max = data[[*deque.front().unwrap(),ch]];
                        if i_max != usize::MAX {
                            possible_peak[i_max] = false;
                        }
                        i_max = ch;
                    }
                    else {
                        possible_peak[ch] = false;
                    }
                }
            }
        }
    }
        
    if ["neg","both"].contains(&peak_sign) {
        let mut peak_mask_pos: Array2<bool> = Array2::from_elem((n_samples_center, data.ncols()), false);
        if peak_sign == "both" {
            peak_mask_pos = peak_mask.clone();
        }

        // Create the peak mask by comparing each value to the threshold for its channel
        let mut current_min = Array1::from_elem(n_channels, VecDeque::with_capacity(exclude_sweep_size));
        let mut solved_neighbourhood = Array1::from_elem(n_channels, false);
        let mut possible_peak = Array1::from_elem(n_channels, true);

        for i in 0..n_samples {
            solved_neighbourhood.fill(false);
            for j in 0..n_channels {
                if solved_neighbourhood[j] || data[[i,j]] >= -abs_thresholds[j] {
                    continue;
                }

                let neighbours = &adjency_list[j];

                let mut min:f32 = 0.0;
                let mut i_min :usize = usize::MAX;
                for &ch in neighbours {
                    let deque: &mut VecDeque<usize> = &mut current_min[ch];
                    let value = data[[i,ch]];

                    while !solved_neighbourhood[ch] && !deque.is_empty() && i > *deque.front().unwrap() + exclude_sweep_size {
                        if possible_peak[ch] && *deque.front().unwrap() >= exclude_sweep_size{
                            peak_mask[[*deque.front().unwrap() - exclude_sweep_size, ch]] = true;
                        }
                        possible_peak[ch] = false;
                        deque.pop_front();
                    }

                    while !deque.is_empty() && value < data[[*deque.back().unwrap(),ch]] {
                        deque.pop_back();
                    }

                    if deque.is_empty(){
                        possible_peak[ch] = true;
                    }

                    deque.push_back(i);

                    solved_neighbourhood[ch] = true;

                    if min > data[[*deque.front().unwrap(),ch]] {
                        min = data[[*deque.front().unwrap(),ch]];
                        if i_min != usize::MAX {
                            possible_peak[i_min] = false;
                        }
                        i_min = ch;
                    }
                    else {
                        possible_peak[ch] = false;
                    }
                }
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