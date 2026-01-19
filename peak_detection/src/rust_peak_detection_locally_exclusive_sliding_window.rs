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

        let peaks: (Vec<usize>, Vec<usize>) = py.detach(|| {detect_peaks_locally_exclusive(&data, peak_sign, &abs_thresholds, exclude_sweep_size, &adjency_list)});

        (peaks.0.into_pyarray(py), peaks.1.into_pyarray(py))
}

fn detect_peaks_locally_exclusive(data : &ArrayView2<f32>, peak_sign: &str, abs_thresholds: &ArrayView1<f32>, exclude_sweep_size: usize, adjency_list: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>) {

    let n_samples = data.nrows();
    let n_channels = data.ncols();
    if n_samples == 0 {
        return (vec![], vec![]);
    }

    use ndarray::s;
    let data_center = data.slice(s![exclude_sweep_size..n_samples-exclude_sweep_size, ..]);
    let n_samples_center = data_center.nrows();

    let mut sample_vec: Vec<usize> = Vec::with_capacity(n_samples_center / 20);
    let mut channel_vec: Vec<usize> = Vec::with_capacity(n_samples_center / 20);

    if ["pos","both"].contains(&peak_sign) {
        // Create the peak mask by comparing each value to the threshold for its channel
        let mut current_max = Array1::from_elem(n_channels, VecDeque::with_capacity(exclude_sweep_size + 1));
        let mut added_neighbour = Array1::from_elem(n_channels, false);
        let mut compared_neighbour = Array1::from_elem(n_channels, false);
        let mut possible_peak = Array1::from_elem(n_channels, false);

        for i in 0..n_samples {
            added_neighbour.fill(false);
            compared_neighbour.fill(false);
            for j in 0..n_channels {
                let value = data[[i,j]]/abs_thresholds[j];
                if (added_neighbour[j] && compared_neighbour[j]) || value <= 1.0 {
                    continue;
                }

                let deque: &mut VecDeque<usize> = &mut current_max[j];

                if !added_neighbour[j] {
                    while !deque.is_empty() && i > *deque.front().unwrap() + exclude_sweep_size {
                        if possible_peak[j] && *deque.front().unwrap() >= exclude_sweep_size{
                            sample_vec.push(*deque.front().unwrap() - exclude_sweep_size);
                            channel_vec.push(j);
                        }
                        possible_peak[j] = false;
                        deque.pop_front();
                    }

                    while !deque.is_empty() && value > data[[*deque.back().unwrap(),j]]/abs_thresholds[j] {
                        deque.pop_back();
                    }

                    if deque.is_empty(){
                        possible_peak[j] = true;
                    }

                    deque.push_back(i);

                    added_neighbour[j] = true;
                }

                let neighbours = &adjency_list[j];
                let max_current_ch = data[[*deque.front().unwrap(), j]]/abs_thresholds[j];
                compared_neighbour[j] = true;

                for &ch in neighbours {
                    if compared_neighbour[ch] {
                        continue;
                    }

                    let deque: &mut VecDeque<usize> = &mut current_max[ch];

                    if !added_neighbour[ch] {
                        while !deque.is_empty() && i > *deque.front().unwrap() + exclude_sweep_size {
                            if possible_peak[ch] && *deque.front().unwrap() >= exclude_sweep_size{
                                sample_vec.push(*deque.front().unwrap() - exclude_sweep_size);
                                channel_vec.push(ch);
                            }
                            possible_peak[ch] = false;
                            deque.pop_front();
                        }

                        let value = data[[i,ch]]/abs_thresholds[ch];

                        if value > 1.0 {

                            while !deque.is_empty() && value > data[[*deque.back().unwrap(),ch]]/abs_thresholds[ch] {
                                deque.pop_back();
                            }

                            if deque.is_empty(){
                                possible_peak[ch] = true;
                            }

                            deque.push_back(i);
                        }

                        added_neighbour[ch] = true;
                    }

                    if deque.is_empty() {
                        compared_neighbour[ch] = true;
                        continue;
                    }

                    if max_current_ch < data[[*deque.front().unwrap(),ch]]/abs_thresholds[ch] {
                        possible_peak[j] = false;
                    }
                    else {
                        possible_peak[ch] = false;
                    }
                }
            }
        }

        for i in 0..n_channels {
            let deque: &mut VecDeque<usize> = &mut current_max[i];
            while !deque.is_empty() {
                let last =deque.pop_front().unwrap();
                if possible_peak[i] && last >= exclude_sweep_size && last < n_samples - exclude_sweep_size {
                        sample_vec.push(last - exclude_sweep_size);
                        channel_vec.push(i);
                }
                else {
                    break;
                }
            }
        }
    }
        
    if ["neg","both"].contains(&peak_sign) {
        let mut sample_vec_pos: Vec<usize> = Vec::new();
        let mut channel_vec_pos: Vec<usize> = Vec::new();
        if peak_sign == "both" {
            sample_vec_pos = sample_vec.clone();
            channel_vec_pos = channel_vec.clone();
            sample_vec.clear();
            channel_vec.clear();
        }

        // Create the peak mask by comparing each value to the threshold for its channel
        let mut current_min = Array1::from_elem(n_channels, VecDeque::with_capacity(exclude_sweep_size + 1));
        let mut added_neighbour = Array1::from_elem(n_channels, false);
        let mut compared_neighbour = Array1::from_elem(n_channels, false);
        let mut possible_peak = Array1::from_elem(n_channels, false);

        for i in 0..n_samples {
            added_neighbour.fill(false);
            compared_neighbour.fill(false);
            for j in 0..n_channels {
                let value = data[[i,j]]/abs_thresholds[j];
                if (added_neighbour[j] && compared_neighbour[j]) || value >= -1.0 {
                    continue;
                }

                let deque: &mut VecDeque<usize> = &mut current_min[j];

                if !added_neighbour[j] {
                    while !deque.is_empty() && i > *deque.front().unwrap() + exclude_sweep_size {
                        if possible_peak[j] && *deque.front().unwrap() >= exclude_sweep_size{
                            sample_vec.push(*deque.front().unwrap() - exclude_sweep_size);
                            channel_vec.push(j);
                        }
                        possible_peak[j] = false;
                        deque.pop_front();
                    }

                    while !deque.is_empty() && value < data[[*deque.back().unwrap(),j]]/abs_thresholds[j] {
                        deque.pop_back();
                    }

                    if deque.is_empty(){
                        possible_peak[j] = true;
                    }

                    deque.push_back(i);

                    added_neighbour[j] = true;
                }

                let neighbours = &adjency_list[j];
                let min_current_ch = data[[*deque.front().unwrap(), j]]/abs_thresholds[j];
                compared_neighbour[j] = true;

                for &ch in neighbours {
                    if compared_neighbour[ch] {
                        continue;
                    }

                    let deque: &mut VecDeque<usize> = &mut current_min[ch];

                    if !added_neighbour[ch] {
                        while !deque.is_empty() && i > *deque.front().unwrap() + exclude_sweep_size {
                            if possible_peak[ch] && *deque.front().unwrap() >= exclude_sweep_size{
                                sample_vec.push(*deque.front().unwrap() - exclude_sweep_size);
                                channel_vec.push(ch);
                            }
                            possible_peak[ch] = false;
                            deque.pop_front();
                        }

                        let value = data[[i,ch]]/abs_thresholds[ch];

                        if value < -1.0 {

                            while !deque.is_empty() && value < data[[*deque.back().unwrap(),ch]]/abs_thresholds[ch] {
                                deque.pop_back();
                            }

                            if deque.is_empty(){
                                possible_peak[ch] = true;
                            }

                            deque.push_back(i);
                        }

                        added_neighbour[ch] = true;
                    }

                    if deque.is_empty() {
                        compared_neighbour[ch] = true;
                        continue;
                    }

                    if min_current_ch > data[[*deque.front().unwrap(),ch]]/abs_thresholds[ch] {
                        possible_peak[j] = false;
                    }
                    else {
                        possible_peak[ch] = false;
                    }
                }
            }
        }

        for i in 0..n_channels {
            let deque: &mut VecDeque<usize> = &mut current_min[i];
            while !deque.is_empty() {
                let last =deque.pop_front().unwrap();
                if possible_peak[i] && last >= exclude_sweep_size && last < n_samples - exclude_sweep_size {
                        sample_vec.push(last - exclude_sweep_size);
                        channel_vec.push(i);
                }
                else {
                    break;
                }
            }
        }

        if peak_sign == "both" {
            (sample_vec, channel_vec) = fusion_list(&sample_vec, &sample_vec_pos, &channel_vec, &channel_vec_pos);
        }
    }

    (sample_vec, channel_vec)
}

fn fusion_list (vec1: &[usize], vec2: &[usize], link1: &[usize], link2: &[usize]) -> (Vec<usize>, Vec<usize>) {
    if vec1.is_empty() {
        return (vec2.to_vec(), link2.to_vec());
    }
    else if vec2.is_empty() {
        return (vec1.to_vec(), link1.to_vec());
    }
    let total_len = vec1.len() + vec2.len();

    let mut c1 = 0;
    let mut c2 = 0;

    let mut fused_vec: Vec<usize> = Vec::with_capacity(total_len);
    let mut fused_link: Vec<usize> = Vec::with_capacity(total_len);

    for _ in 0..total_len {
        if c1 < vec1.len() && (c2 >= vec2.len() || vec1[c1] <= vec2[c2]) {
            fused_vec.push(vec1[c1]);
            fused_link.push(link1[c1]);
            c1 += 1;
        }
        else {
            fused_vec.push(vec2[c2]);
            fused_link.push(link2[c2]);
            c2 += 1;
        }
    }
    (fused_vec, fused_link)
}