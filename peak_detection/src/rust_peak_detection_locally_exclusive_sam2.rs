use ndarray::{Array1,  ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn detect_peaks_rust_locally_exclusive_on_chunk<'py>(py: Python<'py>, traces: PyReadonlyArray2<f32>, peak_sign: &str,
            abs_thresholds: PyReadonlyArray1<f32>, exclude_sweep_size: usize,
            neighbours_mask: PyReadonlyArray2<bool>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<usize>>) {
    assert!(["pos", "neg", "both"].contains(&peak_sign), "peak_sign must be 'pos', 'neg', or 'both'");

    let traces: ArrayView2<f32> = traces.as_array();
    let abs_thresholds: ArrayView1<f32> = abs_thresholds.as_array();
    let neighbours_mask: ArrayView2<bool> = neighbours_mask.as_array();

    let peaks: (Vec<usize>, Vec<usize>) = py.detach(
        || {detect_peaks_locally_exclusive(&traces, peak_sign, &abs_thresholds, exclude_sweep_size, &neighbours_mask)}
    );
    return (peaks.0.into_pyarray(py), peaks.1.into_pyarray(py));
}



fn detect_peaks_locally_exclusive(traces : &ArrayView2<f32>, peak_sign: &str, abs_thresholds: &ArrayView1<f32>,
    exclude_sweep_size: usize, neighbours_mask: &ArrayView2<bool>) -> (Vec<usize>, Vec<usize>) {

    // use ndarray::s;

    let n_samples = traces.nrows();
    
    if n_samples == 0 {
        return (vec![], vec![]);
    }

    // let traces_center = traces.slice(s![exclude_sweep_size..n_samples-exclude_sweep_size, ..]);

    if ["pos","both"].contains(&peak_sign) {

    }

    if ["neg","both"].contains(&peak_sign) {

        let peaks: (Vec<usize>, Vec<usize>) = traces.indexed_iter()
            .filter_map(
                |((sample_ind, chan_ind), &value)|
                    if value < -abs_thresholds[chan_ind] { Some((sample_ind, chan_ind)) }
                    else { None }
            ).unzip();
        
        let npeaks = peaks.0.len();
        let mut keep_peak: Array1<bool> = Array1::from_elem(npeaks, true);

        let mut next_start: usize =0;
        for i in 0..npeaks{
            if (peaks.0[i] < exclude_sweep_size) || (peaks.0[i] >= (n_samples - exclude_sweep_size)){
                // peak on the border
                keep_peak[[i]] = false;
                continue;
            }

            for j in next_start..npeaks{
                if i == j {continue;}
                
                if (peaks.0[i]  + exclude_sweep_size ) < peaks.0[j] {
                    //  println!("break {}", j);
                    break;
                }
                if (peaks.0[i]  - exclude_sweep_size ) > peaks.0[j]{
                    next_start = j;
                    continue;
                }

                // search for neighbors
                if neighbours_mask[[peaks.1[i], peaks.1[j]]]{
                    // if inside spatial zone
                    if peaks.0[i].abs_diff(peaks.0[j]) <= exclude_sweep_size {
                        // if inside time zone
                        let value_i = traces[[peaks.0[i], peaks.1[i]]] / abs_thresholds[[peaks.1[i]]];
                        let value_j = traces[[peaks.0[j], peaks.1[j]]] / abs_thresholds[[peaks.1[j]]];
                        if ((value_j <= value_i) & (peaks.0[i] > peaks.0[j])) ||
                               ((value_j < value_i) & (peaks.0[i] <= peaks.0[j])) {
                            keep_peak[[i]] = false;
                            break;
                        }
                    }
                }
            }
        }

        let peaks_clean: (Vec<usize>, Vec<usize>) = peaks.0.iter().zip(peaks.1.iter()).enumerate().filter_map(
            |(i, (sample_ind, chan_ind))|
                if keep_peak[i] {Some((sample_ind, chan_ind))}
                else {None}
        ).unzip();

        return peaks_clean;
    }


    return (vec![], vec![]);
}
