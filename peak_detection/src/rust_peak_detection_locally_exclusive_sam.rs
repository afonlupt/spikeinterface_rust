use ndarray::{Array2, ArrayView1, ArrayView2};
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

    let peaks: (Vec<usize>, Vec<usize>) = py.detach(|| {detect_peaks_locally_exclusive(&traces, peak_sign, &abs_thresholds, exclude_sweep_size, &neighbours_mask)});
    return (peaks.0.into_pyarray(py), peaks.1.into_pyarray(py));

}


fn detect_peaks_locally_exclusive(traces : &ArrayView2<f32>, peak_sign: &str, abs_thresholds: &ArrayView1<f32>,
    exclude_sweep_size: usize, neighbours_mask: &ArrayView2<bool>) -> (Vec<usize>, Vec<usize>) {

    use ndarray::s;

    let n_samples = traces.nrows();
    
    if n_samples == 0 {
        return (vec![], vec![]);
    }

    let traces_center = traces.slice(s![exclude_sweep_size..n_samples-exclude_sweep_size, ..]);
    let n_samples_center = traces_center.nrows();

    let mut peak_mask : Array2<bool> = Array2::from_elem((n_samples_center, traces.ncols()), false);


    if ["pos","both"].contains(&peak_sign) {

        for ((i, j), &value) in traces_center.indexed_iter() {
            if value > abs_thresholds[j] {
                peak_mask[[i, j]] = true;
            }
        }
        // remove_neighboring_peaks_pos(&mut peak_mask, &traces, &traces_center, &abs_thresholds, exclude_sweep_size, &neighbours_mask);
    }

    if ["neg","both"].contains(&peak_sign) {
        let mut peak_mask_pos: Array2<bool> = Array2::from_elem((n_samples_center, traces.ncols()), false);
        if peak_sign == "both" {
            peak_mask_pos = peak_mask.clone();
        }

        for ((i, j), &value) in traces_center.indexed_iter() {
            if value < -abs_thresholds[j] {
                peak_mask[[i, j]] = true;
            }
        }

        remove_neighboring_peaks_neg(&mut peak_mask, &traces, &traces_center, &abs_thresholds, exclude_sweep_size, &neighbours_mask);

        if peak_sign == "both" {
            peak_mask = peak_mask | peak_mask_pos;
        }
    }

    let peaks: (Vec<usize>, Vec<usize>) = peak_mask.indexed_iter()
        .filter_map(|((i, j), &is_peak)| if is_peak { Some((i + exclude_sweep_size, j)) } else { None })
        .unzip();
    peaks

}


fn remove_neighboring_peaks_neg(peak_mask: &mut Array2<bool>, traces: &ArrayView2<f32>, traces_center: &ArrayView2<f32>,
    abs_thresholds: &ArrayView1<f32>, exclude_sweep_size: usize, neighbours_mask: &ArrayView2<bool>) {
    let num_channels = traces.ncols();
    // let num_samples = traces_center.nrows();

    // for chan_ind in 0..num_channels{
    //     for s in 0..num_samples{
    for ((s, chan_ind), &abs_value) in traces_center.indexed_iter() {
            let mut pm: bool = peak_mask[[s, chan_ind]];
            // let tc = traces_center[[s, chan_ind]];

            let value = abs_value / abs_thresholds[chan_ind];

            if !pm {
                continue;
            }
            for neighbour in 0..num_channels{
                if !neighbours_mask[[chan_ind, neighbour]]{
                    continue;
                }
                let neighbour_thresh = abs_thresholds[[neighbour]];

                if (chan_ind != neighbour) && peak_mask[[s, neighbour]]{
                    pm &= value <= (traces_center[[s, neighbour]] / neighbour_thresh);

                }
                for i in 0..exclude_sweep_size{
                    pm &= value < (traces[[s + i, neighbour]] / neighbour_thresh);
                    pm &= value <= (traces[[exclude_sweep_size + s + i + 1, neighbour]] / neighbour_thresh);
                    peak_mask[[s, chan_ind]] = pm;
                    if !pm {break;}
                }
                if !pm {break;}
            }
        // }
    }
}
