use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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

    use ndarray::s;

    let n_samples = traces.nrows();
    
    if n_samples == 0 {
        return (vec![], vec![]);
    }

    let traces_center = traces.slice(s![exclude_sweep_size..n_samples-exclude_sweep_size, ..]);
    // let n_samples_center = traces_center.nrows();

    // let mut peak_mask : Array2<bool> = Array2::from_elem((n_samples_center, traces.ncols()), false);


    // let mut peaks_clean: (Vec<usize>, Vec<usize>);

    if ["pos","both"].contains(&peak_sign) {

        // for ((i, j), &value) in traces_center.indexed_iter() {
        //     if value > abs_thresholds[j] {
        //         peak_mask[[i, j]] = true;
        //     }
        // }
        // remove_neighboring_peaks_pos(&mut peak_mask, &traces, &traces_center, &abs_thresholds, exclude_sweep_size, &neighbours_mask);
    }

    if ["neg","both"].contains(&peak_sign) {

        // for ((sample_ind, chan_ind), &value) in traces_center.indexed_iter() {
        //     if value < -abs_thresholds[chan_ind] {
        //         peak_mask[[sample_ind, chan_ind]] = true;
        //     }
        // }

        // let peaks: (Vec<usize>, Vec<usize>) = peak_mask.indexed_iter()
        //     .filter_map(|((sample_ind, chan_ind), &is_peak)| if is_peak { Some((sample_ind, chan_ind)) } else { None })
        //     .unzip();
        let peaks: (Vec<usize>, Vec<usize>) = traces_center.indexed_iter()
            .filter_map(|((sample_ind, chan_ind), &value)| if value > abs_thresholds[chan_ind] { Some((sample_ind, chan_ind)) } else { None })
            .unzip();


        let npeaks = peaks.0.len();
        let mut keep_peak: Array1<bool> = Array1::from_elem(npeaks, true);
        // println!("npeaks{} shape{:?} {:?}", npeaks, traces_center.shape(), keep_peak.shape());
        
        
        let mut next_start: usize =0;
        let mut nloop: usize =0;
        let mut nremoved: usize =0;

        for i in 0..npeaks{
            println!("{} {}", i, next_start);
            for j in next_start..npeaks{
            // for j in 0..npeaks{
                if i == j{
                    continue;
                }
                if (peaks.0[i]  + exclude_sweep_size + 1) < peaks.0[j] {
                     println!("break {}", j);
                    break;
                }
                if (peaks.0[i]  - exclude_sweep_size - 1 ) > peaks.0[j]{
                    next_start = j;
                }

                // search for neighbors in time
                if neighbours_mask[[peaks.1[i], peaks.1[j]]]{
                    // if inside spatial zone
                    if (usize::abs(peaks.0[i] - peaks.0[j]) <= exclude_sweep_size){
                        // if inside time zone
                        let value_i = traces_center[[peaks.0[i], peaks.1[i]]] / abs_thresholds[[peaks.1[i]]];
                        let value_j = traces_center[[peaks.0[j], peaks.1[j]]] / abs_thresholds[[peaks.1[j]]];
                        if value_j < value_i {
                            keep_peak[[i]] = false;
                            nremoved += 1;
                            println!("break not peak {} {}", i ,j);
                            break;
                        }
                    }
                }
                nloop += 1;

            }
        }
        println!("nloop {} npeaks {} nremoved{}", nloop, npeaks, nremoved);

        let peaks_clean: (Vec<usize>, Vec<usize>) = peaks.0.iter().zip(peaks.1.iter()).enumerate().filter_map(
            |(i, (sample_ind, chan_ind))| if keep_peak[i] {Some((sample_ind + exclude_sweep_size, chan_ind))} else {None}
        ).unzip();

        return peaks_clean;
    }


    return (vec![], vec![]);
}
