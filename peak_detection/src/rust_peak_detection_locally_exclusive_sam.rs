use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn detect_peaks_rust_locally_exclusive_on_chunk<'py>(py: Python<'py>, traces: PyReadonlyArray2<f32>, peak_sign: &str, abs_thresholds: PyReadonlyArray1<f32>, exclude_sweep_size: usize, neighbours_mask: PyReadonlyArray2<bool>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<usize>>) {
    assert!(["pos", "neg", "both"].contains(&peak_sign), "peak_sign must be 'pos', 'neg', or 'both'");

    let traces: ArrayView2<f32> = traces.as_array();
    let abs_thresholds: ArrayView1<f32> = abs_thresholds.as_array();
    let neighbours_mask: ArrayView2<bool> = neighbours_mask.as_array();
    // let adjency_list: Vec<Vec<usize>> = neighbours_mask.axis_iter(ndarray::Axis(0))
    //     .map(|row| row.indexed_iter()
    //         .filter_map(|(j, &is_neighbor)| if is_neighbor { Some(j) } else { None })
    //         .collect()
    //     )
    //     .collect();

    // let peaks = detect_peaks_locally_exclusive(&traces, peak_sign, &abs_thresholds, exclude_sweep_size, &adjency_list);
    // let peaks: (Vec<usize>, Vec<usize>) = py.detach(|| {detect_peaks_locally_exclusive(&traces, peak_sign, &abs_thresholds, exclude_sweep_size, &adjency_list)});

    let n_samples = traces.nrows();
    if n_samples == 0 {
        // return (vec![], vec![]);
        return (vec![].into_pyarray(py), vec![].into_pyarray(py));
    }
    else {

    use ndarray::s;
    let traces_center = traces.slice(s![exclude_sweep_size..n_samples-exclude_sweep_size, ..]);
    let n_samples_center = traces_center.nrows();

    let mut peak_mask : Array2<bool> = Array2::from_elem((n_samples_center, traces.ncols()), false);

    let peaks: (Vec<usize>, Vec<usize>) = py.detach(|| 
        {

            if ["pos","both"].contains(&peak_sign) {
                // Create the peak mask by comparing each value to the threshold for its channel
                for ((i, j), &value) in traces_center.indexed_iter() {
                    if value > abs_thresholds[j] {
                        peak_mask[[i, j]] = true;
                    }
                    // else {
                    //     peak_mask[[i, j]] = false;
                    // }
                }
                // remove_neighboring_peaks(&mut peak_mask, &traces,&traces_center, &adjency_list, exclude_sweep_size,"pos");
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
                    // else {
                    //     peak_mask[[i, j]] = false;
                    // }
                }
                // remove_neighboring_peaks(&mut peak_mask, &traces,&traces_center, &adjency_list, exclude_sweep_size,"neg");


                remove_neighboring_peaks_neg(&mut peak_mask, &traces, &traces_center, exclude_sweep_size, neighbours_mask);

                if peak_sign == "both" {
                    peak_mask = peak_mask | peak_mask_pos;
                }
            }
        let peaks: (Vec<usize>, Vec<usize>) = peak_mask.indexed_iter()
            .filter_map(|((i, j), &is_peak)| if is_peak { Some((i + exclude_sweep_size, j)) } else { None })
            .unzip();
        peaks
        }
    );



    return (peaks.0.into_pyarray(py), peaks.1.into_pyarray(py));
    }

}



fn remove_neighboring_peaks_neg(peak_mask: &mut Array2<bool>, traces: &ArrayView2<f32>, traces_center: &ArrayView2<f32>, exclude_sweep_size: usize, neighbours_mask: ArrayView2<bool>) {
    let num_channels = traces.ncols();
    let num_samples = traces_center.nrows();

    for chan_ind in 0..num_channels{
        for s in 0..num_samples{
            if !peak_mask[[s, chan_ind]] {
                continue;
            }
            for neighbour in 0..num_channels{
                if !neighbours_mask[[chan_ind, neighbour]]{
                    continue;
                }
                for i in 0..exclude_sweep_size{
                    if chan_ind != neighbour{
                        // peak_mask[s, chan_ind] &= traces_center[s, chan_ind] <= traces_center[s, neighbour];
                        peak_mask[[s, chan_ind]] = peak_mask[[s, chan_ind]] & (traces_center[[s, chan_ind]] <= traces_center[[s, neighbour]]);
                    }
                    // peak_mask[s, chan_ind] &= traces_center[s, chan_ind] < traces[s + i, neighbour];
                    peak_mask[[s, chan_ind]] = peak_mask[[s, chan_ind]] & (traces_center[[s, chan_ind]] < traces[[s + i, neighbour]]);
                    // peak_mask[s, chan_ind] &= traces_center[s, chan_ind] <= traces[exclude_sweep_size + s + i + 1, neighbour];
                    peak_mask[[s, chan_ind]] = peak_mask[[s, chan_ind]] & (traces_center[[s, chan_ind]] <= traces[[exclude_sweep_size + s + i + 1, neighbour]]);

                    if !peak_mask[[s, chan_ind]]{
                        break;
                    }
                }
                if !peak_mask[[s, chan_ind]]{
                    break;
                }
            }

        }
    }
}

// fn remove_neighboring_peaks_pos(peak_mask: &mut Array2<bool>, traces: &ArrayView2<f32>, traces_center: &ArrayView2<f32>, adjency_list: &Vec<Vec<usize>>, exclude_sweep_size: usize, peak_sign: &str) {



// }



// fn remove_neighboring_peaks(result_peak_mask: &mut Array2<bool>, traces: &ArrayView2<f32>, traces_center: &ArrayView2<f32>, adjency_list: &Vec<Vec<usize>>, exclude_sweep_size: usize, peak_sign: &str) {
//     assert!(["pos", "neg"].contains(&peak_sign), "peak_sign must be 'pos' or 'neg'");

//     if peak_sign == "pos" {
//         let num_channels = traces.ncols();
//         let num_samples = traces_center.nrows();
//         for chan_ind in 0..num_channels{
//             for s in 0..num_samples{
//                 if !result_peak_mask[[s, chan_ind]] {
//                     continue;
//                 }
//                 for &neighbour in adjency_list[chan_ind].iter(){
//                     if chan_ind != neighbour{
//                         if traces_center[[s, chan_ind]] >= traces_center[[s, neighbour]]{
//                             result_peak_mask[[s, neighbour]] = false;
//                         }
//                         else{
//                             result_peak_mask[[s, chan_ind]] = false;
//                             break;
//                         }
//                     }

//                     for i in 0..exclude_sweep_size{
//                         if traces_center[[s, chan_ind]] > traces[[s + i, neighbour]]{
//                             if (s + i) as isize - exclude_sweep_size as isize >=0 {
//                                 result_peak_mask[[s + i -exclude_sweep_size, neighbour]] = false;
//                             }
//                         }
//                         else{
//                             result_peak_mask[[s, chan_ind]] = false;
//                             break;
//                         }

//                         if traces_center[[s, chan_ind]] >= traces[[exclude_sweep_size + s + i + 1, neighbour]]{
//                             if s + i + 1 < num_samples {
//                                 result_peak_mask[[s + i + 1, neighbour]] = false;
//                             }
//                         }
//                         else{
//                             result_peak_mask[[s, chan_ind]] = false;
//                             break;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     else{
//         let num_channels = traces.ncols();
//         let num_samples = traces_center.nrows();
//         for chan_ind in 0..num_channels{
//             for s in 0..num_samples{
//                 if !result_peak_mask[[s, chan_ind]] {
//                     continue;
//                 }
//                 for &neighbour in adjency_list[chan_ind].iter(){
//                     if chan_ind != neighbour{
//                         if traces_center[[s, chan_ind]] <= traces_center[[s, neighbour]]{
//                             result_peak_mask[[s, neighbour]] = false;
//                         }
//                         else{
//                             result_peak_mask[[s, chan_ind]] = false;
//                             break;
//                         }
//                     }

//                     for i in 0..exclude_sweep_size{
//                         if traces_center[[s, chan_ind]] < traces[[s + i, neighbour]]{
//                             if (s + i) as isize - exclude_sweep_size as isize >=0 {
//                                 result_peak_mask[[s + i -exclude_sweep_size, neighbour]] = false;
//                             }
//                         }
//                         else{
//                             result_peak_mask[[s, chan_ind]] = false;
//                             break;
//                         }

//                         if traces_center[[s, chan_ind]] <= traces[[exclude_sweep_size + s + i + 1, neighbour]]{
//                             if s + i + 1 < num_samples {
//                                 result_peak_mask[[s + i + 1, neighbour]] = false;
//                             }
//                         }
//                         else{
//                             result_peak_mask[[s, chan_ind]] = false;
//                             break;
//                         }
//                     }
//                 }
//             }
//         }
//     }

// }




