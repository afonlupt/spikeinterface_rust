import numpy as np
import importlib.util
import time

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False

try:
    from peak_detection import detect_peaks_rust_locally_exclusive_on_chunk
    HAVE_RUST = True
except ImportError:
    HAVE_RUST = False

from spikeinterface.core.node_pipeline import (
    PeakDetector,
)
from spikeinterface.core.recording_tools import get_channel_distances
# from .by_channel import ByChannelTorchPeakDetector

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    torch_nn_functional_spec = importlib.util.find_spec("torch.nn")
    if torch_nn_functional_spec is not None:
        HAVE_TORCH = True
    else:
        HAVE_TORCH = False
else:
    HAVE_TORCH = False

# opencl_spec = importlib.util.find_spec("pyopencl")
# if opencl_spec is not None:
#     HAVE_PYOPENCL = True
# else:
#     HAVE_PYOPENCL = False

# from .by_channel import ByChannelPeakDetector


class LocallyExclusivePeakDetector(PeakDetector):
    """Detect peaks using the "locally exclusive" method."""

    name = "locally_exclusive"
    need_noise_levels = True
    preferred_mp_context = None
    # params_doc = (
    #     ByChannelPeakDetector.params_doc
    #     + """
    # radius_um: float
    #     The radius to use to select neighbour channels for locally exclusive detection.
    # """
    # )

    def __init__(
        self,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.5,
        radius_um=50,
        noise_levels=None,
        return_output=True,
        engine="rust",
    ):
        if not HAVE_NUMBA and engine == "numba":
            raise ModuleNotFoundError('"locally_exclusive" needs numba which is not installed')
        
        if not HAVE_RUST and engine == "rust":
            raise ModuleNotFoundError('"locally_exclusive" needs the rust extension which is not installed')

        PeakDetector.__init__(self, recording, return_output=return_output)

        assert peak_sign in ("both", "neg", "pos")
        assert noise_levels is not None
        self.noise_levels = noise_levels

        self.abs_thresholds = self.noise_levels * detect_threshold
        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        # print('self.exclude_sweep_size', self.exclude_sweep_size)
        
        self.radius_um = radius_um
        self.detect_threshold = detect_threshold
        self.peak_sign = peak_sign
        # if remove_median:

        #     chunks = get_random_data_chunks(recording, return_in_uV=False, concatenated=True, **random_chunk_kwargs)
        #     medians = np.median(chunks, axis=0)
        #     medians = medians[None, :]
        #     print('medians', medians, noise_levels)
        # else:
        #     medians = None

        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um
        # print(self.neighbours_mask)
        self.engine = engine

        if engine not in ("numba", "rust"):
            raise ValueError(f'Engine "{engine}" not recognized. Should be "numba" or "rust".')

    def get_trace_margin(self):
        # the +1 in the border is important because we need peak in the border +1
        return self.exclude_sweep_size + 1

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        assert HAVE_NUMBA or HAVE_RUST, "You need to install numba or use the rust engine"

        start_time = time.time()
        if self.engine == "numba":
            peak_sample_ind, peak_chan_ind = detect_peaks_numba_locally_exclusive_on_chunk(
                traces, self.peak_sign, self.abs_thresholds, self.exclude_sweep_size, self.neighbours_mask
            )
        elif self.engine == "rust":
            peak_sample_ind, peak_chan_ind = detect_peaks_rust_locally_exclusive_on_chunk(
                traces, self.peak_sign, self.abs_thresholds, self.exclude_sweep_size, self.neighbours_mask
            )
        #print(f"Compute peaks on chunk time ({self.engine}): {time.time() - start_time:.3f} s")

        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]

        local_peaks = np.zeros(peak_sample_ind.size, dtype=self.get_dtype())
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index

        return (local_peaks,)

##############################
## equivalent à rust sam 2

# if HAVE_NUMBA:
#     import numba

#     def detect_peaks_numba_locally_exclusive_on_chunk(
#         traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask
#     ):

#         # if medians is not None:
#         #     traces = traces - medians

#         traces_center = traces[exclude_sweep_size:-exclude_sweep_size, :]

#         if peak_sign in ("pos", "both"):
#             peak_mask = traces_center > abs_thresholds[None, :]
#             peak_mask = _numba_detect_peak_pos(
#                 traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
#             )

#         if peak_sign in ("neg", "both"):
#             if peak_sign == "both":
#                 peak_mask_pos = peak_mask.copy()

#             peak_mask = traces_center < -abs_thresholds[None, :]
#             # print('npeak before clean', np.sum(peak_mask))
#             peak_mask = _numba_detect_peak_neg(
#                 traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
#             )
#             # print('npeak after clean', np.sum(peak_mask))

#             if peak_sign == "both":
#                 peak_mask = peak_mask | peak_mask_pos

#         # Find peaks and correct for time shift
#         peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
#         peak_sample_ind += exclude_sweep_size

#         return peak_sample_ind, peak_chan_ind

#     @numba.jit(nopython=True, parallel=False, nogil=True)
#     def _numba_detect_peak_pos(
#         traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
#     ):
#         num_chans = traces_center.shape[1]
#         for chan_ind in range(num_chans):
#             for s in range(peak_mask.shape[0]):
#                 if not peak_mask[s, chan_ind]:
#                     continue
#                 for neighbour in range(num_chans):
#                     if not neighbours_mask[chan_ind, neighbour]:
#                         continue
#                     if chan_ind != neighbour and peak_mask[s, neighbour]:
#                         peak_mask[s, chan_ind] &= traces_center[s, chan_ind] >= traces_center[s, neighbour]

#                     for i in range(exclude_sweep_size):
#                         # if not peak_mask[s+ i, neighbour] and not peak_mask[exclude_sweep_size + s + i +1, neighbour]:
#                         #     continue

#                         peak_mask[s, chan_ind] &= traces_center[s, chan_ind] > traces[s + i, neighbour]
#                         peak_mask[s, chan_ind] &= (
#                             traces_center[s, chan_ind] >= traces[exclude_sweep_size + s + i + 1, neighbour]
#                         )
#                         if not peak_mask[s, chan_ind]:
#                             break
#                     if not peak_mask[s, chan_ind]:
#                         break
#         return peak_mask



#     @numba.jit(nopython=True, parallel=False, nogil=True)
#     def _numba_detect_peak_neg(
#         traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
#     ):
#         num_chans = traces_center.shape[1]
#         # for chan_ind in range(num_chans):
#         #     for s in range(peak_mask.shape[0]):
#         for s in range(peak_mask.shape[0]):
#             for chan_ind in range(num_chans):
            
#                 if not peak_mask[s, chan_ind]:
#                     continue
#                 value = traces_center[s, chan_ind] / abs_thresholds[chan_ind]
#                 for neighbour in range(num_chans):
#                     if not neighbours_mask[chan_ind, neighbour]:
#                         continue

#                     # if chan_ind != neighbour and peak_mask[s, neighbour]:  <<<<<<<< difference entre les 2 algos
#                     if chan_ind != neighbour:
#                         neighbour_value = traces_center[s, neighbour] / abs_thresholds[neighbour]
#                         peak_mask[s, chan_ind] &= value <= neighbour_value

#                     for i in range(exclude_sweep_size):
#                         # if not peak_mask[s+ i, neighbour] and not peak_mask[exclude_sweep_size + s + i +1, neighbour]:
#                         #     continue
#                         neighbour_value = traces[s + i, neighbour] / abs_thresholds[neighbour]
#                         peak_mask[s, chan_ind] &= value < neighbour_value
#                         neighbour_value = traces[exclude_sweep_size + s + i + 1, neighbour] / abs_thresholds[neighbour]
#                         peak_mask[s, chan_ind] &= value <= neighbour_value
#                         if not peak_mask[s, chan_ind]:
#                             break
#                     if not peak_mask[s, chan_ind]:
#                         break
#         return peak_mask



##############################
## equivalent à rust sam 3

if HAVE_NUMBA:
    import numba

    # def detect_peaks_numba_locally_exclusive_on_chunk(
    #     traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask
    # ):

    #     peak_sample_ind, peak_chan_ind = _numba_detect_peak(
    #         traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask, 
    #     )

    #     return peak_sample_ind, peak_chan_ind





    @numba.jit(nopython=True, parallel=False, nogil=True)
    def detect_peaks_numba_locally_exclusive_on_chunk(
        traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask, 
    ):
        num_chans = traces.shape[1]
        num_samples = traces.shape[0]

        
        do_pos = peak_sign in ("pos", "both")
        do_neg = peak_sign in ("neg", "both")

        # first find peaks
        peak_mask = np.zeros(traces.shape, dtype="bool")
        for s in range(1, num_samples - 1):
            for chan_ind in range(num_chans):
                if do_neg:
                    if (traces[s, chan_ind] <= -abs_thresholds[chan_ind]) and \
                            (traces[s, chan_ind] < traces[s-1, chan_ind]) and \
                            (traces[s, chan_ind] <= traces[s+1, chan_ind]):
                        peak_mask[s, chan_ind] = True

                if do_pos :
                    if (traces[s, chan_ind] >= abs_thresholds[chan_ind]) and \
                            (traces[s, chan_ind] > traces[s-1, chan_ind]) and \
                            (traces[s, chan_ind] >= traces[s+1, chan_ind]):
                        peak_mask[s, chan_ind] = True

        samples_inds, chan_inds = np.nonzero(peak_mask)

        npeaks = samples_inds.size
        keep_peak = np.ones(npeaks, dtype="bool")
        next_start = 0
        for i in range(npeaks):

            if (samples_inds[i] < exclude_sweep_size + 1) or (samples_inds[i]>= (num_samples - exclude_sweep_size - 1)):
                keep_peak[i] = False
                continue
            
            for j in range(next_start, npeaks):
                if i == j:
                    continue

                if samples_inds[i] + exclude_sweep_size < samples_inds[j]:
                    break

                if samples_inds[i] - exclude_sweep_size > samples_inds[j]:
                    next_start = j
                    continue
                
                #search for neighbors
                if neighbours_mask[chan_inds[i], chan_inds[j]]:
                    # if inside spatial zone
                    if abs(samples_inds[i] - samples_inds[j]) <= exclude_sweep_size:
                        value_i = abs(traces[samples_inds[i], chan_inds[i]]) / abs_thresholds[chan_inds[i]]
                        value_j = abs(traces[samples_inds[j], chan_inds[j]]) / abs_thresholds[chan_inds[j]]
                    
                        if ((value_j >= value_i) & (samples_inds[i] > samples_inds[j])) | ((value_j > value_i) & (samples_inds[i] <= samples_inds[j])):
                            keep_peak[i] = False
                            break

        samples_inds, chan_inds = samples_inds[keep_peak], chan_inds[keep_peak]

        return samples_inds, chan_inds

