# envelope.py  (Stage 3)
#
# After Stage 2, each channel contains a band-limited waveform -- a signal
# that oscillates rapidly at that channel's centre frequency. For example,
# channel 10 (centred around 1500 Hz) oscillates 1500 times per second.
#
# The cochlear implant does NOT send these raw oscillations to the electrodes.
# Instead it only cares about HOW LOUD each channel is at each moment in time.
# That slowly-changing loudness profile is called the TEMPORAL ENVELOPE.
#
# Think of it like watching a song's VU meter (the bar that bounces up and down
# to the music). The meter doesn't show individual sound waves -- it shows the
# overall loudness. That's the envelope.
#
# This module does three things:
#
#   1. ENVELOPE EXTRACTION
#      Rectify (flip negatives to zero) then smooth (low-pass filter).
#      Result: for each channel, a slowly-varying amplitude between 0 and 1.
#
#   2. ACE n-of-m CHANNEL SELECTION
#      Cochlear's "Advanced Combination Encoder" strategy:
#      Out of m=22 channels, only the n loudest (e.g. n=8) fire each cycle.
#      This reduces power consumption and inter-channel interference.
#
#   3. DYNAMIC RANGE COMPRESSION
#      The electrical range of an implant is ~30 dB. Speech spans ~80 dB.
#      A logarithmic mapping squeezes the acoustic range into the electrical one.


import numpy as np

# scipy.signal -- digital filter design and application
from scipy.signal import butter, sosfiltfilt


def extract_envelopes(channel_signals, sr, lp_cutoff=400.0):
    # Extracts the temporal envelope of each channel using two steps.
    #
    # ---- STEP 1: Half-wave rectification ----
    #
    #   The band-pass filtered signal alternates between positive and negative
    #   values as it oscillates. If we averaged positive and negative halves,
    #   they would cancel out to roughly zero, telling us nothing about loudness.
    #   Rectification fixes this by setting all negative values to zero,
    #   leaving only the positive half of each oscillation cycle.
    #
    #   np.maximum(a, b) returns an array where each element is the larger
    #   of the corresponding elements in a and b. Here b=0.0, so any element
    #   in channel_signals that is negative gets replaced with 0.0.
    #
    #   channel_signals has shape (n_electrodes, n_samples).
    #   np.maximum works element-wise across the entire 2D array.

    rectified = np.maximum(channel_signals, 0.0)

    # ---- STEP 2: Low-pass filter at 400 Hz ----
    #
    #   After rectification the signal still "wiggles" rapidly -- it goes up
    #   every positive half-cycle and hits zero every negative half-cycle.
    #   A low-pass filter with a 400 Hz cutoff smooths out oscillations faster
    #   than 400 Hz, leaving only the slower shape (the envelope).
    #
    #   Why 400 Hz?
    #   Speech rhythm (syllables, words) lives below ~20 Hz.
    #   Voicing pitch (F0) is roughly 100-300 Hz.
    #   A 400 Hz cutoff preserves both while removing the carrier oscillations.
    #   This is the standard value used in all clinical CI processors.
    #
    #   nyq = sr / 2.0 is the Nyquist frequency.
    #   scipy's butter() wants frequencies expressed as fractions of Nyquist,
    #   so the cutoff is given as lp_cutoff / nyq.
    #   butter(2, ...) designs a 2nd-order (12 dB/octave) Butterworth low-pass.
    #   output="sos" returns it in Second-Order Sections format (numerically stable).

    nyq = sr / 2.0
    sos = butter(2, lp_cutoff / nyq, btype="low", output="sos")

    # Pre-allocate an array of zeros with the same shape as rectified.
    # np.zeros_like(x) creates a zero-filled array with the same shape and
    # dtype as x, without copying any data.
    envelopes = np.zeros_like(rectified)

    # Filter each channel independently using a Python for loop.
    # range(rectified.shape[0]) generates integers from 0 to n_electrodes-1.
    # .shape[0] accesses the size of the first dimension (number of rows).
    for k in range(rectified.shape[0]):
        # sosfiltfilt applies the filter forward then backward (zero phase).
        # np.maximum(..., 0.0) clamps any tiny negative values that filtering
        # might have introduced back to zero.
        envelopes[k] = np.maximum(sosfiltfilt(sos, rectified[k]), 0.0)

    # .astype(np.float32) ensures memory efficiency by converting to 32-bit float.
    return envelopes.astype(np.float32)


def select_n_of_m(band_energies, n_active):
    # Implements the ACE (Advanced Combination Encoder) n-of-m strategy.
    #
    # 'band_energies' has shape (num_frames, n_electrodes).
    # Each row is one analysis frame; each column is one electrode channel.
    # The value in each cell is how much energy that channel has in that frame.
    #
    # The job of this function is to look at each frame (row) and mark only
    # the n_active cells with the highest energy as "active" (True).
    # All other cells are marked as "inactive" (False) and set to zero.
    #
    # WHY BOTHER?
    #   - Power: stimulating all 22 electrodes every cycle wastes battery.
    #   - Interference: nearby electrodes' electrical fields overlap.
    #     Fewer active electrodes means less channel interaction.
    #   - Salience: the loudest channels carry the most important speech information
    #     (formants, fricative noise). Quiet channels add little perceptual value.
    #
    # The set of active channels changes every frame because the spectrum changes
    # with the speech -- during /a/ the low-frequency formant channels dominate,
    # during /s/ the high-frequency channels dominate.

    # Unpack the shape tuple into two variables.
    # T = number of frames (rows), M = number of channels (columns).
    T, M     = band_energies.shape

    # Safety check: n_active cannot be larger than the total number of channels.
    n_active = min(n_active, M)

    # np.argpartition(array, k, axis) rearranges the indices so that
    # the k-th smallest element is in its correct sorted position, and
    # all elements before k are smaller (but not necessarily sorted among
    # themselves), all elements after k are larger (but not necessarily sorted).
    # This is faster than a full sort (O(n) vs O(n log n)) when we only need
    # the top n elements.
    #
    # We want the top n_active, so we partition at position (M - n_active).
    # [:, M - n_active:] takes everything from that position to the end of each
    # row, which gives the indices of the n_active largest values in each frame.
    top_indices   = np.argpartition(band_energies, M - n_active, axis=1)[:, M - n_active:]

    # Create the boolean mask, initially all False.
    selected_mask = np.zeros((T, M), dtype=bool)

    # For each frame t, mark the top n_active channel indices as True.
    # selected_mask[t, top_indices[t]] uses "fancy indexing":
    # top_indices[t] is an array of column indices, and this line sets all
    # those columns in row t to True simultaneously.
    for t in range(T):
        selected_mask[t, top_indices[t]] = True

    # np.where(condition, x, y) returns x where condition is True, y elsewhere.
    # Here it returns band_energies where the mask is True, 0.0 everywhere else.
    selected_energies = np.where(selected_mask, band_energies, 0.0)

    # Return both the boolean mask (used in Stage 4 to skip inactive channels)
    # and the zeroed-out energy matrix (used for compression below).
    return selected_mask, selected_energies.astype(np.float32)


def compress(envelopes, t_level=0.01, c_level=1.0, q=20.0):
    # Maps the acoustic envelope amplitude into the electrical stimulation range.
    #
    # THE PROBLEM:
    #   Normal hearing operates over a dynamic range of roughly 120 dB (from
    #   the threshold of hearing to the threshold of pain). Speech itself spans
    #   about 50-80 dB depending on environment. A cochlear implant electrode,
    #   however, only has a usable electrical range of about 20-30 dB -- the
    #   gap between the threshold current (T-level, below which nothing is heard)
    #   and the most comfortable level (C-level, above which stimulation is
    #   uncomfortably loud or even painful).
    #
    # THE SOLUTION:
    #   A logarithmic compression function maps the wide acoustic range onto
    #   the narrow electrical range. Logarithms are the natural choice because
    #   human loudness perception itself is roughly logarithmic (the dB scale
    #   is a log scale for this reason).
    #
    # THE FORMULA:
    #   electrical = T + (C - T) * log(1 + q*x) / log(1 + q)
    #
    #   x is the normalised acoustic amplitude (0 to 1).
    #   q=20 is the compression strength. Higher q = stronger compression.
    #   When x=0: result = T + (C-T)*0 = T  (threshold)
    #   When x=1: result = T + (C-T)*1 = C  (maximum comfortable level)
    #   In between, the curve is concave (compresses loud more than quiet).
    #
    # PARAMETERS:
    #   t_level -- electrical threshold (default 0.01 = 1% of full scale)
    #   c_level -- comfortable maximum (default 1.0 = full scale)
    #   q       -- compression exponent (default 20.0)

    # .max() returns the single largest value in the entire array.
    # If the maximum is zero or negative, the signal is silent -- return all zeros.
    env_max = envelopes.max()
    if env_max <= 0:
        return np.zeros_like(envelopes)

    # Divide by the maximum so all values are in the range [0, 1].
    # This normalisation makes the compression formula independent of
    # the absolute amplitude level of the input signal.
    normalised = envelopes / env_max

    # Apply the logarithmic compression formula.
    # np.log computes the natural logarithm (base e).
    # The entire formula is applied element-wise to the normalised array.
    electrical = t_level + (c_level - t_level) * np.log(1 + q * normalised) / np.log(1 + q)

    # Set values where the normalised amplitude is below 0.1% to exactly zero.
    # np.where(condition, x, y): where condition is True use x, otherwise use y.
    # This prevents very quiet signals from producing tiny non-zero currents
    # that could cause sub-threshold tingling sensations.
    electrical = np.where(normalised < 1e-3, 0.0, electrical)
    # 1e-3 is scientific notation for 0.001 (one tenth of one percent)

    # np.clip(array, min, max) clamps every value to stay within [min, max].
    # This ensures no output exceeds the safe electrical operating range.
    return np.clip(electrical, 0.0, c_level).astype(np.float32)
