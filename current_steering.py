# current_steering.py  (Stage 4)
#
# This is the central contribution of the project.
#
# ============================================================
# BACKGROUND: WHY DO WE NEED VIRTUAL CHANNELS?
# ============================================================
#
# A Cochlear Nucleus implant has 22 physical electrodes arranged along
# the cochlea. Each electrode stimulates the auditory nerve fibres near it,
# and each group of fibres corresponds to a specific frequency.
# So the device can create AT MOST 22 distinct pitch sensations.
#
# 22 channels sounds like a lot, but consider: a piano has 88 keys.
# Normal hearing can distinguish thousands of pitches. Research shows
# that CI users typically have effective spectral resolution equivalent to
# only 4-8 truly independent channels, because the electrical fields from
# adjacent electrodes overlap and blur together.
#
# Virtual channels are a way to create intermediate pitch percepts
# WITHOUT adding more physical electrodes.
#
# ============================================================
# HOW CURRENT STEERING CREATES VIRTUAL CHANNELS
# ============================================================
#
# If only electrode k is active, the electrical field peaks AT electrode k.
# If only electrode k+1 is active, the field peaks AT electrode k+1.
#
# Now suppose BOTH are active simultaneously, but electrode k receives 70%
# of the current and electrode k+1 receives 30%. The electrical fields from
# both electrodes add together, and the combined field peaks at a position
# 30% of the way from electrode k toward electrode k+1.
#
# The auditory nerve fibres at that intermediate location are maximally excited.
# The brain perceives a pitch BETWEEN the pitches of the two electrodes.
# This is a virtual channel.
#
# The steering coefficient ALPHA (a) controls the split:
#
#     Current to electrode k   = (1 - a) * total_amplitude
#     Current to electrode k+1 =      a  * total_amplitude
#
#   a = 0.00 --> all current to k       --> pitch of physical electrode k
#   a = 0.25 --> 75% to k, 25% to k+1  --> pitch 1/4 of the way between them
#   a = 0.50 --> 50/50 split            --> pitch exactly between the two
#   a = 0.75 --> 25% to k, 75% to k+1  --> pitch 3/4 of the way between them
#   a = 1.00 --> all current to k+1     --> pitch of physical electrode k+1
#
# With 22 electrodes, there are 21 adjacent pairs.
# With 5 alpha steps per pair (0.0, 0.25, 0.5, 0.75, 1.0), that is 5 positions
# per gap (but shared endpoints means we don't double-count), giving about 85
# total virtual channel positions from 22 physical electrodes.


import numpy as np


def build_virtual_channel_map(n_electrodes, n_steps=5):
    # Creates a lookup table listing every virtual channel position.
    # Each entry in the list is a Python dictionary (a key-value store)
    # describing one virtual channel.
    #
    # WHY A DICTIONARY?
    #   Dictionaries let us access values by name ("elec_low") instead of
    #   by position ([0]). This makes the code easier to read and debug
    #   because the field names are self-documenting.
    #
    # STRUCTURE of each entry:
    #   {
    #     "vc_id"    : unique integer ID of this virtual channel,
    #     "elec_low" : index of the lower physical electrode (0-based),
    #     "elec_high": index of the higher physical electrode,
    #     "alpha"    : the steering coefficient for this position
    #   }

    # np.linspace(0.0, 1.0, n_steps) creates n_steps values from 0 to 1.
    # For n_steps=5: [0.0, 0.25, 0.5, 0.75, 1.0]
    alphas = np.linspace(0.0, 1.0, n_steps)

    vc_map = []   # start with an empty Python list
    vc_id  = 0    # counter incremented for each new virtual channel

    # Outer loop: iterate over each adjacent electrode pair.
    # range(n_electrodes - 1) generates 0, 1, 2, ..., 20 for 22 electrodes.
    # k is the index of the LOWER electrode in each pair.
    for k in range(n_electrodes - 1):

        # Inner loop: generate each alpha value for this pair.
        for a in alphas:

            # Special case: alpha=1.0 for a non-final pair would place the
            # virtual channel exactly AT electrode k+1. But that same position
            # will also be generated as alpha=0.0 of the NEXT pair (k+1, k+2).
            # Skipping it here prevents counting that electrode position twice.
            if a == 1.0 and k < n_electrodes - 2:
                continue   # 'continue' jumps to the next iteration of the loop

            # float(a) converts the numpy scalar to a plain Python float.
            # This avoids potential type issues when the dictionary is used later.
            vc_map.append({
                "vc_id"    : vc_id,
                "elec_low" : k,
                "elec_high": k + 1,
                "alpha"    : float(a),
            })
            vc_id += 1   # increment the ID counter (+= 1 is shorthand for vc_id = vc_id + 1)

    # vc_map is a list of dictionaries, one per virtual channel.
    # With 22 electrodes and n_steps=5: 21 pairs * 4 interior + 22 endpoints = 85 entries.
    return vc_map


def find_best_virtual_channel(freq_hz, centre_freqs, vc_map):
    # Given a frequency in Hz, find which virtual channel best represents it.
    #
    # The approach:
    #   1. Find which pair of physical electrodes brackets this frequency
    #   2. Compute the ideal alpha using linear interpolation
    #   3. Scan the vc_map for the entry with the closest matching alpha
    #
    # WHY LINEAR INTERPOLATION?
    #   The centre frequencies are ERB-spaced, not linear. So if a band's
    #   centre frequency falls at, say, 40% of the distance between electrode
    #   k (at 900 Hz) and electrode k+1 (at 1100 Hz), we want alpha = 0.40.
    #   Linear interpolation computes exactly this fractional position.

    # np.clip(value, min, max) ensures freq_hz stays within the valid range.
    # If it's below the lowest centre frequency, clip to the lowest.
    # If it's above the highest, clip to the highest.
    # float() converts the numpy scalar to a plain Python float.
    freq_hz = float(np.clip(freq_hz, centre_freqs[0], centre_freqs[-1]))
    # centre_freqs[-1] is Python's way of accessing the LAST element of an array.

    # np.searchsorted(array, value, side="right") returns the index where
    # 'value' would be inserted to keep 'array' sorted.
    # Subtracting 1 gives the index of the last element <= value, i.e. the
    # lower electrode of the bracketing pair.
    # np.clip(..., 0, len-2) prevents the index from going out of bounds.
    k = int(np.clip(
        np.searchsorted(centre_freqs, freq_hz, side="right") - 1,
        0,
        len(centre_freqs) - 2
    ))

    # Linear interpolation formula:
    #   alpha = (target - lower) / (upper - lower)
    # This gives the fraction of the way from electrode k to electrode k+1.
    alpha = (freq_hz - centre_freqs[k]) / (centre_freqs[k + 1] - centre_freqs[k])
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Scan the virtual channel map for the closest match.
    # We only consider entries where elec_low == k (same electrode pair),
    # then pick the one with the smallest alpha difference.
    best_vc   = None
    best_diff = float("inf")   # initialise to infinity so any real diff is smaller

    for vc in vc_map:
        if vc["elec_low"] == k:        # right electrode pair?
            diff = abs(vc["alpha"] - alpha)
            if diff < best_diff:       # closer than the best seen so far?
                best_diff = diff
                best_vc   = vc

    return best_vc   # returns a dictionary, or None if nothing was found


def compute_electrode_currents(compressed_amps, centre_freqs, vc_map,
                               n_electrodes, selected_mask):
    # This is the heart of the current steering algorithm.
    #
    # For every time frame and every active channel, this function:
    #   1. Finds the virtual channel that best represents that frequency band
    #   2. Splits the amplitude between the two physical electrodes using alpha
    #   3. Adds the current contributions to the electrode current matrix
    #
    # 'compressed_amps' has shape (num_frames, n_electrodes).
    # 'selected_mask'   has shape (num_frames, n_electrodes), dtype bool.
    #
    # The output 'electrode_currents' has shape (num_frames, n_electrodes).
    # Note: n_electrodes in both input and output -- the number of PHYSICAL
    # electrodes. The virtual channel logic happens inside, but the result
    # is always expressed as currents on physical electrodes.

    # Unpack the shape of compressed_amps into T (frames) and M (channels).
    T, M               = compressed_amps.shape

    # Pre-allocate the output matrix as all zeros.
    electrode_currents = np.zeros((T, n_electrodes), dtype=np.float32)

    # stim_events stores a record of every firing event for debugging/logging.
    # It is a list of lists: stim_events[t] contains all events in frame t.
    stim_events = []

    # Outer loop: iterate over every frame
    for t in range(T):
        frame_events = []   # events for this specific frame

        # Inner loop: iterate over every channel
        for ch in range(M):

            # If this channel was NOT selected by n-of-m, skip it.
            # 'not selected_mask[t, ch]' means "the mask is False here".
            if not selected_mask[t, ch]:
                continue   # jump immediately to the next channel

            # Get the compressed amplitude for this channel in this frame.
            # float() converts the numpy scalar to a plain Python float.
            amplitude = float(compressed_amps[t, ch])

            # Skip channels with negligible amplitude (below 0.000001).
            # 1e-6 is scientific notation for 0.000001.
            if amplitude < 1e-6:
                continue

            # Find the virtual channel for this frequency band.
            # centre_freqs[ch] is the centre frequency of channel ch in Hz.
            vc = find_best_virtual_channel(centre_freqs[ch], centre_freqs, vc_map)

            # Defensive check: if no virtual channel was found, skip.
            if vc is None:
                continue

            # Extract the alpha and electrode indices from the dictionary.
            # vc["alpha"] is how to access a dictionary value by key name.
            a       = vc["alpha"]
            elec_lo = vc["elec_low"]
            elec_hi = vc["elec_high"]

            # Apply the current steering split formula:
            #   lower electrode gets (1 - alpha) fraction
            #   upper electrode gets (alpha) fraction
            # The two always sum to exactly 1.0 * amplitude, preserving total charge.
            i_low  = (1.0 - a) * amplitude
            i_high =       a   * amplitude

            # '+=' adds to whatever was already there from previous channels.
            # Multiple channels can contribute current to the same electrode,
            # so we accumulate rather than overwrite.
            electrode_currents[t, elec_lo] += i_low
            electrode_currents[t, elec_hi] += i_high

            # Log this stimulation event as a dictionary for later inspection.
            frame_events.append({
                "channel"  : ch,
                "vc_id"    : vc["vc_id"],
                "elec_low" : elec_lo,
                "elec_high": elec_hi,
                "alpha"    : a,
                "i_low"    : i_low,
                "i_high"   : i_high,
            })

        stim_events.append(frame_events)

    # np.clip ensures no electrode exceeds the safe range [0, 1].
    # Multiple channels contributing to the same electrode can sum above 1.0,
    # and clipping prevents over-stimulation.
    electrode_currents = np.clip(electrode_currents, 0.0, 1.0)

    return electrode_currents, stim_events


def estimate_pitch(electrode_currents, centre_freqs):
    # Estimates what pitch the listener would perceive in each frame.
    #
    # When multiple electrodes fire at once with different current levels,
    # the perceived pitch tends toward the CURRENT-WEIGHTED AVERAGE of their
    # centre frequencies. This is similar to a centre-of-mass calculation.
    #
    # Example: electrode 5 (at 600 Hz) carries 70% of the total current,
    # electrode 6 (at 700 Hz) carries 30%.
    # Estimated pitch = 0.70 * 600 + 0.30 * 700 = 420 + 210 = 630 Hz.
    #
    # This function computes that weighted average for every frame at once
    # using matrix operations instead of explicit loops.

    # .sum(axis=1, keepdims=True) sums across columns (axis=1 = across electrodes)
    # keepdims=True keeps the result as a column vector of shape (T, 1)
    # instead of a flat vector of shape (T,). This is needed for broadcasting below.
    total = electrode_currents.sum(axis=1, keepdims=True)

    # Guard against frames where all electrodes have zero current.
    # np.where(condition, x, y): where total < 1e-9, use 1.0; otherwise use total.
    # Dividing by 1.0 instead of 0.0 avoids a division-by-zero error.
    total = np.where(total < 1e-9, 1.0, total)

    # Normalise: divide each electrode's current by the total current in that frame.
    # The result 'weights' has the same shape as electrode_currents.
    # In each row, the values sum to 1.0 (they are fractional weights).
    weights = electrode_currents / total

    # centre_freqs[None, :] reshapes centre_freqs from shape (n_electrodes,)
    # to shape (1, n_electrodes) so it can broadcast across all frames.
    # Multiplying weights (T, n_electrodes) by centre_freqs (1, n_electrodes)
    # gives (T, n_electrodes) element-wise products.
    # .sum(axis=1) then sums across electrodes, giving the weighted average
    # frequency for each frame as a 1D array of shape (T,).
    pitch = (weights * centre_freqs[None, :]).sum(axis=1)

    return pitch.astype(np.float32)
