# filterbank.py  (Stage 2)
#
# A healthy cochlea physically separates sound by frequency. The basilar
# membrane inside the cochlea vibrates maximally at different positions
# depending on the incoming frequency -- high frequencies near the base
# of the spiral, low frequencies near the apex. This is called tonotopy.
#
# A cochlear implant bypasses the damaged cochlea and stimulates the auditory
# nerve directly with electrodes. Each electrode sits at a different position
# along the cochlea and is therefore responsible for a different frequency range.
# This module replicates the cochlea's frequency-separation job in software.
#
# HOW IT WORKS:
#   One band-pass filter is designed per electrode.
#   A band-pass filter lets through only a specific range of frequencies
#   and blocks (attenuates) everything outside that range.
#   After filtering, the signal in each channel contains only the frequencies
#   "assigned" to that electrode.
#
# WHY ERB SPACING?
#   We could space the electrode centre frequencies linearly (e.g. 250, 620,
#   990, ... Hz) or logarithmically. But neither matches how the human auditory
#   system actually resolves pitch. The ERB (Equivalent Rectangular Bandwidth)
#   scale is a psychoacoustic scale derived from experiments measuring how well
#   humans distinguish nearby frequencies. Frequencies below ~500 Hz are packed
#   more tightly on this scale because the ear is better at resolving them.
#   Using ERB spacing gives the best perceptual coverage across the cochlea.


# numpy -- arrays and math
import numpy as np

# scipy.signal contains digital filter design tools.
# butter   -- designs a Butterworth filter (maximally flat passband)
# sosfiltfilt -- applies a filter forward and backward for zero phase delay
# sosfreqz    -- computes the frequency response of a filter (for plotting)
from scipy.signal import butter, sosfiltfilt, sosfreqz


def hz_to_erb(f):
    # Converts a frequency value from Hz to the ERB-rate scale.
    # The formula is from Moore & Glasberg (1983), a landmark psychoacoustics paper.
    #
    # np.log10(x) computes the base-10 logarithm.
    # 4.37e-3 is scientific notation for 0.00437.
    # The "+1" inside the log prevents log(0) when f=0.
    #
    # Higher ERB-rate numbers correspond to higher frequencies,
    # but the scale is compressed at high frequencies (they are spaced farther apart).
    return 21.4 * np.log10(4.37e-3 * f + 1.0)


def erb_to_hz(erb):
    # Inverse function: converts an ERB-rate value back to Hz.
    # 10 ** (erb / 21.4) raises 10 to the power of (erb/21.4).
    # Subtracting 1 and dividing by 4.37e-3 reverses the hz_to_erb formula.
    return (10 ** (erb / 21.4) - 1.0) / 4.37e-3


def erb_spaced_frequencies(f_low, f_high, n_bands):
    # Returns a list of n_bands centre frequencies spaced evenly on the ERB scale.
    # "Evenly spaced on the ERB scale" means the perceptual distance between
    # adjacent channels is constant, not the physical distance in Hz.
    #
    # Steps:
    #   1. Convert the low and high frequency limits to ERB-rate units
    #   2. Create n_bands evenly-spaced points between those ERB-rate limits
    #   3. Convert each ERB-rate point back to Hz
    #
    # np.linspace(start, stop, num) returns 'num' equally-spaced values
    # from 'start' to 'stop', inclusive of both endpoints.

    erb_low  = hz_to_erb(f_low)
    erb_high = hz_to_erb(f_high)

    # erbs is a 1D array of n_bands ERB-rate values, equally spaced
    erbs = np.linspace(erb_low, erb_high, n_bands)

    # List comprehension: [erb_to_hz(e) for e in erbs] iterates through each
    # value e in the erbs array and calls erb_to_hz on it.
    # np.array(...) converts the resulting Python list into a NumPy array.
    return np.array([erb_to_hz(e) for e in erbs])


def build_filterbank(sr, n_electrodes=22, f_low=250.0, f_high=8000.0):
    # Designs the complete filter bank -- one filter per electrode.
    #
    # WHAT IS A BUTTERWORTH FILTER?
    #   It is a type of filter whose frequency response is as flat as possible
    #   in the passband (the range of frequencies it lets through). "4th-order"
    #   means it uses four stages, giving a steeper roll-off than lower orders --
    #   frequencies outside the passband are attenuated more aggressively.
    #
    # WHAT IS SOS FORMAT?
    #   IIR filters can be represented in several mathematical forms.
    #   SOS (Second-Order Sections) decomposes the filter into a cascade of
    #   simple 2nd-order building blocks. This is numerically more stable than
    #   the direct polynomial form, especially for high-order filters.
    #   scipy's butter() function returns SOS when output="sos" is specified.
    #
    # WHAT IS ERB BANDWIDTH?
    #   Each filter needs a bandwidth (how wide its passband is).
    #   The ERB bandwidth at a given centre frequency cf is:
    #       ERB_bw = 24.7 * (4.37e-3 * cf + 1.0)
    #   This formula says that the bandwidth grows with frequency -- a filter
    #   centred at 4000 Hz is much wider than one centred at 300 Hz.
    #   Using ERB bandwidth ensures the filter widths match the ear's resolution.

    centre_freqs = erb_spaced_frequencies(f_low, f_high, n_electrodes)

    # nyq is the Nyquist frequency -- the maximum frequency that can exist
    # in a digital signal sampled at 'sr' samples per second.
    # Any frequency above nyq would be aliased (falsely represented).
    nyq = sr / 2.0

    # sos_filters will be a Python list; each element is one filter's SOS array.
    sos_filters = []

    # Iterate over every centre frequency in the bank.
    # 'cf' takes the value of each centre frequency in turn.
    for cf in centre_freqs:

        # Calculate how wide this filter should be
        erb_bw = 24.7 * (4.37e-3 * cf + 1.0)

        # Lower edge of the passband: half-bandwidth below the centre
        # max(..., 1.0) prevents the lower edge from going below 1 Hz
        f_lo = max(cf - erb_bw / 2.0, 1.0)

        # Upper edge of the passband: half-bandwidth above the centre
        # min(..., nyq - 1.0) prevents the upper edge from reaching Nyquist
        f_hi = min(cf + erb_bw / 2.0, nyq - 1.0)

        # butter() designs the filter.
        # Arguments: order=4, critical frequencies, btype="bandpass", output format
        # The frequencies must be given as fractions of Nyquist (between 0 and 1),
        # which is why f_lo and f_hi are divided by nyq.
        # [f_lo/nyq, f_hi/nyq] is a Python list of two values (the band edges).
        sos = butter(4, [f_lo / nyq, f_hi / nyq], btype="bandpass", output="sos")

        # Append adds the new filter SOS array to the end of the list
        sos_filters.append(sos)

    # Returns two things: the list of filter coefficients and the centre frequencies
    return sos_filters, centre_freqs


def apply_filterbank(signal, sos_filters):
    # Filters the entire audio signal through every channel filter.
    #
    # sosfiltfilt applies the filter TWICE -- once forward in time and
    # once backward. This "zero-phase" approach ensures the filter does not
    # shift the signal in time, which would misalign the envelopes in Stage 3.
    # The cost is that the computation takes twice as long, but for the
    # signal lengths we deal with (<< 1 minute) this is negligible.
    #
    # The output is a 2D array where:
    #   - Each ROW is one electrode channel
    #   - Each COLUMN is one time sample
    #   So channel_signals[k, n] is the amplitude of channel k at sample n.

    n_bands = len(sos_filters)   # how many filters (= how many electrodes)

    # np.zeros((n_bands, len(signal)), dtype=np.float32) creates a 2D array
    # filled with zeros. (n_bands, len(signal)) is a tuple specifying the shape.
    # This pre-allocates memory before the loop, which is faster than growing
    # the array one row at a time.
    channel_signals = np.zeros((n_bands, len(signal)), dtype=np.float32)

    # enumerate(sos_filters) gives (index, value) pairs.
    # 'k' is the electrode index, 'sos' is the filter for that electrode.
    for k, sos in enumerate(sos_filters):
        # sosfiltfilt(sos, signal) returns the filtered signal as a float64 array.
        # .astype(np.float32) converts it to float32 for memory efficiency.
        # channel_signals[k] accesses the k-th row and assigns the result to it.
        channel_signals[k] = sosfiltfilt(sos, signal).astype(np.float32)

    return channel_signals


def fft_filterbank(frames, sr, n_electrodes=22, f_low=250.0, f_high=8000.0, n_fft=512):
    # An FFT-based alternative to the IIR filter bank.
    # Instead of filtering the full signal, this function works on the already-framed data
    # from Stage 1 and computes how much energy each frame has in each frequency band.
    #
    # WHY USE THIS ALONGSIDE THE IIR FILTER BANK?
    #   The IIR filter bank (apply_filterbank) gives smooth continuous waveforms per
    #   channel, which are needed to extract the temporal envelopes in Stage 3.
    #   But the ACE n-of-m selection in Stage 3 needs per-FRAME energy values --
    #   one number per channel per frame. That is exactly what this function returns.
    #   Computing FFTs on short frames is much faster than running IIR filters for
    #   this purpose.
    #
    # HOW FFT WORKS (briefly):
    #   The FFT (Fast Fourier Transform) takes a time-domain signal of N samples
    #   and returns N complex numbers representing the amplitude and phase at N
    #   evenly-spaced frequencies. np.fft.rfft uses the fact that the input is
    #   real-valued to return only N//2 + 1 unique frequency bins.

    centre_freqs = erb_spaced_frequencies(f_low, f_high, n_electrodes)

    # np.fft.rfftfreq(n_fft, d=1.0/sr) returns the actual Hz values corresponding
    # to each FFT bin when the signal has sample rate sr.
    # d=1.0/sr is the sampling period (seconds per sample).
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)   # shape: (n_fft//2 + 1,)

    # Compute the ERB bandwidth for every centre frequency at once.
    # This is a vectorised operation -- numpy applies the formula to all 22
    # values in centre_freqs simultaneously, returning an array of 22 bandwidths.
    erb_bw  = 24.7 * (4.37e-3 * centre_freqs + 1.0)
    f_lows  = centre_freqs - erb_bw / 2.0   # lower edge of each channel's band
    f_highs = centre_freqs + erb_bw / 2.0   # upper edge of each channel's band

    # np.fft.rfft(frames, n=n_fft) computes the FFT of every row in 'frames'
    # simultaneously. The result has shape (num_frames, n_fft//2 + 1).
    # np.abs(...) takes the magnitude (throws away the phase), giving the
    # amplitude spectrum. We only need amplitude, not phase, for envelope detection.
    spec = np.abs(np.fft.rfft(frames, n=n_fft))   # shape: (num_frames, n_fft//2 + 1)

    T             = frames.shape[0]   # number of frames (.shape[0] is the first dimension)
    band_energies = np.zeros((T, n_electrodes), dtype=np.float32)

    # zip(f_lows, f_highs) pairs each lower edge with its corresponding upper edge.
    # enumerate gives (index, (lower, upper)) -- 'k' is the channel index.
    for k, (fl, fh) in enumerate(zip(f_lows, f_highs)):

        # Boolean mask: True wherever the FFT frequency bin falls within this channel's band.
        # (freqs >= fl) returns an array of True/False values, one per frequency bin.
        # The & operator combines two boolean arrays element-wise (logical AND).
        mask = (freqs >= fl) & (freqs < fh)

        # mask.any() returns True if at least one element of mask is True.
        # This guards against edge cases where a very narrow band has no FFT bins.
        if mask.any():
            # spec[:, mask] selects all rows (all frames) but only the columns
            # (FFT bins) where mask is True. This gives a sub-matrix of shape
            # (num_frames, num_bins_in_band).
            # .mean(axis=1) averages across columns (bins), giving one value per frame.
            # The result is stored in column k of band_energies.
            band_energies[:, k] = spec[:, mask].mean(axis=1)

    # Returns:
    #   band_energies -- shape (num_frames, n_electrodes), energy per channel per frame
    #   centre_freqs  -- shape (n_electrodes,), the centre frequency of each channel
    return band_energies, centre_freqs
