# pulse_train.py  (Stage 5)
#
# ============================================================
# WHY BIPHASIC PULSES INSTEAD OF CONTINUOUS CURRENT?
# ============================================================
#
# The most intuitive way to stimulate a nerve would be to send a continuous
# current proportional to the signal amplitude. But this causes two problems:
#
#   1. ELECTRODE CORROSION: Sustained direct current causes electrochemical
#      reactions at the electrode surface. Over time this damages the electrode
#      and releases toxic byproducts into the cochlear fluid.
#
#   2. NEURAL ADAPTATION: Auditory nerve fibres stop firing if they are
#      stimulated continuously. They need brief, discrete pulses to respond
#      reliably every time.
#
# The solution is CHARGE-BALANCED BIPHASIC PULSES. Each pulse consists of
# two phases of equal charge but opposite polarity:
#
#   [--- cathodic phase ---] [gap] [+++ anodic phase +++]
#
#   The cathodic (negative) phase is the "stimulating" phase -- it makes the
#   nerve fibre membrane potential cross the threshold and fire (depolarisation).
#   The anodic (positive) phase immediately follows and delivers equal charge
#   in the opposite direction, returning the electrode to electrochemical
#   balance. Net charge delivered = 0.
#
# ============================================================
# WHY IS THE CATHODIC PHASE FIRST?
# ============================================================
#
# Auditory nerve fibres respond more efficiently to negative (cathodic) current.
# The cathodic phase pushes positive sodium ions INTO the cell, rapidly
# raising the membrane potential above the threshold for an action potential.
# The anodic phase that follows does not re-excite the fibre because it is
# in its refractory period (a brief recovery window after firing).
#
# ============================================================
# THE CIS STRATEGY (Continuous Interleaved Sampling)
# ============================================================
#
# If two adjacent electrodes fired simultaneously, their electrical fields
# would add together and stimulate a wider patch of nerve fibres than intended.
# This blurs the spectral information.
#
# CIS (Wilson et al., 1991) prevents this by staggering the pulses in time.
# The 22 electrodes take turns, one at a time, in rapid succession. Electrode 0
# fires first, then electrode 1, then electrode 2, and so on. By the time
# electrode 21 fires, electrode 0 is ready to fire again. The interval between
# pulses on the same electrode is called the stimulation period (1/rate seconds).
# At 900 pulses/second per channel, the stimulation period is ~1.1 ms.
# Each electrode fires during its own 1.1 ms/22 = ~50 us slot, with no overlap.


import numpy as np
import soundfile as sf   # soundfile handles reading and writing WAV files

# These constants match typical clinical CI processor settings.
# They are defined at the module level so any function in this file can use them,
# and so they appear clearly at the top rather than buried inside a function.
PHASE_DURATION_US = 25.0    # microseconds per phase (typical range: 18-75 us)
INTERPHASE_GAP_US = 8.0     # brief silence between the two phases (reduces charge injection)
STIM_RATE_HZ      = 900.0   # pulses per second per electrode channel


def make_biphasic_pulse(amplitude, phase_samples, gap_samples):
    # Builds a single biphasic pulse as a short 1D NumPy array.
    # The array looks like this (schematically):
    #
    #  [-amp, -amp, ..., -amp,  0, 0, ..., 0,  +amp, +amp, ..., +amp]
    #  |---- phase_samples ----|-- gap_samples --|---- phase_samples --|
    #
    # Total length = 2 * phase_samples + gap_samples.
    #
    # 'amplitude' is a value between 0.0 and 1.0 (normalised current level).
    # Phase amplitudes are negative for cathodic, positive for anodic.

    # np.zeros(n) creates a 1D array of n zeros (float64 by default).
    # The total number of samples needed is: cathodic + gap + anodic.
    pulse = np.zeros(2 * phase_samples + gap_samples)

    # Slice assignment: sets elements from index 0 to phase_samples-1 to -amplitude.
    # pulse[:phase_samples] is shorthand for pulse[0:phase_samples].
    pulse[:phase_samples]               = -amplitude   # cathodic (stimulating) phase

    # Sets elements from (phase_samples + gap_samples) to the end to +amplitude.
    # pulse[n:] is shorthand for pulse[n:len(pulse)].
    pulse[phase_samples + gap_samples:] =  amplitude   # anodic (balancing) phase

    return pulse


def generate_pulse_trains(electrode_currents, sr, hop_len, n_electrodes,
                          stim_rate=STIM_RATE_HZ,
                          phase_us=PHASE_DURATION_US,
                          gap_us=INTERPHASE_GAP_US):
    # Converts the per-frame amplitude schedule from Stage 4 into a sampled
    # waveform of biphasic pulses, one waveform per electrode.
    #
    # 'electrode_currents' has shape (num_frames, n_electrodes).
    # Each value is the normalised current (0 to 1) that electrode should deliver
    # during that frame. This function translates each value into an actual
    # biphasic pulse placed at the right time position in the output waveform.
    #
    # OUTPUT:
    #   'pulse_trains' has shape (n_electrodes, total_samples).
    #   Each row is a time-domain waveform for one electrode.
    #   Most samples are zero (silence between pulses); pulses appear briefly.

    T             = electrode_currents.shape[0]   # number of frames
    total_samples = T * hop_len                   # total length of the output waveform

    # Convert pulse durations from microseconds to sample counts.
    # 1 microsecond = 1e-6 seconds.
    # Multiplying by sr (samples/second) gives samples.
    # round() rounds to the nearest integer; int() converts float to int.
    # max(1, ...) ensures at least 1 sample even if the duration is very short.
    phase_samp = max(1, int(round(phase_us * 1e-6 * sr)))
    gap_samp   = max(0, int(round(gap_us   * 1e-6 * sr)))
    pulse_len  = 2 * phase_samp + gap_samp

    # Pre-allocate the output array as all zeros.
    # Shape (n_electrodes, total_samples): rows are electrodes, columns are time.
    pulse_trains = np.zeros((n_electrodes, total_samples), dtype=np.float32)

    # Compute how many samples fit in one stimulation period (1/rate seconds).
    # At 900 Hz: stim_interval = round(16000 / 900) = 18 samples.
    stim_interval = int(round(sr / stim_rate))

    # Outer loop: iterate over frames
    for t in range(T):
        frame_start = t * hop_len   # sample index where this frame starts

        # Inner loop: iterate over electrodes
        for k in range(n_electrodes):
            amp = float(electrode_currents[t, k])

            # Skip silent electrodes to avoid writing zero-amplitude pulses
            if amp < 1e-6:
                continue

            # CIS interleaving: each electrode k gets its own time slot within
            # the stimulation interval. Electrode 0 fires at offset 0, electrode 1
            # fires stim_interval/n_electrodes samples later, etc.
            # int(k * stim_interval / n_electrodes) computes the sample offset.
            offset      = int(k * stim_interval / n_electrodes)
            pulse_start = frame_start + offset

            # Safety check: make sure the pulse fits within the allocated array.
            if pulse_start + pulse_len > total_samples:
                continue

            # Build the pulse and add it to the electrode's waveform.
            # '+=' adds to existing values (instead of overwriting), so if two
            # frames schedule pulses at overlapping times, they accumulate.
            pulse = make_biphasic_pulse(amp, phase_samp, gap_samp)
            pulse_trains[k, pulse_start: pulse_start + pulse_len] += pulse
            # pulse_trains[k, a:b] is a slice of row k from column a to b-1.

    return pulse_trains


def synthesise_vocoder(envelopes, centre_freqs, sr, carrier="sine"):
    # Produces an acoustic simulation of what the CI-processed speech sounds like.
    #
    # A vocoder works by:
    #   1. Taking a carrier signal at each channel's centre frequency
    #   2. Multiplying (amplitude-modulating) it by that channel's envelope
    #   3. Summing all channels together
    #
    # The result has the temporal RHYTHM of the original speech (captured in the
    # envelopes) but only the coarse SPECTRAL shape of the 22-channel encoding.
    # High-frequency detail and fine pitch information are lost -- which is
    # exactly what happens with a real cochlear implant.
    #
    # CARRIER OPTIONS:
    #   "sine": a clean sine wave at the centre frequency. This gives the
    #           clearest demonstration of which frequency bands are active.
    #   "noise": white Gaussian noise. When amplitude-modulated, this creates
    #            a "buzzy" sound more representative of real electric hearing,
    #            where spectral fine structure is absent.

    M, N   = envelopes.shape   # M = number of channels, N = number of samples
    t      = np.arange(N) / sr # time axis in seconds: [0, 1/sr, 2/sr, ..., (N-1)/sr]
    output = np.zeros(N, dtype=np.float32)

    # Process each channel independently and add its contribution to the output.
    for k in range(M):

        if carrier == "sine":
            # Standard sine wave formula: sin(2 * pi * frequency * time)
            carrier_wave = np.sin(2 * np.pi * centre_freqs[k] * t)
        else:
            # np.random.randn(N) generates N samples of Gaussian white noise
            # (mean=0, standard deviation=1). Each call gives a new random sequence.
            carrier_wave = np.random.randn(N)

        # Amplitude modulation: multiply the carrier by the envelope.
        # envelopes[k] has shape (N,) -- one value per sample.
        # The multiplication is element-wise: each sample of the carrier is
        # scaled by the corresponding envelope value.
        # .astype(np.float32) ensures the data type stays consistent.
        output += (envelopes[k] * carrier_wave).astype(np.float32)

    # Find the peak absolute value across the entire output signal.
    peak = np.max(np.abs(output))

    # Normalise to 90% of full scale to prevent clipping.
    # Only do this if the peak is non-zero (avoid division by zero).
    if peak > 0:
        output = output / peak * 0.9

    return output


def save_audio(signal, sr, filepath):
    # Writes the audio signal to a WAV file.
    # soundfile.write(filepath, data, samplerate) handles all the file format
    # details automatically. It works with .wav, .flac, .ogg, etc.
    # The file is created (or overwritten) at the given filepath.
    sf.write(filepath, signal, sr)
