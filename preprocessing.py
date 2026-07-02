# preprocessing.py  (Stage 1)
#
# Before any signal processing can happen, the raw audio needs to be prepared.
# This module does three things, in this exact order:
#
#   1. Load a real audio file from disk, OR generate a synthetic test signal
#   2. Apply pre-emphasis  -- a filter that boosts high frequencies slightly
#   3. Chop the signal into short overlapping chunks called "frames"
#
# Why do we need frames?
#   Speech is not stationary -- a person says "ah" and then "ss" and the
#   frequency content changes completely between those two sounds. If the
#   filter bank analysed the entire recording at once, all that detail would
#   be averaged away. By cutting it into 16 ms frames, each frame captures
#   one "snapshot" of the spectrum, and the pipeline processes them one by one.
#
# Why 16 ms specifically?
#   It is short enough that the signal inside the frame looks approximately
#   stable (speech changes on a scale of 50-200 ms for individual phonemes),
#   but long enough that a 512-point FFT can resolve the frequencies we care
#   about (250-8000 Hz needs at least ~125 us resolution, which 16 ms gives).


# numpy is the fundamental library for numerical arrays in Python.
# Almost every scientific computation in this project uses np arrays.
import numpy as np

# librosa is a library specifically designed for audio analysis.
# It handles loading files, resampling, and many other audio tasks.
import librosa


def load_audio(filepath, target_sr=16000):
    # This function loads any supported audio file and standardises it.
    #
    # 'filepath' is a string containing the path to the audio file,
    # for example: "C:/Users/jishu/Desktop/speech.wav"
    #
    # 'target_sr=16000' means the default sample rate is 16000 Hz (16 kHz).
    # The "=" makes it an optional argument -- if the caller doesn't pass it,
    # Python automatically uses 16000.
    #
    # librosa.load does two useful things automatically:
    #   - If the file has stereo (left+right channels), mono=True mixes
    #     them down to a single channel, which is all a CI needs.
    #   - If the file was recorded at 44100 Hz (CD quality), sr=target_sr
    #     resamples it down to 16000 Hz by removing samples in a smart way.
    #
    # The function returns two values:
    #   signal -- a 1D array of amplitude values between -1.0 and +1.0
    #   sr     -- the actual sample rate after resampling (should equal target_sr)

    signal, sr = librosa.load(filepath, sr=target_sr, mono=True)

    # .astype(np.float32) converts the array to 32-bit floating point.
    # librosa already returns floats, but this guarantees the type is
    # consistent with every other array in the pipeline.
    return signal.astype(np.float32), sr


def generate_speech_like_signal(duration=1.0, sr=16000):
    # When no real audio file is provided, this function manufactures a
    # synthetic signal that shares the key properties of a spoken vowel.
    #
    # Real speech is built from:
    #   - A fundamental frequency F0 (~100-300 Hz in adults): the rate at
    #     which the vocal cords open and close, heard as the "pitch" of a voice.
    #   - Formants F1, F2, F3...: resonance frequencies of the throat, mouth
    #     and lips. Their positions determine which vowel is being spoken.
    #     For /a/ (as in "father"), F1 is around 700 Hz and F2 around 1220 Hz.
    #   - A small amount of broadband noise to simulate the noise floor.
    #
    # The signal is a weighted sum of three sine waves plus noise.
    # The weights (0.50, 0.35, 0.25) reflect the fact that the fundamental
    # is loudest and higher harmonics are progressively softer.

    # np.linspace(start, stop, num) creates 'num' evenly-spaced values
    # from 'start' to 'stop'. Here it creates a time axis in seconds.
    # int(sr * duration) converts the float result to an integer sample count.
    # endpoint=False means the stop value itself is NOT included, which avoids
    # a repeated sample when looping the signal.
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    f0  = 120.0   # fundamental -- how fast the vocal cords vibrate (Hz)
    f1  = 700.0   # first formant -- resonance of the lower vocal tract (Hz)
    f2  = 1220.0  # second formant -- resonance of the upper vocal tract (Hz)

    # np.sin(2 * np.pi * f * t) is the standard formula for a sine wave.
    # 2*pi converts from cycles to radians (sine takes radians as input).
    # Multiplying by t makes the frequency f cycles happen per second.
    # np.random.randn(len(t)) generates Gaussian (bell-curve) random noise,
    # one sample for each sample in t.
    signal = (
          0.50 * np.sin(2 * np.pi * f0 * t)   # fundamental
        + 0.35 * np.sin(2 * np.pi * f1 * t)   # first formant
        + 0.25 * np.sin(2 * np.pi * f2 * t)   # second formant
        + 0.05 * np.random.randn(len(t))       # noise floor
    )

    # np.max(np.abs(signal)) finds the single largest absolute value in the array.
    # Dividing by that value scales the array so its peak is exactly 1.0.
    # Multiplying by 0.9 then brings it just below full scale, leaving headroom.
    # This prevents "clipping" (when a value exceeds the allowed range of -1 to 1).
    signal = signal / np.max(np.abs(signal)) * 0.9

    # .astype(np.float32) casts the array to 32-bit float for consistency.
    return signal.astype(np.float32), sr


def generate_test_tone(frequency=1000.0, duration=1.0, sr=16000):
    # Creates a single, clean sine wave at one specific frequency.
    # This is the simplest possible test signal -- useful for verifying
    # that the filter bank puts energy in the right electrode channel.
    # For example, a 1000 Hz tone should show up strongly in the electrode
    # whose centre frequency is closest to 1000 Hz.

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 0.8 is the amplitude -- keeps the signal at 80% of full scale.
    signal = 0.8 * np.sin(2 * np.pi * frequency * t)

    return signal.astype(np.float32), sr


def pre_emphasis(signal, coeff=0.97):
    # This function applies a simple high-pass filter called pre-emphasis.
    #
    # The problem it solves:
    #   Natural speech has an average spectral slope of about -6 dB/octave.
    #   That means for every doubling of frequency, the energy drops by a factor
    #   of 4. So by the time the signal reaches 8000 Hz, its energy is about
    #   1000x weaker than at 250 Hz. This imbalance makes it hard to design a
    #   filter bank where every channel has a meaningful signal.
    #
    # The formula: y[n] = x[n] - coeff * x[n-1]
    #   Each output sample is the current input sample MINUS 0.97 times the
    #   previous input sample. This amplifies differences between successive
    #   samples, which is what high-frequency components are (they oscillate fast,
    #   so consecutive samples differ a lot). Low-frequency components barely
    #   change between samples, so their difference is small.
    #
    # Why 0.97?
    #   The closer coeff is to 1.0, the stronger the boost. 0.97 is the
    #   standard value used in speech processing (e.g. in MFCC extraction).
    #   It partially compensates for the -6 dB/octave slope without over-boosting.
    #
    # Syntax note:
    #   signal[1:]  is a slice -- it takes every element from index 1 onward
    #               (skipping the first element at index 0).
    #   signal[:-1] takes every element except the last one.
    #   Both slices have the same length, so the subtraction is element-wise.
    #
    #   np.append(a, b) glues array b onto the end of array a.
    #   Here it prepends signal[0] at the front, because the formula y[0] = x[0]
    #   (there is no x[-1] sample before the start of the recording).

    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(signal, sr, frame_ms=16.0, hop_ms=8.0):
    # Cuts the signal into short overlapping windows called frames.
    #
    # Key parameters:
    #   frame_ms = 16.0  --> each frame is 16 ms long = 256 samples at 16 kHz
    #   hop_ms   = 8.0   --> the next frame starts 8 ms later = 128 samples
    #   50% overlap (frame_ms / hop_ms = 2) is the standard in speech processing
    #
    # Why overlap?
    #   If frames did not overlap, any speech transition that happened to fall
    #   exactly at a frame boundary would be split across two frames, and neither
    #   frame would capture it cleanly. With 50% overlap, every part of the
    #   signal is always near the middle of at least one frame.
    #
    # Why the Hann window?
    #   When an FFT analyses a frame, it implicitly assumes the signal repeats
    #   periodically. If the first and last samples of the frame have different
    #   values, the forced discontinuity at the join creates artificial
    #   high-frequency content (called "spectral leakage"). The Hann window
    #   tapers both ends of the frame smoothly to zero, so no discontinuity exists.

    # Convert milliseconds to sample counts by multiplying by samples-per-millisecond.
    # int() truncates the result to a whole number (samples must be integers).
    frame_len = int(sr * frame_ms / 1000.0)   # e.g. 16000 * 16 / 1000 = 256
    hop_len   = int(sr * hop_ms  / 1000.0)    # e.g. 16000 *  8 / 1000 = 128

    # Padding: if the signal length is not a perfect multiple of hop_len,
    # the last frame would be cut short. np.pad adds extra samples to the end
    # using "reflect" mode, which mirrors the last few samples. This is better
    # than zero-padding because it does not create an artificial silence boundary.
    pad = frame_len - (len(signal) % hop_len) if len(signal) % hop_len != 0 else 0
    signal = np.pad(signal, (0, pad), mode="reflect")
    # (0, pad) means "add 0 samples at the start, and 'pad' samples at the end"

    # Frame extraction using index arithmetic (stride trick):
    # num_frames is how many complete frames fit in the padded signal.
    num_frames = 1 + (len(signal) - frame_len) // hop_len
    # "//" is integer (floor) division -- it rounds down to the nearest integer.

    # np.arange(frame_len) creates [0, 1, 2, ..., frame_len-1].
    # [None, :] reshapes it to shape (1, frame_len) -- adding a new first axis.
    # np.arange(num_frames)[:, None] creates a column vector of frame start indices.
    # Adding them together broadcasts: each row of the result is the sample
    # indices for one frame. This builds the index matrix in one line without a loop.
    indices = np.arange(frame_len)[None, :] + np.arange(num_frames)[:, None] * hop_len

    # Fancy indexing: signal[indices] pulls out the samples specified by the
    # 2D index matrix. The result is a 2D array of shape (num_frames, frame_len),
    # where each row is one frame.
    frames = signal[indices]

    # np.hanning(frame_len) returns the Hann window as an array of frame_len values.
    # Multiplying frames * np.hanning(...) applies the window to every frame at once
    # because numpy broadcasting stretches the 1D window across all rows.
    frames = frames * np.hanning(frame_len)

    # Returns three things:
    #   frames    -- 2D array (num_frames, frame_len)
    #   frame_len -- integer, number of samples per frame (256)
    #   hop_len   -- integer, step between frames (128), needed by later stages
    return frames, frame_len, hop_len
