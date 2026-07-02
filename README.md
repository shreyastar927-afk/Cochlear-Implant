# Cochlear-Implant
A signal-processing project focused on reducing the number of stimulation channels in cochlear implants by analyzing and optimizing frequency band allocation, aiming to simplify processing while preserving essential auditory information.

Cochlear Implant Signal Processing Simulation:
A fully-modular Python implementation of a cochlear implant (CI) speech encoding pipeline with virtual channel current steering, based on the project brief "Speech-to-Electrode Encoding Using Virtual Channels for Cochlear Implants".

Project Overview
Standard cochlear implants have 12–22 physical electrodes. Current steering (virtual channels) creates additional pitch percepts by simultaneously stimulating two adjacent electrodes at complementary current ratios. This simulator implements the full pipeline from raw audio → biphasic pulse schedule.

Pipeline Architecture:
Raw Audio
   │
   ▼
┌─────────────────────────────────┐
│  Stage 1 · preprocessing.py     │  Pre-emphasis + framing
└──────────────┬──────────────────┘
               │
   ┌───────────┴────────────┐
   │                        │
   ▼                        ▼
┌──────────────┐    ┌──────────────────┐
│ filterbank   │    │  filterbank      │  IIR filter bank (full signal)
│ apply_filter │    │  fft_filterbank  │  FFT energy per frame
└──────┬───────┘    └────────┬─────────┘
       │                     │
       ▼                     ▼
┌─────────────────────────────────┐
│  Stage 3 · envelope.py          │  Rectify → LPF → n-of-m → compress
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 4 · current_steering.py  │  Virtual channel α mapping
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 5 · pulse_train.py       │  Biphasic CIS pulses + vocoder
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 6 · visualization.py     │  8 analysis plots + dashboard
└─────────────────────────────────┘
## Modules

| Module | Stage | Responsibility |
|---|---|---|
| `preprocessing.py` | 1 | Load audio, pre-emphasis filter, overlapping frames |
| `filterbank.py` | 2 | ERB-spaced Butterworth filter bank, FFT energy bands |
| `envelope.py` | 3 | Envelope extraction, ACE n-of-m selection, loudness compression |
| `current_steering.py` | 4 | Virtual channel map, α-weighted current splitting, pitch estimation |
| `pulse_train.py` | 5 | CIS biphasic pulse train generation, vocoder audio synthesis |
| `visualization.py` | 6 | 8 plot types: waveform, electrodogram, filter bank, pitch, dashboard |
| `main.py` | — | CLI orchestrator, chains all stages |

---

## Installation

```bash
pip install -r requirements.txt
```

Requires **Python ≥ 3.10**

---

## Usage

### Quick demo (synthetic speech signal)

```bash
python main.py
```

### Use a real WAV file

```bash
python main.py --audio path/to/speech.wav
```

### Pure tone test (1 kHz)

```bash
python main.py --tone 1000 --duration 0.5
```

### Full parameter control

```bash
python main.py \
  --audio speech.wav \
  --electrodes 22 \
  --virtual-steps 5 \
  --n-active 8 \
  --f-low 250 \
  --f-high 8000 \
  --carrier sine \
  --save-dir output_figures \
  --save-audio \
  --no-show
```

---

## All CLI Options

| Option | Description |
|---|---|
| `--audio FILE` | Real audio file (WAV/FLAC/OGG) |
| `--tone HZ` | Generate a pure tone at this frequency |
| `--speech` | Generate a synthetic speech-like signal (default) |
| `--sr INT` | Sample rate [default: 16000] |
| `--duration FLOAT` | Test signal duration in seconds [default: 1.0] |
| `--electrodes INT` | Number of physical electrodes [default: 22] |
| `--f-low FLOAT` | Lower frequency bound Hz [default: 250] |
| `--f-high FLOAT` | Upper frequency bound Hz [default: 8000] |
| `--virtual-steps INT` | Intermediate virtual channels per pair [default: 5] |
| `--n-active INT` | ACE n-of-m: active channels per frame [default: 8] |
| `--frame-ms FLOAT` | Analysis frame size ms [default: 16] |
| `--hop-ms FLOAT` | Frame hop size ms [default: 8] |
| `--carrier {sine,noise}` | Vocoder carrier type [default: sine] |
| `--save-dir DIR` | Save all figures to this directory |
| `--save-audio` | Write vocoder_output.wav |
| `--no-show` | Suppress interactive display |

---

## Output Figures

| # | Filename | Description |
|---|---|---|
| 1 | `01_input_signal.png` | Waveform + spectrogram |
| 2 | `02_filterbank_responses.png` | ERB filter frequency responses |
| 3 | `03_channel_envelopes.png` | Temporal envelopes per channel |
| 4 | `04_electrodogram.png` | Stimulation pattern (time × electrode) |
| 5 | `05_virtual_channels.png` | Physical vs. virtual channel positions |
| 6 | `06_pitch_trajectory.png` | Estimated virtual pitch over time |
| 7 | `07_vocoder_comparison.png` | Original vs. CI vocoder output |
| 8 | `08_pipeline_dashboard.png` | All-in-one summary dashboard |

---

## Key Concepts

### Virtual Channels (Current Steering)

With electrodes **k** and **k+1**, steering coefficient **α ∈ [0, 1]** allocates:

- Current to electrode **k**:

```text
I_k = (1 − α) · A
```

- Current to electrode **k+1**:

```text
I_(k+1) = α · A
```

**22 physical electrodes × 5 virtual steps → ~105 virtual channels → ~5× spectral resolution improvement.**

---

### ACE n-of-m Strategy

Per stimulation cycle, only the **n channels** (default: 8) with the highest spectral energy are activated from the total of **m (22)**. This focuses power on speech formants and mirrors Cochlear Ltd.'s clinical ACE strategy.

---

### Dynamic Range Compression

Acoustic envelopes span **~80 dB**; cochlear implant electrodes span only **~30 dB** between threshold (**T-level**) and comfort level (**C-level**). The loudness growth function maps this non-linearly.

---

## References

1. Wilson, B. S., et al. (1991). *Better speech recognition with cochlear implants.* Nature, 352, 236–238.  
2. Loizou, P. C. (1998). *Mimicking the human ear.* IEEE Signal Processing Magazine, 15(5), 101–130.  
3. Moore, B. C. J., & Glasberg, B. R. (1983). *Auditory-filter bandwidths and excitation patterns.* J. Acoust. Soc. Am., 74(3), 750–753.  
4. Donaldson, G. S., et al. (2005). *Improving speech understanding with current steering.* Ear and Hearing, 26(4S), 109S–119S.  
5. Koch, D. B., et al. (2007). *Combining current steering and current focusing.* Cochlear Implants International, 8(4), 202–213.  
6. Shannon, R. V., et al. (2004). *Speech recognition with primarily temporal cues.* Science, 303, 1780–1782.  
7. Vandali, A. E., et al. (2000). *Speech perception as a function of electrical stimulation rate.* Ear and Hearing, 21(6), 608–624.  
8. Zeng, F.-G., et al. (2002). *Cochlear implant speech recognition with civil-level stimulation.* J. Acoust. Soc. Am., 112(5), 1829–1842.
