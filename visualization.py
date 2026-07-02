# visualization.py  --  Stage 6 of the CI pipeline
#
# All the plots for understanding and presenting what the pipeline does.
# Each function takes the data from one or more stages and returns a
# Matplotlib Figure that can be displayed or saved to disk.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import sosfreqz

# Clean, minimal look for all plots
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 110,
    "axes.labelsize"   : 10,
    "axes.titlesize"   : 11,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
})

# Colour choices -- keeping it consistent across plots
C_BLUE    = "#4A90D9"
C_ORANGE  = "#E87040"
C_GREEN   = "#3DAA6B"
CMAP_ELEC = "plasma"


def plot_input_signal(signal, sr):
    # Shows what the raw input audio looks like in two ways:
    # the waveform (amplitude over time) and the spectrogram
    # (which frequencies are present at each moment).
    t   = np.arange(len(signal)) / sr
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), tight_layout=True)
    fig.suptitle("Input Audio Signal", fontweight="bold")

    axes[0].plot(t, signal, color=C_BLUE, linewidth=0.6)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("Waveform")
    axes[0].set_xlim(t[0], t[-1])

    axes[1].specgram(signal, NFFT=512, Fs=sr, noverlap=256, cmap="inferno", scale="dB")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Spectrogram")
    axes[1].set_ylim(0, 8000)

    return fig


def plot_filterbank_responses(sos_filters, centre_freqs, sr):
    # Shows the frequency response of every band-pass filter in the bank.
    # Each coloured line is one electrode channel -- lower electrodes
    # are blue/green, higher electrodes are yellow/green.
    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    fig.suptitle("ERB-Spaced Filter Bank Frequency Responses", fontweight="bold")

    n_bands = len(sos_filters)
    cmap    = plt.get_cmap("viridis", n_bands)

    for k, sos in enumerate(sos_filters):
        w, h = sosfreqz(sos, worN=4096, fs=sr)
        dB   = 20 * np.log10(np.abs(h) + 1e-12)
        ax.plot(w, dB, color=cmap(k), linewidth=0.9, alpha=0.85)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_ylim(-60, 5)
    ax.set_xlim(0, sr / 2)
    ax.axhline(-3, color="grey", linestyle="--", linewidth=0.7, label="-3 dB cutoff")
    ax.legend(fontsize=8)

    sm = ScalarMappable(cmap="viridis", norm=Normalize(1, n_bands))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Electrode #", rotation=270, labelpad=12)

    return fig


def plot_envelopes(envelopes, centre_freqs, sr, max_channels=8):
    # Shows the extracted envelope for a selection of channels.
    # Each row is one electrode channel; the shaded area shows how loud
    # that channel is at every point in time.
    M, N     = envelopes.shape
    channels = np.linspace(0, M - 1, min(max_channels, M), dtype=int)
    t        = np.arange(N) / sr

    fig, axes = plt.subplots(
        len(channels), 1,
        figsize=(10, 1.4 * len(channels)),
        tight_layout=True, sharex=True
    )
    fig.suptitle("Temporal Envelopes per Electrode Channel", fontweight="bold")

    if len(channels) == 1:
        axes = [axes]

    cmap = plt.get_cmap("plasma", M)
    for i, ch in enumerate(channels):
        axes[i].fill_between(t, envelopes[ch], alpha=0.75, color=cmap(ch))
        axes[i].set_ylabel(f"{centre_freqs[ch]:.0f} Hz", fontsize=8, rotation=0, labelpad=42)
        axes[i].set_ylim(0, None)
        axes[i].set_yticks([])

    axes[-1].set_xlabel("Time (s)")
    return fig


def plot_electrodogram(electrode_currents, centre_freqs, hop_len, sr, max_frames=200):
    # The electrodogram is the main output of the pipeline.
    # Time runs along the horizontal axis, electrodes along the vertical axis.
    # Bright colours = high current; dark/black = no stimulation.
    # This shows which electrodes fire when, and how strongly.
    T, M    = electrode_currents.shape
    T_plot  = min(T, max_frames)
    data    = electrode_currents[:T_plot].T            # shape: (M, T_plot)
    t_ms    = np.arange(T_plot) * hop_len / sr * 1000  # convert to milliseconds

    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    fig.suptitle("Electrodogram -- Stimulation Pattern", fontweight="bold")

    im = ax.imshow(
        data, aspect="auto", origin="lower", cmap=CMAP_ELEC,
        extent=[t_ms[0], t_ms[-1], 0, M],
        vmin=0, vmax=1, interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalised Current Amplitude", rotation=270, labelpad=14)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Electrode #")

    tick_step = max(1, M // 10)
    ax.set_yticks(np.arange(0, M, tick_step) + 0.5)
    ax.set_yticklabels(np.arange(1, M + 1, tick_step))

    return fig


def plot_virtual_channel_map(vc_map, n_electrodes, centre_freqs):
    # Visualises where the virtual channels sit relative to the physical ones.
    # Physical electrodes (diamonds) are fixed positions on the array.
    # Virtual channels (small circles) fill the gaps between them.
    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    fig.suptitle("Virtual Channel Positions via Current Steering", fontweight="bold")

    # Physical electrodes on the bottom row
    ax.scatter(
        centre_freqs, np.zeros(n_electrodes),
        s=100, zorder=5, color=C_ORANGE,
        label="Physical electrodes", marker="D",
    )

    # Virtual channels on the row above
    vc_freqs = [
        (1.0 - vc["alpha"]) * centre_freqs[vc["elec_low"]]
        + vc["alpha"]        * centre_freqs[vc["elec_high"]]
        for vc in vc_map
    ]
    ax.scatter(
        vc_freqs, np.ones(len(vc_freqs)) * 0.5,
        s=20, alpha=0.6, color=C_BLUE,
        label=f"Virtual channels ({len(vc_freqs)} total)",
    )

    # Faint lines connecting physical electrodes to their virtual channels
    for vc in vc_map:
        f_lo = centre_freqs[vc["elec_low"]]
        f_hi = centre_freqs[vc["elec_high"]]
        f_vc = (1.0 - vc["alpha"]) * f_lo + vc["alpha"] * f_hi
        ax.plot([f_lo, f_vc], [0, 0.5], color="grey", alpha=0.1, linewidth=0.5)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_yticks([0, 0.5])
    ax.set_yticklabels(["Physical\nElectrodes", "Virtual\nChannels"])
    ax.set_xscale("log")
    ax.set_xlim(centre_freqs[0] * 0.8, centre_freqs[-1] * 1.2)
    ax.legend(fontsize=9)
    return fig


def plot_pitch_trajectory(pitch_hz, hop_len, sr, centre_freqs):
    # Shows the estimated pitch the listener would hear at each moment.
    # The faint horizontal lines mark the positions of the physical electrodes.
    # If the pitch line falls between two lines, a virtual channel is active.
    T    = len(pitch_hz)
    t_ms = np.arange(T) * hop_len / sr * 1000

    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    fig.suptitle("Estimated Virtual Pitch Trajectory", fontweight="bold")

    for cf in centre_freqs:
        ax.axhline(cf, color="lightgrey", linewidth=0.5, zorder=0)

    ax.plot(t_ms, pitch_hz, color=C_ORANGE, linewidth=1.2, label="Virtual pitch")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_yscale("log")
    ax.set_ylim(centre_freqs[0] * 0.8, centre_freqs[-1] * 1.2)
    ax.legend(fontsize=9)
    return fig


def plot_vocoder_comparison(vocoder_signal, original_signal, sr):
    # Side-by-side comparison of the original audio and the vocoder output.
    # The vocoder sounds "buzzy" and has limited pitch detail -- that is
    # intentional, since it is simulating what a CI user would hear.
    t_orig = np.arange(len(original_signal)) / sr
    t_voc  = np.arange(len(vocoder_signal))  / sr

    fig = plt.figure(figsize=(12, 7), tight_layout=True)
    fig.suptitle("Original vs. Vocoder-Simulated CI Output", fontweight="bold")

    gs   = gridspec.GridSpec(2, 2, figure=fig)
    ax_w1 = fig.add_subplot(gs[0, 0])
    ax_w2 = fig.add_subplot(gs[0, 1])
    ax_s1 = fig.add_subplot(gs[1, 0])
    ax_s2 = fig.add_subplot(gs[1, 1])

    ax_w1.plot(t_orig, original_signal, color=C_BLUE,   linewidth=0.6)
    ax_w1.set_title("Original Waveform")
    ax_w1.set_ylabel("Amplitude")
    ax_w1.set_xlabel("Time (s)")

    ax_w2.plot(t_voc, vocoder_signal,   color=C_ORANGE, linewidth=0.6)
    ax_w2.set_title("CI Vocoder Output")
    ax_w2.set_ylabel("Amplitude")
    ax_w2.set_xlabel("Time (s)")

    ax_s1.specgram(original_signal, NFFT=512, Fs=sr, noverlap=256, cmap="inferno", scale="dB")
    ax_s1.set_title("Original Spectrogram")
    ax_s1.set_ylabel("Freq (Hz)")
    ax_s1.set_xlabel("Time (s)")
    ax_s1.set_ylim(0, 8000)

    ax_s2.specgram(vocoder_signal,  NFFT=512, Fs=sr, noverlap=256, cmap="inferno", scale="dB")
    ax_s2.set_title("Vocoder Spectrogram")
    ax_s2.set_ylabel("Freq (Hz)")
    ax_s2.set_xlabel("Time (s)")
    ax_s2.set_ylim(0, 8000)

    return fig


def plot_dashboard(signal, envelopes, electrode_currents, pitch_hz,
                   centre_freqs, hop_len, sr):
    # A single summary figure showing the most important outputs of the pipeline
    # all in one place. Useful for a quick overview or a slide in a presentation.
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Speech-to-Electrode Encoding -- Pipeline Dashboard\n"
        "Virtual Channel Cochlear Implant Simulation",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    t_sig = np.arange(len(signal)) / sr

    # Waveform
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t_sig, signal, color=C_BLUE, linewidth=0.6)
    ax0.set_title("Input Waveform")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")

    # Spectrogram
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.specgram(signal, NFFT=512, Fs=sr, noverlap=256, cmap="inferno", scale="dB")
    ax1.set_title("Input Spectrogram")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Freq (Hz)")
    ax1.set_ylim(0, 8000)

    # Envelope heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    T_env = envelopes.shape[1]
    t_env = np.arange(T_env) / sr * 1000
    ax2.imshow(
        envelopes, aspect="auto", origin="lower", cmap="viridis",
        extent=[0, t_env[-1], 0, len(centre_freqs)],
    )
    ax2.set_title("Channel Envelopes")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Channel #")

    # Electrodogram
    ax3 = fig.add_subplot(gs[1, 1])
    T_curr = min(electrode_currents.shape[0], 200)
    t_curr = np.arange(T_curr) * hop_len / sr * 1000
    ax3.imshow(
        electrode_currents[:T_curr].T, aspect="auto", origin="lower",
        cmap=CMAP_ELEC, extent=[t_curr[0], t_curr[-1], 0, electrode_currents.shape[1]],
        vmin=0, vmax=1,
    )
    ax3.set_title("Electrodogram (Current Steering)")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Electrode #")

    # Pitch trajectory
    ax4 = fig.add_subplot(gs[2, 0])
    T_p  = len(pitch_hz)
    t_p  = np.arange(T_p) * hop_len / sr * 1000
    for cf in centre_freqs[::3]:
        ax4.axhline(cf, color="lightgrey", linewidth=0.4, zorder=0)
    ax4.plot(t_p, pitch_hz, color=C_ORANGE, linewidth=1.0)
    ax4.set_title("Virtual Pitch Trajectory")
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Freq (Hz)")
    ax4.set_yscale("log")
    ax4.set_ylim(centre_freqs[0] * 0.8, centre_freqs[-1] * 1.2)

    # Text summary box
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    summary = (
        f"Pipeline Summary\n"
        f"{'---'*9}\n"
        f"Sample rate:         {sr} Hz\n"
        f"Physical electrodes: {len(centre_freqs)}\n"
        f"Freq range:          {centre_freqs[0]:.0f}-{centre_freqs[-1]:.0f} Hz\n"
        f"Frames processed:    {T_p}\n"
        f"Pitch range:         {pitch_hz.min():.0f}-{pitch_hz.max():.0f} Hz\n"
    )
    ax5.text(
        0.05, 0.95, summary,
        transform=ax5.transAxes,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4FF", alpha=0.8),
    )

    return fig
