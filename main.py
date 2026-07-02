# main.py  (Pipeline Orchestrator)
#
# This file is the entry point -- the first file Python runs when the
# project is executed. Its job is to:
#
#   1. Parse command-line arguments (so the user can control settings
#      without editing source code)
#   2. Call each pipeline stage in order, passing outputs to the next stage
#   3. Collect all results and hand them to the visualisation functions
#
# HOW TO RUN:
#   python main.py                         -- demo with synthetic speech
#   python main.py --audio speech.wav      -- use a real audio file
#   python main.py --tone 1000             -- test with a 1 kHz pure tone
#   python main.py --no-show --save-dir figs  -- save plots without displaying
#
# Stages run in this order:
#   Stage 1  preprocessing.py     load audio, pre-emphasis, frame
#   Stage 2  filterbank.py        split signal into frequency bands
#   Stage 3  envelope.py          extract loudness, pick top n channels, compress
#   Stage 4  current_steering.py  map channels to electrode currents via alpha
#   Stage 5  pulse_train.py       build biphasic pulse trains and vocoder audio
#   Stage 6  visualization.py     generate all eight plots

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import preprocessing
import filterbank
import envelope
import current_steering
import pulse_train
import visualization


def build_parser():
    # argparse is Python's standard library for handling command-line arguments.
    # It reads the flags the user types after 'python main.py' and turns them
    # into an object (args) whose attributes map to those flags.
    # For example: 'python main.py --electrodes 16' sets args.electrodes = 16.
    # If a flag is not provided, argparse uses the default= value instead.
    #
    # ArgumentParser creates the parser object.
    # description= is the text shown when the user runs: python main.py --help
    # formatter_class=ArgumentDefaultsHelpFormatter makes --help automatically
    # show the default value for every argument.
    p = argparse.ArgumentParser(
        description="Speech-to-Electrode CI Simulation (Virtual Channels)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # add_mutually_exclusive_group() creates a group of arguments where only
    # one can be provided at a time. Providing two would cause an error.
    # This enforces the rule: you can give an audio file, OR a tone, OR use
    # the default speech signal -- but not a combination.
    source = p.add_mutually_exclusive_group()
    source.add_argument("--audio",  metavar="FILE",
                        help="Path to a WAV/FLAC/OGG audio file.")
    source.add_argument("--tone",   type=float, metavar="HZ",
                        help="Generate a test pure tone at this frequency.")
    source.add_argument("--speech", action="store_true",
                        help="Generate a synthetic speech-like signal (default).")

    p.add_argument("--sr",       type=int,   default=16000, help="Sample rate in Hz.")
    p.add_argument("--duration", type=float, default=1.0,   help="Test signal duration in seconds.")

    p.add_argument("--electrodes",    type=int,   default=22,     help="Number of physical electrodes.")
    p.add_argument("--f-low",         type=float, default=250.0,  help="Lowest frequency in Hz.")
    p.add_argument("--f-high",        type=float, default=8000.0, help="Highest frequency in Hz.")
    p.add_argument("--virtual-steps", type=int,   default=5,      help="Virtual channel steps per electrode pair.")
    p.add_argument("--n-active",      type=int,   default=8,      help="Active channels per frame (n-of-m).")

    p.add_argument("--frame-ms", type=float, default=16.0, help="Frame length in milliseconds.")
    p.add_argument("--hop-ms",   type=float, default=8.0,  help="Frame hop size in milliseconds.")

    p.add_argument("--carrier",    choices=["sine", "noise"], default="sine",
                   help="Vocoder carrier type.")
    p.add_argument("--save-dir",   metavar="DIR", default=None,
                   help="Folder to save all figures (created automatically if missing).")
    p.add_argument("--save-audio", action="store_true",
                   help="Save vocoder output as vocoder_output.wav.")
    p.add_argument("--no-show",    action="store_true",
                   help="Skip interactive plot windows (useful when saving to disk).")

    return p


def run_pipeline(args):
    # Runs all five processing stages and returns their results.
    #
    # WHY RETURN A DICTIONARY?
    #   A dictionary (called 'artefacts' in the caller) acts as a shared
    #   data store for the whole pipeline. Each stage writes its outputs
    #   into the dictionary using descriptive key names like 'envelopes'
    #   or 'electrode_currents'. The visualisation step later reads from it
    #   by name. This is cleaner than passing 20 separate variables around.
    #
    # time.perf_counter() is a high-resolution timer. Calling it at the start
    # and end of each stage and subtracting gives the elapsed time in seconds.
    # This is useful to identify which stage is the computational bottleneck.

    total_start = time.perf_counter()

    print("\n[Stage 1] Pre-processing...")
    t0 = time.perf_counter()

    if args.audio:
        signal, sr = preprocessing.load_audio(args.audio, target_sr=args.sr)
        print(f"  Loaded '{args.audio}'  sr={sr} Hz  {len(signal)/sr:.2f}s")
    elif args.tone:
        signal, sr = preprocessing.generate_test_tone(args.tone, args.duration, args.sr)
        print(f"  Pure tone at {args.tone} Hz  duration={args.duration}s")
    else:
        signal, sr = preprocessing.generate_speech_like_signal(args.duration, args.sr)
        print(f"  Synthetic speech-like signal  duration={args.duration}s")

    # Keep the original before pre-emphasis so Stage 6 can compare them
    original_signal = signal.copy()

    signal = preprocessing.pre_emphasis(signal)
    frames, frame_len, hop_len = preprocessing.frame_signal(
        signal, sr, frame_ms=args.frame_ms, hop_ms=args.hop_ms
    )
    print(f"  {len(frames)} frames  frame_len={frame_len}  hop_len={hop_len}")
    print(f"  Done in {time.perf_counter()-t0:.3f}s")

    print("\n[Stage 2] Filter bank...")
    t0 = time.perf_counter()

    sos_filters, centre_freqs = filterbank.build_filterbank(
        sr, n_electrodes=args.electrodes,
        f_low=args.f_low, f_high=args.f_high,
    )
    # IIR filter on the full signal gives smooth envelopes in Stage 3
    channel_signals = filterbank.apply_filterbank(signal, sos_filters)
    # FFT filter on frames gives per-frame energy for n-of-m selection in Stage 3
    band_energies, _ = filterbank.fft_filterbank(
        frames, sr, n_electrodes=args.electrodes,
        f_low=args.f_low, f_high=args.f_high,
    )
    print(f"  {args.electrodes} channels  {centre_freqs[0]:.0f}-{centre_freqs[-1]:.0f} Hz")
    print(f"  Done in {time.perf_counter()-t0:.3f}s")

    print("\n[Stage 3] Envelope extraction and channel selection...")
    t0 = time.perf_counter()

    envelopes                        = envelope.extract_envelopes(channel_signals, sr)
    selected_mask, selected_energies = envelope.select_n_of_m(band_energies, args.n_active)
    compressed                       = envelope.compress(selected_energies)

    print(f"  n-of-m: {args.n_active} of {args.electrodes} ({selected_mask.mean()*100:.1f}% channels active)")
    print(f"  Done in {time.perf_counter()-t0:.3f}s")

    print("\n[Stage 4] Current steering (virtual channels)...")
    t0 = time.perf_counter()

    vc_map = current_steering.build_virtual_channel_map(
        args.electrodes, n_steps=args.virtual_steps
    )
    electrode_currents, stim_events = current_steering.compute_electrode_currents(
        compressed, centre_freqs, vc_map,
        n_electrodes=args.electrodes,
        selected_mask=selected_mask,
    )
    pitch_hz = current_steering.estimate_pitch(electrode_currents, centre_freqs)

    n_virtual = len(vc_map)
    print(f"  Physical electrodes: {args.electrodes}")
    print(f"  Virtual channels:    {n_virtual} (+{n_virtual - args.electrodes} intermediate)")
    print(f"  Done in {time.perf_counter()-t0:.3f}s")

    print("\n[Stage 5] Pulse trains and vocoder...")
    t0 = time.perf_counter()

    pulse_trains   = pulse_train.generate_pulse_trains(
        electrode_currents, sr, hop_len, args.electrodes
    )
    vocoder_signal = pulse_train.synthesise_vocoder(
        envelopes, centre_freqs, sr, carrier=args.carrier
    )
    print(f"  Pulse train shape: {pulse_trains.shape}")
    print(f"  Vocoder output: {len(vocoder_signal)/sr:.2f}s at {sr} Hz")
    print(f"  Done in {time.perf_counter()-t0:.3f}s")

    if args.save_audio:
        pulse_train.save_audio(vocoder_signal, sr, "vocoder_output.wav")
        print("  Saved vocoder_output.wav")

    print(f"\n  Total pipeline time: {time.perf_counter()-total_start:.2f}s")

    return {
        "signal"            : original_signal,
        "signal_preemph"    : signal,
        "frames"            : frames,
        "sr"                : sr,
        "hop_len"           : hop_len,
        "sos_filters"       : sos_filters,
        "centre_freqs"      : centre_freqs,
        "channel_signals"   : channel_signals,
        "band_energies"     : band_energies,
        "envelopes"         : envelopes,
        "selected_mask"     : selected_mask,
        "compressed"        : compressed,
        "vc_map"            : vc_map,
        "electrode_currents": electrode_currents,
        "stim_events"       : stim_events,
        "pitch_hz"          : pitch_hz,
        "pulse_trains"      : pulse_trains,
        "vocoder_signal"    : vocoder_signal,
    }


def run_visualisation(artefacts, args):
    # Generates all eight analysis plots and optionally saves them as images.
    #
    # 'a = artefacts' just creates a short alias so the code below can write
    # a["signal"] instead of artefacts["signal"]. Both refer to the same object.

    a = artefacts

    # os.makedirs creates the directory and any missing parent directories.
    # exist_ok=True means it does NOT raise an error if the directory already exists.
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    def save(fig, filename):
        # This is a nested (inner) function -- it is defined inside run_visualisation
        # and can access 'args.save_dir' from the enclosing scope (a closure).
        # Defining it here avoids repeating os.path.join and savefig eight times.
        #
        # fig.savefig(path, dpi=150, bbox_inches="tight"):
        #   dpi=150 sets the output resolution (dots per inch). Higher = sharper but bigger file.
        #   bbox_inches="tight" trims excess whitespace around the figure edges.
        if args.save_dir:
            path = os.path.join(args.save_dir, filename)
            # os.path.join combines a directory path and a filename safely,
            # using the correct separator for the operating system (\ on Windows, / on Linux).
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")

    print("\n[Stage 6] Generating plots...")

    fig1 = visualization.plot_input_signal(a["signal"], a["sr"])
    save(fig1, "01_input_signal.png")

    fig2 = visualization.plot_filterbank_responses(a["sos_filters"], a["centre_freqs"], a["sr"])
    save(fig2, "02_filterbank_responses.png")

    fig3 = visualization.plot_envelopes(a["envelopes"], a["centre_freqs"], a["sr"])
    save(fig3, "03_channel_envelopes.png")

    fig4 = visualization.plot_electrodogram(a["electrode_currents"], a["centre_freqs"], a["hop_len"], a["sr"])
    save(fig4, "04_electrodogram.png")

    fig5 = visualization.plot_virtual_channel_map(a["vc_map"], args.electrodes, a["centre_freqs"])
    save(fig5, "05_virtual_channels.png")

    fig6 = visualization.plot_pitch_trajectory(a["pitch_hz"], a["hop_len"], a["sr"], a["centre_freqs"])
    save(fig6, "06_pitch_trajectory.png")

    fig7 = visualization.plot_vocoder_comparison(a["vocoder_signal"], a["signal"], a["sr"])
    save(fig7, "07_vocoder_comparison.png")

    fig8 = visualization.plot_dashboard(
        a["signal"], a["envelopes"], a["electrode_currents"],
        a["pitch_hz"], a["centre_freqs"], a["hop_len"], a["sr"],
    )
    save(fig8, "08_pipeline_dashboard.png")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")

    print("  All plots done.")


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if not args.audio and not args.tone:
        args.speech = True

    print("=" * 60)
    print("  Speech-to-Electrode CI Simulation -- Virtual Channels")
    print("=" * 60)
    print(f"  Electrodes:     {args.electrodes}")
    print(f"  Virtual steps:  {args.virtual_steps} per electrode pair")
    print(f"  n-of-m active:  {args.n_active}")
    print(f"  Freq range:     {args.f_low:.0f}-{args.f_high:.0f} Hz")
    print(f"  Sample rate:    {args.sr} Hz")
    print(f"  Carrier:        {args.carrier}")

    artefacts = run_pipeline(args)
    run_visualisation(artefacts, args)


if __name__ == "__main__":
    main()