"""Segment TFHE traces by loop iteration and compare secret-bit classes."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    # 1. Edit these settings. All indices refer to SAVED waveform samples,
    # after any acquisition downsampling, not target CPU cycles.
    input_path = Path("./data/tfhe_all_qminus1_training_data.npz")
    output_path = Path("./data/tfhe_segment_tvla_training_data.npz")
    figure_path = Path("./data/tfhe_segment_tvla_training_data.pdf")
    num_segments = 500
    start_index = 25       # First sample of the first coefficient operation: EDIT.
    segment_spacing = 11 # Samples between successive operations: EDIT.
    segment_length = 11  # Samples retained from each operation: EDIT.
    threshold = 4.5

    # if output_path.exists() or figure_path.exists():
    #     raise FileExistsError("Choose new output filenames to preserve previous results")
    if start_index < 0 or segment_spacing < 1 or segment_length < 1:
        raise ValueError("Invalid segmentation settings")
    if segment_length > segment_spacing:
        raise ValueError("Use nonoverlapping segments: length <= spacing")

    # 2. Load captured waveforms and ground-truth binary secret coefficients.
    # Supports one shared key [500] or a separate key per trace [T,500].
    with np.load(input_path, allow_pickle=False) as data:
        traces = np.asarray(data["traces"], dtype=np.float64)
        key = np.asarray(data["key"])
    if traces.ndim != 2 or traces.shape[0] == 0:
        raise ValueError("Expected nonempty traces with shape [num_traces, num_samples]")
    if not np.isfinite(traces).all():
        raise ValueError("Waveforms contain NaN or infinity")
    num_traces, num_samples = traces.shape
    if key.shape == (num_segments,):
        labels = np.broadcast_to(key, (num_traces, num_segments))
    elif key.shape == (num_traces, num_segments):
        labels = key
    else:
        raise ValueError(f"Expected key shape ({num_segments},) or ({num_traces}, {num_segments}); got {key.shape}")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Secret coefficients must be 0 or 1")

    # 3. Slice exactly 500 windows per trace, in coefficient order.
    # segments[t,i,:] corresponds to secret coefficient key[i] in trace t.
    starts = start_index + np.arange(num_segments) * segment_spacing
    indices = starts[:, None] + np.arange(segment_length)[None, :]
    if indices[-1, -1] >= num_samples:
        raise ValueError(f"Need {indices[-1,-1]+1} samples, only {num_samples} available; adjust settings")
    segments = traces[:, indices]

    # 4. Pool all coefficient windows according to their known secret bit.
    class0 = segments[labels == 0][:10000]
    class1 = segments[labels == 1][:10000]
    n0, n1 = len(class0), len(class1)
    if min(n0, n1) < 2:
        raise ValueError("Each class needs at least two segments")
    print(f"Segments: {segments.shape}; class 0: {n0}; class 1: {n1}")

    # 5. Samplewise Welch t-statistic, using unbiased sample variances.
    # Positive t means class 0 has the larger mean waveform value.
    mean0, mean1 = class0.mean(axis=0), class1.mean(axis=0)
    var0, var1 = class0.var(axis=0, ddof=1), class1.var(axis=0, ddof=1)
    difference = mean0 - mean1
    standard_error = np.sqrt(var0 / n0 + var1 / n1)
    t_values = np.zeros(segment_length, dtype=np.float64)
    np.divide(difference, standard_error, out=t_values, where=standard_error > 0)
    # Zero variance with unequal means has an infinite statistic; equal means
    # and zero variance are assigned zero as an explicit plotting convention.
    degenerate = (standard_error == 0) & (difference != 0)
    t_values[degenerate] = np.sign(difference[degenerate]) * np.inf
    significant = np.abs(t_values) > threshold
    print(f"max |t| = {np.max(np.abs(t_values)):.3f}")
    print(f"Samples exceeding +/-{threshold}: {np.flatnonzero(significant).tolist()}")

    # Repeated windows in one acquisition may be correlated. This pooled
    # statistic is exploratory; threshold crossings alone do not establish
    # independent-trace TVLA significance. Fixed coefficient position and
    # accumulator history can also confound the two classes for a shared key.
    print("Caution: pooled segments share traces; position/history may confound classes.")

    # 6. Plot the mean segments and class-conditioned t-statistic.
    x = np.arange(segment_length)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(x, mean0, label=f"secret=0 (n={n0})")
    axes[0].plot(x, mean1, label=f"secret=1 (n={n1})")
    axes[0].set_ylabel("Mean power", fontsize=20)
    axes[0].legend(fontsize=18)

    axes[1].plot(x, t_values, color="black",
                 label="Welch t: class 0 - class 1")
    axes[1].axhline(threshold, color="red", linestyle="--")
    axes[1].axhline(-threshold, color="red", linestyle="--")
    axes[1].set_ylabel("t-statistic", fontsize=20)
    axes[1].set_xlabel("Sample within coefficient segment", fontsize=20)
    axes[1].legend(fontsize=18)

    # Font size of the numbers along both axes.
    for ax in axes:
        ax.tick_params(axis="both", labelsize=18)

    # fig.suptitle("TFHE secret-conditioned segment comparison", fontsize=22)
    fig.tight_layout()
    fig.align_ylabels(axes)

    # 7. Save both classes, labels, source trace/coefficient indices and results.
    # Provenance allows later analysis without losing which trace each window
    # came from. No alignment, filtering or normalization is applied here.
    trace_index = np.broadcast_to(np.arange(num_traces)[:, None], labels.shape)
    coefficient_index = np.broadcast_to(np.arange(num_segments)[None, :], labels.shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    # with output_path.open("xb") as output:
    # output = Path("./data/mlp_watermark_reference_traces.npz")
    np.savez_compressed(
        output_path, segments=segments, labels=labels, class0=class0, class1=class1,
        class0_trace_index=trace_index[labels == 0],
        class1_trace_index=trace_index[labels == 1],
        class0_coefficient_index=coefficient_index[labels == 0],
        class1_coefficient_index=coefficient_index[labels == 1],
        mean0=mean0, mean1=mean1, t_values=t_values, significant=significant,
        start_index=start_index, segment_spacing=segment_spacing,
        segment_length=segment_length, threshold=threshold,
        source_path=np.array(str(input_path)),
    )
    fig.savefig(figure_path, dpi=200)
    print(f"Saved {output_path} and {figure_path}")
    plt.show()


if __name__ == "__main__":
    main()
