"""Compare two TFHE datasets: raw mean differences above, Welch t below."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Edit both paths and legend names. Segmentation can differ between files.
datasets = [
    dict(
        path=Path("./data/tfhe_all_qminus1_training_data.npz"),
        name="Chosen ciphertext", color="tab:blue",
        start_index=25, segment_spacing=11, segment_length=11,
    ),
    dict(
        path=Path("./data/tfhe_random_ciphertext_training_data.npz"),
        name="Random ciphertext", color="tab:orange",
        start_index=25, segment_spacing=11, segment_length=11,
    ),
]
output_dir = Path("./data/tvla_comparison")
num_segments = 500
max_segments_per_class = 10000  # Preserve the original per-class cap.
threshold = 4.5

# Separate font-size controls.
xlabel_fontsize = 20
mean_ylabel_fontsize = 20
tvla_ylabel_fontsize = 20
xtick_fontsize = 18
ytick_fontsize = 18
mean_legend_fontsize = 13
tvla_legend_fontsize = 14


def main():
    if len(datasets) != 2:
        raise ValueError("Configure exactly two input datasets")
    if num_segments < 1 or max_segments_per_class < 2:
        raise ValueError("Invalid segment count or per-class cap")
    for config in datasets:
        if not config["path"].is_file():
            raise FileNotFoundError(f"Edit the input path: {config['path']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    plot_data = dict(
        threshold=np.array(threshold),
        difference_definition=np.array("mean0 - mean1 (untransformed samples)"),
    )

    # Process each file independently: never pool samples across files.
    for dataset_index, config in enumerate(datasets, start=1):
        input_path = config["path"]
        start_index = config["start_index"]
        segment_spacing = config["segment_spacing"]
        segment_length = config["segment_length"]
        if start_index < 0 or segment_spacing < 1 or segment_length < 1:
            raise ValueError(f"{input_path}: invalid segmentation settings")
        if segment_length > segment_spacing:
            raise ValueError("Use nonoverlapping segments: length <= spacing")

        # Accept traces [T,S] or grouped traces [keys,repeats,S].
        with np.load(input_path, allow_pickle=False) as data:
            traces = np.asarray(data["traces"], dtype=np.float64)
            key = np.asarray(data["key"])
        if traces.ndim == 3:
            num_keys, num_repeats, num_samples = traces.shape
            if key.shape != (num_keys, num_segments):
                raise ValueError(
                    f"Expected key shape ({num_keys}, {num_segments}), got {key.shape}"
                )
            traces = traces.reshape(num_keys * num_repeats, num_samples)
            key = np.repeat(key, num_repeats, axis=0)
        if traces.ndim != 2 or traces.shape[0] == 0:
            raise ValueError(f"{input_path}: expected nonempty traces [T,S]")
        if not np.isfinite(traces).all():
            raise ValueError(f"{input_path}: waveforms contain NaN or infinity")
        num_traces, num_samples = traces.shape
        if key.shape == (num_segments,):
            labels = np.broadcast_to(key, (num_traces, num_segments))
        elif key.shape == (num_traces, num_segments):
            labels = key
        else:
            raise ValueError(
                f"{input_path}: expected key [{num_segments}] or "
                f"[{num_traces},{num_segments}], got {key.shape}"
            )
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Secret coefficients must be 0 or 1")

        starts = start_index + np.arange(num_segments) * segment_spacing
        indices = starts[:, None] + np.arange(segment_length)[None, :]
        if indices[-1, -1] >= num_samples:
            raise ValueError(
                f"{input_path}: need {indices[-1, -1] + 1} samples; "
                f"only {num_samples} available"
            )
        segments = traces[:, indices]

        # Keep provenance aligned with the capped class arrays.
        trace0, coefficient0 = np.nonzero(labels == 0)
        trace1, coefficient1 = np.nonzero(labels == 1)
        trace0, coefficient0 = trace0[:max_segments_per_class], coefficient0[:max_segments_per_class]
        trace1, coefficient1 = trace1[:max_segments_per_class], coefficient1[:max_segments_per_class]
        class0 = segments[trace0, coefficient0]
        class1 = segments[trace1, coefficient1]
        n0, n1 = len(class0), len(class1)
        if min(n0, n1) < 2:
            raise ValueError(f"{input_path}: each class needs at least two segments")

        # Both the mean difference and Welch t use original samples.
        mean0, mean1 = class0.mean(axis=0), class1.mean(axis=0)
        var0, var1 = class0.var(axis=0, ddof=1), class1.var(axis=0, ddof=1)
        difference = mean0 - mean1
        standard_error = np.sqrt(var0 / n0 + var1 / n1)
        t_values = np.zeros(segment_length, dtype=np.float64)
        np.divide(difference, standard_error, out=t_values, where=standard_error > 0)
        degenerate = (standard_error == 0) & (difference != 0)
        t_values[degenerate] = np.sign(difference[degenerate]) * np.inf
        significant = np.abs(t_values) > threshold

        x = np.arange(segment_length)
        result = dict(
            x=x, mean0=mean0, mean1=mean1,
            mean_difference=difference,
            t_values=t_values, significant=significant,
            n0=np.array(n0), n1=np.array(n1),
            start_index=np.array(start_index),
            segment_spacing=np.array(segment_spacing),
            segment_length=np.array(segment_length),
            source_path=np.array(str(input_path)),
            name=np.array(config["name"]),
        )
        results.append(result)
        # Prefixes allow different segment lengths without object arrays/pickle.
        plot_data.update({f"file{dataset_index}_{key}": value for key, value in result.items()})
        np.savez_compressed(
            output_dir / f"file{dataset_index}_segment_tvla.npz",
            **result, segments=segments, labels=labels, class0=class0, class1=class1,
            class0_trace_index=trace0, class1_trace_index=trace1,
            class0_coefficient_index=coefficient0,
            class1_coefficient_index=coefficient1,
            threshold=threshold,
        )
        print(f"{config['name']}: class 0 = {n0}, class 1 = {n1}, max |t| = {np.max(np.abs(t_values)):.3f}")
        print(f"Samples exceeding +/-{threshold}: {np.flatnonzero(significant).tolist()}")

    # Two raw mean-difference curves above; two independent t curves below.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for config, result in zip(datasets, results):
        axes[0].plot(
            result["x"], result["mean_difference"],
            color=config["color"], label=config["name"],
        )
        axes[1].plot(result["x"], result["t_values"],
                     color=config["color"], label=config["name"])

    axes[0].axhline(0, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Mean difference (0 - 1)\n(capture units)", fontsize=mean_ylabel_fontsize)
    axes[0].legend(fontsize=mean_legend_fontsize, ncol=2)
    axes[1].axhline(threshold, color="red", linestyle="--", label=rf"$\pm {threshold}$")
    axes[1].axhline(-threshold, color="red", linestyle="--")
    axes[1].set_ylabel("Welch t-statistic", fontsize=tvla_ylabel_fontsize)
    axes[1].set_xlabel("Sample within coefficient segment", fontsize=xlabel_fontsize)
    axes[1].legend(fontsize=tvla_legend_fontsize)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=xtick_fontsize)
        ax.tick_params(axis="y", labelsize=ytick_fontsize)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.align_ylabels(axes)

    # Compact data for quickly replotting, plus PDF/PNG exports.
    np.savez_compressed(output_dir / "tvla_comparison_plot_data.npz", **plot_data)
    fig.savefig(output_dir / "tvla_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "tvla_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved results to {output_dir}")
    # Original caveat: pooled windows within a trace can be correlated.
    print("Caution: pooled segments share traces; position/history may confound classes.")
    plt.show()


if __name__ == "__main__":
    main()
