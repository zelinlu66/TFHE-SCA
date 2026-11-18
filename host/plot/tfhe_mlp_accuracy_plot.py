"""Train the multi-trace MLP on random-ciphertext TFHE captures."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    output_dir = Path("./data/tfhe_random_ciphertext_multi_trace_results_x4")

    with np.load(output_dir / "accuracy_vs_trace_count.npz") as data:
        trace_counts = data["trace_counts"]
        bit_accuracy = data["bit_accuracy"]
        full_recovery = data["full_recovery"]

        # Configure each font size separately.
        xlabel_fontsize = 20
        ylabel_fontsize = 18
        xtick_fontsize = 18
        ytick_fontsize = 18
        legend_fontsize = 16

        fig, ax = plt.subplots(figsize=(6.5, 4.4))

        ax.plot(
            trace_counts, bit_accuracy, "o-", color="tab:blue",
            label="Coefficient accuracy"
        )
        ax.plot(
            trace_counts, full_recovery, "s-", color="tab:orange",
            label="Full-key recovery rate"
        )

        ax.set_xlabel("Number of traces per key", fontsize=xlabel_fontsize)
        ax.set_ylabel("Accuracy / recovery rate (%)", fontsize=ylabel_fontsize)
        ax.tick_params(axis="x", labelsize=xtick_fontsize)
        ax.tick_params(axis="y", labelsize=ytick_fontsize)

        ax.set_xticks(trace_counts)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=legend_fontsize)

        fig.tight_layout()
        figure_path = output_dir / "accuracy_vs_trace_count.pdf"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(bit_accuracy)
        print(full_recovery)