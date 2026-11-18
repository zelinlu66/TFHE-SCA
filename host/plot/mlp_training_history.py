"""Plot the loss and accuracy saved in classifier.history.npz."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# Put the script beside the history file, or enter its full path here.
history_path = Path("./data/tfhe_multiple_trace_results_x4_loss/classifier.history.npz")

xlabel_fontsize = 20
ylabel_fontsize = 18
xtick_fontsize = 16
ytick_fontsize = 16
legend_fontsize = 16


def main():
    with np.load(history_path, allow_pickle=False) as data:
        epoch = data["epoch"]
        fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for split, label, style in [
            ("train", "Training", "o-"),
            ("val", "Validation", "s--"),
        ]:
            ax_loss.plot(epoch, data[f"{split}_loss"], style, label=label)
            ax_acc.plot(epoch, data[f"{split}_acc"], style, label=label)

    ax_loss.set_ylabel("Loss", fontsize=ylabel_fontsize)
    ax_acc.set_ylabel("Coefficient accuracy (%)", fontsize=ylabel_fontsize)
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("Epoch", fontsize=xlabel_fontsize)
        ax.tick_params(axis="x", labelsize=xtick_fontsize)
        ax.tick_params(axis="y", labelsize=ytick_fontsize)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=legend_fontsize)

    fig.tight_layout()
    for filename in ("training_curves.png", "training_curves.pdf"):
        path = history_path.with_name(filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()


if __name__ == "__main__":
    main()
