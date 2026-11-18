from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

input_path = Path(
    "./data/"
    "tvla_comparison/tvla_comparison_plot_data.npz"
)

# Configure font sizes separately.
xlabel_fontsize = 20
difference_ylabel_fontsize = 20
tvla_ylabel_fontsize = 20
xtick_fontsize = 18
ytick_fontsize = 18
difference_legend_fontsize = 16
tvla_legend_fontsize = 16

colors = ["tab:blue", "tab:orange"]
dataset_names = ["Our attack", "Baseline attack"]

fig, axes = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True)

with np.load(input_path, allow_pickle=False) as data:
    threshold = float(data["threshold"])

    for j, color in enumerate(colors):
        prefix = f"file{j + 1}_"
        name = (
            dataset_names[j]
            if dataset_names is not None
            else str(data[prefix + "name"].item())
        )
        x = data[prefix + "x"]

        # Raw mean difference: no square-root transformation.
        if prefix + "mean_difference" in data:
            difference = data[prefix + "mean_difference"]
        else:
            difference = data[prefix + "mean0"] - data[prefix + "mean1"]

        axes[0].plot(x, difference, color=color, label=name)
        axes[1].plot(
            x, data[prefix + "t_values"],
            color=color, label=name
        )
        print(np.max(data[prefix + "t_values"]))

axes[0].axhline(0, color="gray", linestyle=":", linewidth=1)
axes[0].set_ylabel(
    "Mean difference",
    fontsize=difference_ylabel_fontsize
)
axes[0].legend(fontsize=difference_legend_fontsize)

axes[1].axhline(
    threshold, color="red", linestyle="--",
    label=rf"$\pm {threshold:g}$"
)
axes[1].axhline(-threshold, color="red", linestyle="--")
axes[1].set_ylabel("Welch t-statistic", fontsize=tvla_ylabel_fontsize)
axes[1].set_xlabel(
    "Sample within coefficient segment", fontsize=xlabel_fontsize
)
axes[1].legend(fontsize=tvla_legend_fontsize)

for ax in axes:
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.align_ylabels(axes)

fig.savefig(
    input_path.parent / "tvla_comparison_redraw.png",
    dpi=300, bbox_inches="tight"
)
fig.savefig(
    input_path.parent / "tvla_comparison_redraw.pdf",
    bbox_inches="tight"
)
plt.show()