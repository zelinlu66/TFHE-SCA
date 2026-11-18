"""Train the multi-trace MLP on random-ciphertext TFHE captures."""
from pathlib import Path
import json
import numpy as np
from utils import load,segment,key_tokens,split,fit_or_load,aggregate,report
import matplotlib.pyplot as plt

def main():
    # 1. List files containing REPEATED captures of each key.
    # Either repeat key[T,500] for flat traces[T,S], or save traces[K,R,S]
    # with key[K,500]. Separate files with the same keys are also supported.
    input_paths=[Path("D:./data/tfhe_random_ciphertext_training_data.npz")]
    output_dir=Path("D:./data/tfhe_random_ciphertext_multi_trace_results_x4")
    num_segments=500
    timing=(25*4,11*4,11*4)  # first sample, operation spacing, window length
    repeats_to_test=[1,2,3,5,10]
    fusion_method="mean_probability"  # Or "log_odds".
    seed=42
    epochs=10
    # Set to a previously trained random-ciphertext classifier to reload it.
    model_path = Path("./data/tfhe_random_ciphertext_multi_trace_results_x4/classifier.pkl")

    # 2. Group keys before splitting: no acquisition of a test key trains the MLP.
    if output_dir.exists() and model_path is None:
        raise FileExistsError(output_dir)
    elif model_path is None:
        output_dir.mkdir(parents=True)

    x,keys=load(input_paths,num_segments)
    tokens=key_tokens(keys)
    partitions=split(tokens,seed)
    segments=segment(x,timing,num_segments)
    test_keys=sorted(partitions[2])
    minimum=max(repeats_to_test)
    if any(m<1 for m in repeats_to_test):
        raise ValueError("Repeat counts must be positive")
    counts={t:tokens.count(t) for t in test_keys}
    if any(c<minimum for c in counts.values()):
        raise ValueError(f"Every test key needs at least {minimum} traces; counts={list(counts.values())}")

    p,ids=fit_or_load(segments,keys,partitions,output_dir/"classifier.pkl",
                     seed,epochs,model_path)

    # 3. Use the same test keys for every repeat count. Randomize acquisition
    # selection once, without labels; use nested subsets for comparable results.
    rng=np.random.default_rng(seed+1)
    selected=[rng.permutation([i for i,t in enumerate(tokens) if t==k])[:minimum]
              for k in test_keys]
    selected=np.asarray(selected)
    truth=keys[selected[:,0]]
    metrics=[]
    arrays={}
    for m in repeats_to_test:
        fused=np.stack([aggregate(p[row[:m]],fusion_method) for row in selected])
        prediction,errors,result=report(f"{m} traces/key",truth,fused)
        metrics.append(result)
        arrays[f"probability_{m}"]=fused
        arrays[f"predicted_key_{m}"]=prediction
        arrays[f"errors_{m}"]=errors

    # Plot coefficient accuracy and full-key recovery versus traces per key.
    # Sort the results by trace count for plotting.
    order = np.argsort(repeats_to_test)
    trace_counts = np.asarray(repeats_to_test)[order]

    bit_accuracy = np.array([
        result["bit_accuracy"] for result in metrics
    ])[order] * 100

    full_recovery = np.array([
        result["full_key_recovery"] for result in metrics
    ])[order] * 100

    # Configure each font size separately.
    xlabel_fontsize = 20
    ylabel_fontsize = 20
    xtick_fontsize = 18
    ytick_fontsize = 18
    legend_fontsize = 16

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        trace_counts, bit_accuracy, "o-", color="tab:blue",
        label="Coefficient prediction accuracy"
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
    figure_path = output_dir / "accuracy_vs_trace_count.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved {figure_path}")

    np.savez(
        output_dir / "accuracy_vs_trace_count.npz",
        trace_counts=np.asarray(trace_counts),
        bit_accuracy=np.asarray(bit_accuracy),
        full_recovery=np.asarray(full_recovery),
    )

    # 4. Save fused predictions and exactly which acquisitions were combined.
    np.savez_compressed(output_dir/"results.npz",true_keys=truth,
        selected_trace_indices=selected,train_trace_indices=ids[0],
        validation_trace_indices=ids[1],test_trace_indices=ids[2],
        metrics_json=np.array(json.dumps(metrics)),fusion_method=np.array(fusion_method),
        timing=np.array(timing),**arrays)

if __name__=="__main__":
    main()
