"""Train a binary segment classifier; edit settings directly in main()."""
from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Settings. Use the SAME segmentation indices as your TVLA script.
    input_path = Path("./data/tfhe_all_qminus1_mask31.npz")
    output_dir = Path("./data/tfhe_mlp_results_mask31")
    num_segments = 500
    start_index = 25
    segment_spacing = 11
    segment_length = 11

    # Optional relative sample indices within each segment, e.g. [4, 5, 6].
    # Select these using training data or an independent TVLA acquisition.
    feature_indices = None
    seed = 42
    train_fraction = 0.7
    validation_fraction = 0.15
    hidden_layers = (64, 32)
    epochs = 50
    batch_size = 512
    learning_rate = 0.001

    # Avoid replacing earlier models/results.
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output_dir: {output_dir}")
    if (start_index < 0 or segment_spacing < 1 or segment_length < 1
            or segment_length > segment_spacing):
        raise ValueError("Invalid segmentation settings")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1-train_fraction:
        raise ValueError("Invalid split fractions")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    # 2. Load waveforms and keys. Different-key captures use key[T,500].
    with np.load(input_path, allow_pickle=False) as data:
        traces = np.asarray(data["traces"], dtype=np.float64)
        keys = np.asarray(data["key"])
    if traces.ndim != 2 or traces.shape[0] < 3 or not np.isfinite(traces).all():
        raise ValueError("Need at least three finite waveforms [T,samples]")
    total_traces = len(traces)
    if keys.shape != (total_traces, num_segments) or not np.isin(keys, [0, 1]).all():
        raise ValueError(f"Expected binary key array [{total_traces},{num_segments}]")

    # 3. Group identical keys together BEFORE splitting. This also protects
    # against repeated acquisitions of the same key appearing in different sets.
    unique_keys, key_groups = np.unique(keys, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    groups = rng.permutation(len(unique_keys))
    n_train = int(len(groups) * train_fraction)
    n_val = int(len(groups) * validation_fraction)
    if min(n_train, n_val, len(groups)-n_train-n_val) < 1:
        raise ValueError("Too few distinct keys for the requested split")
    train_ids = np.flatnonzero(np.isin(key_groups, groups[:n_train]))
    val_ids = np.flatnonzero(np.isin(key_groups, groups[n_train:n_train+n_val]))
    test_ids = np.flatnonzero(np.isin(key_groups, groups[n_train+n_val:]))
    print(f"Traces: train={len(train_ids)}, validation={len(val_ids)}, test={len(test_ids)}")

    # 4. Extract windows. Flattening keeps coefficient order within each trace.
    starts = start_index + np.arange(num_segments) * segment_spacing
    indices = starts[:, None] + np.arange(segment_length)[None, :]
    if indices[-1, -1] >= traces.shape[1]:
        raise ValueError("Segmentation extends beyond the saved waveform")
    features = np.arange(segment_length) if feature_indices is None else np.asarray(feature_indices)
    if (features.ndim != 1 or features.size == 0 or features.dtype.kind not in "iu"
            or np.any(features < 0) or np.any(features >= segment_length)):
        raise ValueError("Invalid feature_indices")
    datasets = []
    for ids in (train_ids, val_ids, test_ids):
        segments = traces[ids][:, indices]
        x = segments[:, :, features].reshape(-1, len(features))
        y = keys[ids].reshape(-1).astype(np.uint8)
        if len(np.unique(y)) != 2:
            raise ValueError("Every split needs both secret-bit classes")
        datasets.append((x, y))
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = datasets

    # 5. Fit normalization on TRAINING segments only. Do not normalize using
    # the test set or choose sample features based on test-label TVLA results.
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers, activation="relu", solver="adam",
        batch_size=min(batch_size, len(x_train)), learning_rate_init=learning_rate,
        random_state=seed, shuffle=True, early_stopping=False,
    )

    # 6. partial_fit performs one epoch per call. Retain the epoch with best
    # validation log loss; the held-out test set is untouched until selection ends.
    best_loss = np.inf
    best_state = None
    history = []
    for epoch in range(epochs):
        model.partial_fit(x_train_scaled, y_train, classes=np.array([0, 1]))
        p = model.predict_proba(x_val_scaled)[:, 1]
        p = np.clip(p, 1e-12, 1-1e-12)
        val_loss = float(-np.mean(y_val*np.log(p) + (1-y_val)*np.log(1-p)))
        val_accuracy = accuracy_score(y_val, p >= 0.5)
        history.append((float(model.loss_), val_loss, float(val_accuracy)))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = pickle.dumps(model)
            best_epoch = epoch + 1
        print(f"Epoch {epoch+1}/{epochs}: train loss={model.loss_:.4f}, "
              f"val loss={val_loss:.4f}, val accuracy={val_accuracy:.4%}")
    model = pickle.loads(best_state)
    classifier = Pipeline([("scaler", scaler), ("mlp", model)])

    # 7. Evaluate once on held-out keys. Reshape predictions to [traces,500]
    # to report per-key bit accuracy and exact whole-key recovery as well.
    probabilities = classifier.predict_proba(x_test)[:, 1]
    predicted_keys = (probabilities >= 0.5).astype(np.uint8).reshape(-1, num_segments)
    true_keys = keys[test_ids].astype(np.uint8)
    per_key_accuracy = (predicted_keys == true_keys).mean(axis=1)
    exact_recovery = np.all(predicted_keys == true_keys, axis=1)
    cm = confusion_matrix(y_test, predicted_keys.ravel(), labels=[0, 1])
    metrics = {
        "best_epoch": best_epoch,
        "test_bit_accuracy": float(accuracy_score(y_test, predicted_keys.ravel())),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predicted_keys.ravel())),
        "full_key_recovery_rate": float(exact_recovery.mean()),
        "test_traces": len(test_ids), "test_distinct_keys": len(np.unique(true_keys, axis=0)),
        "confusion_matrix_rows_true_columns_predicted": cm.tolist(),
    }
    print(json.dumps(metrics, indent=2))

    # 8. Save the fitted preprocessing/model and all split indices. Pickle
    # files should only be loaded from trusted sources, with matching sklearn.
    output_dir.mkdir(parents=True)
    with (output_dir / "classifier.pkl").open("wb") as output:
        pickle.dump({"classifier": classifier, "start_index": start_index,
                     "segment_spacing": segment_spacing, "segment_length": segment_length,
                     "feature_indices": features, "num_segments": num_segments}, output)
    with (output_dir / "results.npz").open("wb") as output:
        np.savez_compressed(
            output, train_trace_indices=train_ids, validation_trace_indices=val_ids,
            test_trace_indices=test_ids, true_keys=true_keys, predicted_keys=predicted_keys,
            probability_one=probabilities.reshape(-1, num_segments),
            per_key_accuracy=per_key_accuracy, exact_recovery=exact_recovery,
            confusion_matrix=cm, history=np.asarray(history),
            metrics_json=np.array(json.dumps(metrics)), source_path=np.array(str(input_path)),
        )

    # 9. Plot the learning curve and test confusion matrix.
    history = np.asarray(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.arange(1, epochs+1), history[:, 0], label="Train")
    axes[0].plot(np.arange(1, epochs+1), history[:, 1], label="Validation")
    axes[0].axvline(best_epoch, color="gray", linestyle="--")
    axes[0].set(xlabel="Epoch", ylabel="Log loss")
    axes[0].legend()
    axes[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i,j]), ha="center", va="center")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted bit", ylabel="True bit",
                title=f"Held-out bit accuracy: {metrics['test_bit_accuracy']:.2%}")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
