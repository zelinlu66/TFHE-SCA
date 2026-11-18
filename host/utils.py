"""Shared training/segmentation utilities for the two experiments."""
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load(paths, n):
    # Accept flat [acquisitions,samples] or grouped [keys,repeats,samples].
    waves, keys = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            x, k = np.asarray(d["traces"]), np.asarray(d["key"])
            if x.ndim == 3:
                t, r, width = x.shape
                if k.shape == (t,n):
                    k = np.repeat(k, r, axis=0)
                elif k.shape == (t,r,n):
                    k = k.reshape(-1,n)
                else:
                    raise ValueError(f"Key shape does not match grouped traces: {path}")
                x = x.reshape(-1,width)
            if x.ndim != 2 or k.shape != (len(x),n):
                raise ValueError(f"Expected traces[T,S], key[T,{n}]: {path}")
            if not np.isfinite(x).all() or not np.isin(k,[0,1]).all():
                raise ValueError("Nonfinite samples or nonbinary keys")
            print(f"{path}: {len(x)} acquisitions")
            if "a" in d:
                print("Saved a values:", np.unique(d["a"]))
            waves.append(x); keys.append(k.astype(np.uint8))
    return np.concatenate(waves), np.concatenate(keys)

def segment(x, timing, n):
    start, spacing, length = timing
    if start < 0 or spacing < 1 or length < 1 or length > spacing:
        raise ValueError("Invalid segment timing")
    index = start + np.arange(n)[:,None]*spacing + np.arange(length)[None,:]
    if index[-1,-1] >= x.shape[1]:
        raise ValueError("Segmentation exceeds waveform")
    return np.asarray(x[:,index],dtype=np.float64)

def key_tokens(keys):
    return [row.tobytes() for row in keys]

def split(tokens, seed, fractions=(0.7,0.15)):
    unique = sorted(set(tokens))
    rng = np.random.default_rng(seed)
    unique = [unique[i] for i in rng.permutation(len(unique))]
    a, b = int(len(unique)*fractions[0]), int(len(unique)*fractions[1])
    if min(a,b,len(unique)-a-b)<1:
        raise ValueError("Too few distinct keys for train/validation/test")
    return [set(unique[:a]),set(unique[a:a+b]),set(unique[a+b:])]

def fit_or_load(segments, keys, partitions, output, seed, epochs, model_path=None):
    tokens = key_tokens(keys)
    ids = [np.array([i for i,t in enumerate(tokens) if t in group],dtype=int)
           for group in partitions]
    if any(len(i)==0 for i in ids):
        raise ValueError("A data split has no acquisitions")
    if model_path is not None:
        # Only artifacts written by this code include key provenance.
        with Path(model_path).open("rb") as f:
            saved=pickle.load(f)
        if not set(saved["training_key_tokens"]).isdisjoint(partitions[1] | partitions[2]):
            raise ValueError("Loaded model trained on current validation/test keys")
        if not set(saved["validation_key_tokens"]).isdisjoint(partitions[2]):
            raise ValueError("Loaded model selected using current test keys")
        model=saved["classifier"]
    else:
        xt=segments[ids[0]].reshape(-1,segments.shape[-1])
        yt=keys[ids[0]].reshape(-1)
        xv=segments[ids[1]].reshape(-1,segments.shape[-1])
        yv=keys[ids[1]].reshape(-1)
        if len(np.unique(yt))!=2:
            raise ValueError("Training needs both classes")
        scaler=StandardScaler().fit(xt)
        xt=scaler.transform(xt); xv=scaler.transform(xv)
        mlp=MLPClassifier(hidden_layer_sizes=(64,32),batch_size=min(512,len(xt)),
                          learning_rate_init=0.001,random_state=seed)
        best = float("inf")
        state = None
        history = []

        for epoch in range(epochs):
            mlp.partial_fit(xt, yt, classes=np.array([0, 1]))

            train_prob = mlp.predict_proba(xt)[:, 1]
            val_prob = mlp.predict_proba(xv)[:, 1]

            train_p = np.clip(train_prob, 1e-9, 1 - 1e-9)
            val_p = np.clip(val_prob, 1e-9, 1 - 1e-9)

            train_loss = -np.mean(yt * np.log(train_p) + (1 - yt) * np.log1p(-train_p))
            val_loss = -np.mean(yv * np.log(val_p) + (1 - yv) * np.log1p(-val_p))
            train_acc = 100 * np.mean((train_prob > 0.5) == yt)
            val_acc = 100 * np.mean((val_prob > 0.5) == yv)

            history.append((train_loss, val_loss, train_acc, val_acc))

            if val_loss < best:
                best = val_loss
                state = pickle.dumps(mlp)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train loss={train_loss:.6f}, acc={train_acc:.2f}% | "
                f"val loss={val_loss:.6f}, acc={val_acc:.2f}%"
            )
        if state is None:
            raise ValueError("epochs must be positive")
        model=Pipeline([("scaler",scaler),("mlp",pickle.loads(state))])
        with Path(output).open("wb") as f:
            pickle.dump({"classifier":model,
                         "training_key_tokens":sorted(partitions[0]),
                         "validation_key_tokens":sorted(partitions[1])},f)
        train_loss, val_loss, train_acc, val_acc = np.asarray(history).T
        np.savez(
            Path(output).with_suffix(".history.npz"),
            epoch=np.arange(1, len(history) + 1),
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
        )
    p=model.predict_proba(segments.reshape(-1,segments.shape[-1]))[:,1]
    return p.reshape(segments.shape[:2]), ids

def aggregate(probabilities, method):
    # Input [repeats,coefficients]. Equal binary class prior is assumed for
    # summed log odds. Correlated repeats can make these scores overconfident.
    p=np.clip(probabilities,1e-9,1-1e-9)
    if method=="mean_probability":
        return p.mean(axis=0)
    if method=="log_odds":
        score=(np.log(p)-np.log1p(-p)).sum(axis=0)
        return 1/(1+np.exp(-np.clip(score,-700,700)))
    raise ValueError(method)

def report(name, truth, probabilities):
    predicted=(probabilities>=0.5).astype(np.uint8)
    errors=(predicted!=truth).sum(axis=1)
    metrics={"name":name,"keys":len(truth),"bit_accuracy":float((predicted==truth).mean()),
             "full_key_recovery":float((errors==0).mean()),
             "full_keys_correct":int((errors==0).sum()),"mean_errors_per_key":float(errors.mean())}
    print(json.dumps(metrics,indent=2))
    return predicted,errors,metrics
