"""Dual-ciphertext inference with independently configured repeats per key."""
from pathlib import Path
import json
import numpy as np
from utils import load,segment,key_tokens,split,fit_or_load,aggregate,report

def main():
    # 1. Settings: grouped traces[K,R,S], key[K,n] or flat repeated captures.
    input_a=[Path("./data/tfhe_all_qminus1_mask32.npz")]
    input_b=[Path("./data/tfhe_all_qminus1_mask31.npz")]
    timing_a=(25,11,11)
    timing_b=(25,11,11)
    output_dir=Path("./data/tfhe_dual_ciphertext_results")
    num_segments=500
    seed=42
    epochs=50

    # INFERENCE acquisitions per test key. Use 1,1 for one trace per ciphertext.
    traces_per_key_a=2
    traces_per_key_b=2
    # False: first requested repeats in saved order. True: randomly select
    # without replacement, reproducibly and independently of labels.
    random_selection=False

    # None trains on ALL available repeats per training/validation key.
    # Set both to 1 to also limit profiling/validation to one trace per key.
    profiling_traces_per_key_a=None
    profiling_traces_per_key_b=None

    # mean_probability weights each selected acquisition equally across A/B.
    # log_odds sums per-acquisition log odds (equal-prior assumption).
    fusion_method="mean_probability"
    model_path_a=output_dir/"classifier_a.pkl"
    model_path_b=output_dir/"classifier_b.pkl"

    # 2. Loading flattens grouped arrays in key-major, then repeat-major order.
    # Flat indices saved below therefore map back as k*R+r for grouped input.
    # if output_dir.exists():
    #     raise FileExistsError(output_dir)
    for count in (traces_per_key_a,traces_per_key_b):
        if not isinstance(count,int) or count<1:
            raise ValueError("Inference repeat counts must be positive integers")
    xa,ka=load(input_a,num_segments)
    xb,kb=load(input_b,num_segments)
    ta,tb=key_tokens(ka),key_tokens(kb)
    if set(ta)!=set(tb):
        raise ValueError("Both inputs must contain exactly the same keys")
    partitions=split(ta,seed)

    # 3. Select repeats per key BEFORE training/inference. All acquisitions
    # belonging to a key stay in its split, even when the repeat counts differ.
    original_ids=[]
    filtered=[]
    for x,k,tokens,inference_count,profiling_count,selection_seed in (
        (xa,ka,ta,traces_per_key_a,profiling_traces_per_key_a,seed+1),
        (xb,kb,tb,traces_per_key_b,profiling_traces_per_key_b,seed+2),
    ):
        if profiling_count is not None and (
            not isinstance(profiling_count,int) or profiling_count<1
        ):
            raise ValueError("Profiling repeat counts must be None or positive")
        rng=np.random.default_rng(selection_seed)
        selected=[]
        for token in sorted(set(tokens)):
            rows=np.array([i for i,t in enumerate(tokens) if t==token],dtype=int)
            count=inference_count if token in partitions[2] else profiling_count
            if count is None:
                count=len(rows)
            if len(rows)<count:
                raise ValueError(f"Requested {count} repeats, available {len(rows)}")
            if random_selection:
                rows=rng.permutation(rows)
            selected.extend(rows[:count])
        selected=np.asarray(selected,dtype=int)
        original_ids.append(selected)
        filtered.append((x[selected],k[selected]))
    (xa,ka),(xb,kb)=filtered

    # 4. Train separate MLPs and scalers using identical KEY partitions.
    output_dir.mkdir(parents=True, exist_ok=True)
    pa,ids_a=fit_or_load(segment(xa,timing_a,num_segments),ka,partitions,
                        output_dir/"classifier_a.pkl",seed,epochs,model_path_a)
    pb,ids_b=fit_or_load(segment(xb,timing_b,num_segments),kb,partitions,
                        output_dir/"classifier_b.pkl",seed,epochs,model_path_b)

    # 5. Pair by complete key contents, then aggregate individual repeat scores.
    # Ground-truth key contents are used only for laboratory grouping/evaluation;
    # deployment must track acquisition/key identity without knowing secret bits.
    tokens_a,tokens_b=key_tokens(ka),key_tokens(kb)
    test_tokens=sorted(partitions[2])
    rows_a=np.array([[i for i,t in enumerate(tokens_a) if t==key] for key in test_tokens])
    rows_b=np.array([[i for i,t in enumerate(tokens_b) if t==key] for key in test_tokens])
    truth=ka[rows_a[:,0]]
    individual_a=pa[rows_a]  # [test_keys,repeats_A,n]
    individual_b=pb[rows_b]
    score_a=np.stack([aggregate(p,fusion_method) for p in individual_a])
    score_b=np.stack([aggregate(p,fusion_method) for p in individual_b])
    # Fuse ORIGINAL per-acquisition scores, avoiding accidental double sigmoid
    # or equal weighting of branches when they have different repeat counts.
    fused=np.stack([aggregate(np.concatenate([a,b],axis=0),fusion_method)
                    for a,b in zip(individual_a,individual_b)])
    pred_a,err_a,ma=report(f"A only: {traces_per_key_a} traces/key",truth,score_a)
    pred_b,err_b,mb=report(f"B only: {traces_per_key_b} traces/key",truth,score_b)
    pred_f,err_f,mf=report("A+B fusion",truth,fused)
    ea,eb=pred_a!=truth,pred_b!=truth
    overlap={"both_wrong_bits":int((ea&eb).sum()),
             "only_a_wrong_bits":int((ea&~eb).sum()),
             "only_b_wrong_bits":int((~ea&eb).sum())}
    print(json.dumps(overlap,indent=2))

    # 6. Save scores and ORIGINAL flattened acquisition indices for auditability.
    config={"traces_per_key_a":traces_per_key_a,"traces_per_key_b":traces_per_key_b,
            "profiling_traces_per_key_a":profiling_traces_per_key_a,
            "profiling_traces_per_key_b":profiling_traces_per_key_b,
            "random_selection":random_selection,"seed":seed,
            "input_a":[str(p) for p in input_a],"input_b":[str(p) for p in input_b]}
    np.savez_compressed(output_dir/"results.npz",true_keys=truth,
        probability_a=score_a,probability_b=score_b,probability_fused=fused,
        individual_probability_a=individual_a,individual_probability_b=individual_b,
        predicted_a=pred_a,predicted_b=pred_b,predicted_fused=pred_f,
        errors_a=err_a,errors_b=err_b,errors_fused=err_f,
        test_original_indices_a=original_ids[0][rows_a],
        test_original_indices_b=original_ids[1][rows_b],
        train_original_indices_a=original_ids[0][ids_a[0]],
        train_original_indices_b=original_ids[1][ids_b[0]],
        validation_original_indices_a=original_ids[0][ids_a[1]],
        validation_original_indices_b=original_ids[1][ids_b[1]],
        metrics_json=np.array(json.dumps([ma,mb,mf])),
        overlap_json=np.array(json.dumps(overlap)),config_json=np.array(json.dumps(config)),
        timing_a=np.array(timing_a),timing_b=np.array(timing_b),
        fusion_method=np.array(fusion_method))

if __name__=="__main__":
    main()
