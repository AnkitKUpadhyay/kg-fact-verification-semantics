# #!/usr/bin/env python3
# """
# Compare two LLM runs on the same claims.

# - Supports JSONL or CSV inputs for each run (A and B)
# - Aligns by `claim` key (inner join)
# - Uses labels from either inputs or an external gold file (--gold)
# - Computes: Accuracy, Macro-F1, Negation Acc, per-type Acc (if available),
#   ECE, Acc@90% coverage, McNemar significance, Bootstrap CIs, agreement,
#   Spearman correlation of confidences
# - Writes a per-example diff CSV for manual inspection

# Usage:
#   python compare_llm_runs.py --a runA.jsonl --b runB.jsonl
#   python compare_llm_runs.py --a runA.csv --b runB.jsonl --gold filtered_test_k10.pkl
# """

# import argparse, json, math, random
# from pathlib import Path
# from typing import Tuple, List

# import numpy as np
# import pandas as pd
# from scipy.stats import spearmanr
# from statsmodels.stats.contingency_tables import mcnemar

# # ------------------------
# # Helpers
# # ------------------------
# def load_any(path: Path) -> pd.DataFrame:
#     if path.suffix.lower() == ".jsonl":
#         rows = [json.loads(x) for x in open(path, "r", encoding="utf-8")]
#         return pd.DataFrame(rows)
#     elif path.suffix.lower() == ".json":
#         data = json.load(open(path, "r", encoding="utf-8"))
#         return pd.DataFrame(data if isinstance(data, list) else data["rows"])
#     elif path.suffix.lower() == ".csv":
#         return pd.read_csv(path)
#     elif path.suffix.lower() in {".pkl", ".pickle"}:
#         return pd.read_pickle(path)
#     else:
#         raise ValueError(f"Unsupported file type: {path}")

# def norm_label(x):
#     if isinstance(x, (list, tuple)) and len(x) == 1:
#         x = x[0]
#     s = str(x).strip().lower()
#     if s in {"1","true","supports","supported"}:
#         return 1
#     if s in {"0","false","refutes","refuted"}:
#         return 0
#     # fallback: treat unknown truthy as 1
#     return 1 if x in (True, 1) else 0

# def norm_verdict(x):
#     s = str(x).strip().lower()
#     if "support" in s:
#         return 1
#     if "refut" in s or "contradict" in s:
#         return 0
#     # fallback if already numeric
#     if s in {"1","0"}:
#         return int(s)
#     return None

# def clip01(v):
#     try:
#         f = float(v)
#     except Exception:
#         return np.nan
#     return max(0.0, min(1.0, f))

# def macro_f1(y_true, y_pred):
#     # binary macro-F1
#     from sklearn.metrics import f1_score
#     return f1_score(y_true, y_pred, average="macro")

# def expected_calibration_error(y_true, y_prob, n_bins=10):
#     y_true = np.asarray(y_true).astype(int)
#     y_prob = np.asarray(y_prob).astype(float)
#     edges = np.linspace(0, 1, n_bins+1)
#     idx = np.digitize(y_prob, edges[:-1]) - 1
#     idx = np.clip(idx, 0, n_bins-1)
#     ece = 0.0
#     for b in range(n_bins):
#         mask = (idx == b)
#         if mask.sum() == 0: 
#             continue
#         acc = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
#         conf = y_prob[mask].mean()
#         ece += (mask.mean()) * abs(acc - conf)
#     return ece

# def acc_at_coverage(y_true, y_pred, y_prob, coverage=0.9):
#     order = np.argsort(-np.asarray(y_prob))
#     k = max(1, int(math.ceil(coverage * len(order))))
#     sel = order[:k]
#     return (np.asarray(y_true)[sel] == np.asarray(y_pred)[sel]).mean(), float(np.asarray(y_prob)[order[k-1]])

# def bootstrap_ci_acc(y_true, y_pred, iters=2000, seed=7):
#     rng = np.random.default_rng(seed)
#     y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
#     n = len(y_true)
#     accs = []
#     for _ in range(iters):
#         idx = rng.integers(0, n, size=n)
#         accs.append((y_true[idx] == y_pred[idx]).mean())
#     lo, hi = np.percentile(accs, [2.5, 97.5])
#     return lo, hi

# def bootstrap_ci_delta_acc(y_true, y_pred_a, y_pred_b, iters=2000, seed=7):
#     rng = np.random.default_rng(seed)
#     y_true = np.asarray(y_true)
#     a = np.asarray(y_pred_a); b = np.asarray(y_pred_b)
#     n = len(y_true)
#     deltas = []
#     for _ in range(iters):
#         idx = rng.integers(0, n, size=n)
#         deltas.append(((y_true[idx]==a[idx]).mean() - (y_true[idx]==b[idx]).mean()))
#     lo, hi = np.percentile(deltas, [2.5, 97.5])
#     return lo, hi

# # ------------------------
# # Main
# # ------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--a", required=True, help="Run A file (JSONL/CSV/PKL)")
#     ap.add_argument("--b", required=True, help="Run B file (JSONL/CSV/PKL)")
#     ap.add_argument("--gold", default=None, help="Optional gold file with claim+label (PKL/CSV/JSONL)")
#     ap.add_argument("--out", default="llm_head_to_head.csv", help="Per-example diff CSV")
#     ap.add_argument("--bins", type=int, default=10, help="ECE bins")
#     args = ap.parse_args()

#     A = load_any(Path(args.a)).copy()
#     B = load_any(Path(args.b)).copy()

#     # Rename columns defensively
#     # Expected: claim, label, verdict, (confidence), (reasoning_type)
#     def standardize(df, tag):
#         cols = {c.lower(): c for c in df.columns}
#         def pick(*names):
#             for n in names:
#                 if n in df.columns:
#                     return n
#                 if n.lower() in cols:
#                     return cols[n.lower()]
#             return None
#         c_claim = pick("claim")
#         c_label = pick("label","gold","y","target")
#         c_pred  = pick("verdict","prediction","pred","y_hat")
#         c_conf  = pick("confidence","prob","score","confidence_score")
#         c_type  = pick("reasoning_type","type","types")
#         out = pd.DataFrame()
#         if c_claim is None or c_pred is None:
#             raise ValueError(f"{tag}: missing required columns (need claim + verdict/prediction).")
#         out["claim"] = df[c_claim].astype(str)
#         if c_label is not None:
#             out["label"] = df[c_label]
#         if c_type is not None:
#             # If it's a list like ['coll:model','existence'], keep last token
#             rt = df[c_type]
#             out["reasoning_type"] = rt.apply(lambda v: v[-1] if isinstance(v, (list,tuple)) and len(v)>0 else v)
#         if c_pred is not None:
#             out["verdict_raw"] = df[c_pred]
#             out["pred"] = df[c_pred].apply(norm_verdict)
#         if c_conf is not None:
#             out["conf"] = df[c_conf].apply(clip01)
#         else:
#             out["conf"] = np.nan
#         return out

#     A = standardize(A, "A")
#     B = standardize(B, "B")

#     # Optional gold
#     if args.gold:
#         G = load_any(Path(args.gold)).copy()
#         # try to standardize gold
#         if "claim" not in G.columns:
#             # maybe dict with claim as key
#             if isinstance(G, dict):  # unlikely with DataFrame, but just in case
#                 G = pd.DataFrame([{"claim":k, **v} for k,v in G.items()])
#             else:
#                 # common case: filtered pkl has 'claim' column
#                 pass
#         if "claim" not in G.columns:
#             raise ValueError("Gold file must contain 'claim' column.")
#         # find label column
#         cand = [c for c in G.columns if c.lower() in {"label","gold","y","target"}]
#         if not cand:
#             # filtered pkl uses 'label'
#             cand = ["label"] if "label" in G.columns else []
#         if not cand:
#             raise ValueError("Gold file must contain a label-like column.")
#         gold = G[["claim", cand[0]]].copy()
#         gold.rename(columns={cand[0]:"label"}, inplace=True)
#         gold["label"] = gold["label"].apply(norm_label)
#         # merge gold into both (by claim)
#         A = A.drop(columns=[c for c in ["label"] if c in A.columns]).merge(gold, on="claim", how="inner")
#         B = B.drop(columns=[c for c in ["label"] if c in B.columns]).merge(gold, on="claim", how="inner")
#     else:
#         # rely on labels inside A/B
#         if "label" not in A.columns and "label" not in B.columns:
#             raise ValueError("No gold provided and neither input has 'label'.")
#         # unify labels if present in one side only
#         if "label" not in A.columns and "label" in B.columns:
#             A = A.merge(B[["claim","label"]], on="claim", how="inner")
#         if "label" not in B.columns and "label" in A.columns:
#             B = B.merge(A[["claim","label"]], on="claim", how="inner")
#         A["label"] = A["label"].apply(norm_label)
#         B["label"] = B["label"].apply(norm_label)

#     # Inner join A vs B by claim
#     cols_keep = ["claim","label","pred","conf","verdict_raw"]
#     if "reasoning_type" in A.columns:
#         cols_keep.append("reasoning_type")
#     AB = A[cols_keep].merge(B[cols_keep], on="claim", suffixes=("_A","_B"), how="inner")

#     # Drop rows with missing predictions
#     AB = AB.dropna(subset=["pred_A","pred_B"])
#     n = len(AB)
#     if n == 0:
#         raise ValueError("No aligned rows after join on 'claim'.")

#     # Metrics
#     y  = AB["label"].to_numpy().astype(int)
#     pa = AB["pred_A"].to_numpy().astype(int)
#     pb = AB["pred_B"].to_numpy().astype(int)
#     ca = AB["conf_A"].fillna(0.5).to_numpy().astype(float)
#     cb = AB["conf_B"].fillna(0.5).to_numpy().astype(float)

#     def acc(y, p): return (y==p).mean()

#     print(f"\nAligned rows: {n}")
#     print("\n== Overall ==")
#     print(f"A Acc: {acc(y,pa):.4f}  | Macro-F1: {macro_f1(y,pa):.4f}")
#     print(f"B Acc: {acc(y,pb):.4f}  | Macro-F1: {macro_f1(y,pb):.4f}")

#     # Negation & per-type (if available)
#     if "reasoning_type_A" in AB.columns:
#         rt = AB["reasoning_type_A"].fillna("unknown")
#         neg_mask = rt.str.contains("negation", case=False, na=False)
#         if neg_mask.any():
#             print("\n== Negation subset ==")
#             print(f"A Negation Acc: {acc(y[neg_mask], pa[neg_mask]):.4f}")
#             print(f"B Negation Acc: {acc(y[neg_mask], pb[neg_mask]):.4f}")
#         print("\n== Per reasoning type (accuracy) ==")
#         for t, sub in AB.groupby(rt):
#             yt = sub["label"].to_numpy()
#             pat = sub["pred_A"].to_numpy()
#             pbt = sub["pred_B"].to_numpy()
#             print(f"{t:20s}  A:{acc(yt,pat):.4f}  B:{acc(yt,pbt):.4f}  n={len(sub)}")

#     # Calibration
#     ece_a = expected_calibration_error(y, ca, n_bins=10)
#     ece_b = expected_calibration_error(y, cb, n_bins=10)
#     acc90_a, th_a = acc_at_coverage(y, pa, ca, coverage=0.9)
#     acc90_b, th_b = acc_at_coverage(y, pb, cb, coverage=0.9)
#     print("\n== Calibration ==")
#     print(f"A ECE: {ece_a:.4f} | Acc@90%: {acc90_a:.4f} (thr~{th_a:.3f})")
#     print(f"B ECE: {ece_b:.4f} | Acc@90%: {acc90_b:.4f} (thr~{th_b:.3f})")

#     # Head-to-head significance: McNemar
#     # table:
#     #             A correct
#     #             yes   no
#     # B correct yes  n11  n01
#     #           no   n10  n00
#     a_correct = (y==pa)
#     b_correct = (y==pb)
#     n11 = int(( a_correct &  b_correct).sum())
#     n01 = int((~a_correct &  b_correct).sum())
#     n10 = int(( a_correct & ~b_correct).sum())
#     n00 = int((~a_correct & ~b_correct).sum())
#     table = [[n11, n01],[n10, n00]]
#     res = mcnemar(table, exact=False, correction=True)
#     print("\n== Head-to-head ==")
#     print(f"Agreement: {(pa==pb).mean():.4f}")
#     print(f"Contingency [[n11,n01],[n10,n00]] = {table}")
#     print(f"McNemar chi2={res.statistic:.4f}, p={res.pvalue:.6f}  (p<0.05 => significant difference)")

#     # Confidence correlation (Spearman)
#     # only if both have non-NaN confidences
#     if not (np.isnan(ca).all() or np.isnan(cb).all()):
#         rho, prho = spearmanr(ca, cb, nan_policy="omit")
#         print(f"\nConfidence Spearman rho={rho:.3f} (p={prho:.2g})")

#     # Bootstrap CIs
#     lo_a, hi_a = bootstrap_ci_acc(y, pa)
#     lo_b, hi_b = bootstrap_ci_acc(y, pb)
#     lo_d, hi_d = bootstrap_ci_delta_acc(y, pa, pb)
#     print("\n== Bootstrap 95% CI ==")
#     print(f"A Acc CI: [{lo_a:.4f}, {hi_a:.4f}]")
#     print(f"B Acc CI: [{lo_b:.4f}, {hi_b:.4f}]")
#     print(f"ΔAcc (A-B) CI: [{lo_d:.4f}, {hi_d:.4f}]")

#     # Write per-example diff
#     out = Path(args.out)
#     cols = [
#         "claim","label",
#         "verdict_raw_A","pred_A","conf_A",
#         "verdict_raw_B","pred_B","conf_B"
#     ]
#     if "reasoning_type_A" in AB.columns:
#         cols.insert(1, "reasoning_type_A")
#     AB[cols].to_csv(out, index=False)
#     print(f"\nWrote per-example diff: {out}")

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# import argparse, json
# from pathlib import Path
# import numpy as np
# import pandas as pd

# TOKENS_PER_TRIPLE = 13.0  # empirical from your stats; good enough for compare

# def load_jsonl(p: Path) -> pd.DataFrame:
#     rows = [json.loads(x) for x in open(p, "r", encoding="utf-8")]
#     return pd.DataFrame(rows)

# def norm_triple(t):
#     if isinstance(t, (list, tuple)) and len(t) == 3:
#         s, p, o = t
#         return (str(s).strip(), str(p).strip(), str(o).strip())
#     return tuple([str(t)])

# def ensure_fields(df: pd.DataFrame, tag: str) -> pd.DataFrame:
#     df = df.copy()
#     if "filtered_triples" not in df.columns:
#         raise ValueError(f"{tag} missing 'filtered_triples'")
#     if "claim" not in df.columns:
#         raise ValueError(f"{tag} missing 'claim'")

#     # auto-compute num_filtered if missing
#     if "num_filtered" not in df.columns:
#         df["num_filtered"] = df["filtered_triples"].apply(lambda ts: len(ts or []))

#     # auto-compute approx_tokens if missing
#     if "approx_tokens" not in df.columns:
#         # ~ claim avg tokens (18–19) + k * 13 + a bit of overhead
#         df["approx_tokens"] = (19.0 + df["num_filtered"] * TOKENS_PER_TRIPLE + 10.0).astype(float)

#     # optional: keep label if present
#     if "label" not in df.columns:
#         df["label"] = np.nan

#     return df[["claim","label","filtered_triples","num_filtered","approx_tokens"]]

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--a", required=True, help="Filtering output A (JSONL)")
#     ap.add_argument("--b", required=True, help="Filtering output B (JSONL)")
#     ap.add_argument("--out", default="compare_filtering.csv")
#     args = ap.parse_args()

#     A = ensure_fields(load_jsonl(Path(args.a)), "A")
#     B = ensure_fields(load_jsonl(Path(args.b)), "B")

#     AB = A.merge(B, on="claim", suffixes=("_A","_B"), how="inner")
#     if len(AB) == 0:
#         raise ValueError("No overlapping claims between runs.")
#     print(f"Aligned claims: {len(AB)}")

#     jaccs, overlap_at_minK, d_kept, d_tokens = [], [], [], []
#     rows = []
#     for _, r in AB.iterrows():
#         ta = [norm_triple(t) for t in (r["filtered_triples_A"] or [])]
#         tb = [norm_triple(t) for t in (r["filtered_triples_B"] or [])]
#         Sa, Sb = set(ta), set(tb)

#         inter = len(Sa & Sb)
#         union = max(1, len(Sa | Sb))
#         j = inter / union

#         kmin = min(len(ta), len(tb))
#         topA, topB = set(ta[:kmin]), set(tb[:kmin])
#         o_at_minK = (len(topA & topB) / max(1, kmin)) if kmin > 0 else 1.0

#         dk   = (r["num_filtered_A"] or 0) - (r["num_filtered_B"] or 0)
#         dtok = float(r["approx_tokens_A"]) - float(r["approx_tokens_B"])

#         jaccs.append(j)
#         overlap_at_minK.append(o_at_minK)
#         d_kept.append(dk)
#         d_tokens.append(dtok)

#         rows.append({
#             "claim": r["claim"],
#             "label": r["label_A"] if not pd.isna(r["label_A"]) else r["label_B"],
#             "num_filtered_A": r["num_filtered_A"],
#             "num_filtered_B": r["num_filtered_B"],
#             "approx_tokens_A": r["approx_tokens_A"],
#             "approx_tokens_B": r["approx_tokens_B"],
#             "jaccard": j,
#             "overlap_at_minK": o_at_minK,
#             "delta_kept": dk,
#             "delta_tokens": dtok,
#         })

#     print(f"Mean Jaccard overlap:       {np.mean(jaccs):.3f}")
#     print(f"Mean Overlap@min(kA,kB):    {np.mean(overlap_at_minK):.3f}")
#     print(f"Mean Δ kept (A-B) triples:  {np.mean(d_kept):+.2f}")
#     print(f"Mean Δ tokens (A-B):        {np.mean(d_tokens):+.1f}")

#     out = Path(args.out)
#     pd.DataFrame(rows).to_csv(out, index=False)
#     print(f"Wrote per-claim comparison: {out}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TOKENS_PER_TRIPLE = 13.0  # empirical from your stats; good enough for compare

def load_jsonl(p: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in open(p, "r", encoding="utf-8")]
    return pd.DataFrame(rows)

def load_pkl_for_reasoning_types(jsonl_path: Path) -> pd.DataFrame:
    """Load corresponding PKL file to get reasoning_type for each claim."""
    # Try to find matching PKL file
    pkl_path = jsonl_path.with_suffix('.pkl')
    if pkl_path.exists():
        df = pd.read_pickle(pkl_path)
        if 'reasoning_type' in df.columns and 'claim' in df.columns:
            return df[['claim', 'reasoning_type']]
    return pd.DataFrame(columns=['claim', 'reasoning_type'])

def norm_triple(t):
    if isinstance(t, (list, tuple)) and len(t) == 3:
        s, p, o = t
        return (str(s).strip(), str(p).strip(), str(o).strip())
    return tuple([str(t)])

def ensure_fields(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    df = df.copy()
    if "filtered_triples" not in df.columns:
        raise ValueError(f"{tag} missing 'filtered_triples'")
    if "claim" not in df.columns:
        raise ValueError(f"{tag} missing 'claim'")
    # auto-compute num_filtered if missing
    if "num_filtered" not in df.columns:
        df["num_filtered"] = df["filtered_triples"].apply(lambda ts: len(ts or []))
    # auto-compute approx_tokens if missing
    if "approx_tokens" not in df.columns:
        # ~ claim avg tokens (18–19) + k * 13 + a bit of overhead
        df["approx_tokens"] = (19.0 + df["num_filtered"] * TOKENS_PER_TRIPLE + 10.0).astype(float)
    # optional: keep label if present
    if "label" not in df.columns:
        df["label"] = np.nan
    return df[["claim","label","filtered_triples","num_filtered","approx_tokens"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Filtering output A (JSONL)")
    ap.add_argument("--b", required=True, help="Filtering output B (JSONL)")
    ap.add_argument("--out", default="compare_filtering.csv")
    args = ap.parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)
    
    A = ensure_fields(load_jsonl(path_a), "A")
    B = ensure_fields(load_jsonl(path_b), "B")
    
    # Try to load reasoning types from PKL files
    reasoning_a = load_pkl_for_reasoning_types(path_a)
    reasoning_b = load_pkl_for_reasoning_types(path_b)
    
    # Merge reasoning types (prefer A, fallback to B)
    if not reasoning_a.empty:
        A = A.merge(reasoning_a, on="claim", how="left")
    if not reasoning_b.empty and 'reasoning_type' not in A.columns:
        A = A.merge(reasoning_b, on="claim", how="left")
    
    AB = A.merge(B, on="claim", suffixes=("_A","_B"), how="inner")
    
    if len(AB) == 0:
        raise ValueError("No overlapping claims between runs.")

    print(f"Aligned claims: {len(AB)}")

    jaccs, overlap_at_minK, d_kept, d_tokens = [], [], [], []
    rows = []

    for _, r in AB.iterrows():
        ta = [norm_triple(t) for t in (r["filtered_triples_A"] or [])]
        tb = [norm_triple(t) for t in (r["filtered_triples_B"] or [])]
        Sa, Sb = set(ta), set(tb)
        inter = len(Sa & Sb)
        union = max(1, len(Sa | Sb))
        j = inter / union
        kmin = min(len(ta), len(tb))
        topA, topB = set(ta[:kmin]), set(tb[:kmin])
        o_at_minK = (len(topA & topB) / max(1, kmin)) if kmin > 0 else 1.0
        dk = (r["num_filtered_A"] or 0) - (r["num_filtered_B"] or 0)
        dtok = float(r["approx_tokens_A"]) - float(r["approx_tokens_B"])
        
        jaccs.append(j)
        overlap_at_minK.append(o_at_minK)
        d_kept.append(dk)
        d_tokens.append(dtok)
        
        row = {
            "claim": r["claim"],
            "label": r["label_A"] if not pd.isna(r["label_A"]) else r["label_B"],
            "num_filtered_A": r["num_filtered_A"],
            "num_filtered_B": r["num_filtered_B"],
            "approx_tokens_A": r["approx_tokens_A"],
            "approx_tokens_B": r["approx_tokens_B"],
            "jaccard": j,
            "overlap_at_minK": o_at_minK,
            "delta_kept": dk,
            "delta_tokens": dtok,
        }
        
        # Add reasoning_type if available
        if 'reasoning_type' in r.index and not pd.isna(r['reasoning_type']):
            row['reasoning_type'] = r['reasoning_type']
        
        rows.append(row)

    print(f"\n=== Overall Statistics ===")
    print(f"Mean Jaccard overlap:       {np.mean(jaccs):.3f}")
    print(f"Mean Overlap@min(kA,kB):    {np.mean(overlap_at_minK):.3f}")
    print(f"Mean Δ kept (A-B) triples:  {np.mean(d_kept):+.2f}")
    print(f"Mean Δ tokens (A-B):        {np.mean(d_tokens):+.1f}")

    df_out = pd.DataFrame(rows)
    
    # Per-reasoning-type analysis
    if 'reasoning_type' in df_out.columns:
        print(f"\n=== Per Reasoning Type Statistics ===")
        grouped = df_out.groupby('reasoning_type').agg({
            'jaccard': ['mean', 'std', 'count'],
            'overlap_at_minK': 'mean',
            'delta_kept': 'mean',
            'num_filtered_A': 'mean',
            'num_filtered_B': 'mean'
        }).round(3)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.rename(columns={
            'jaccard_mean': 'jaccard',
            'jaccard_std': 'jaccard_std',
            'jaccard_count': 'n',
            'overlap_at_minK_mean': 'overlap@minK',
            'delta_kept_mean': 'Δ_kept',
            'num_filtered_A_mean': 'avg_k_A',
            'num_filtered_B_mean': 'avg_k_B'
        })
        
        print(grouped.to_string())
        
        # Show which reasoning types have lowest overlap (most disagreement)
        print(f"\n=== Reasoning Types Ranked by Disagreement ===")
        type_jaccard = df_out.groupby('reasoning_type')['jaccard'].mean().sort_values()
        for rt, j in type_jaccard.items():
            print(f"  {rt:20s}: jaccard={j:.3f}")

    out = Path(args.out)
    df_out.to_csv(out, index=False)
    print(f"\nWrote per-claim comparison: {out}")

if __name__ == "__main__":
    main()