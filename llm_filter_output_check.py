#!/usr/bin/env python3
"""
LLM-Filtered Output Sanity Checker
Usage:
  python llm_filter_output_check.py --split test --k 10

What it does:
- Loads results/llm_filtered/filtered_<split>_k<k>.{pkl,jsonl}
- Validates schema, label ranges, and length invariants
- Tokenizes (pair mode) to measure avg length and truncation @512
- Shows label / reasoning-type distributions and triple counts
- Spot-checks reduced/unchanged/empty examples
- (Optional) Compares token stats vs. UNFILTERED first N examples
- Writes a small CSV preview for manual inspection
"""

import os, json, pickle, random, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","dev","test"], default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--subgraph_dir", type=str, default="data/subgraphs")
    ap.add_argument("--filtered_dir", type=str, default="results/llm_filtered")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--unfiltered_compare_n", type=int, default=500,
                    help="Compare first N unfiltered examples (set 0 to disable)")
    ap.add_argument("--preview_n", type=int, default=25,
                    help="Rows in preview CSV")
    args = ap.parse_args()

    SPLIT = args.split
    K = args.k
    MAX_LEN = args.max_len
    FILTERED_DIR = Path(args.filtered_dir)
    SUBGRAPH_DIR = Path(args.subgraph_dir)
    DATA_DIR = Path(args.data_dir)

    filtered_pkl   = FILTERED_DIR / f"filtered_{SPLIT}_k{K}.pkl"
    filtered_jsonl = FILTERED_DIR / f"filtered_{SPLIT}_k{K}.jsonl"

    assert filtered_pkl.exists(),   f"Missing {filtered_pkl}"
    assert filtered_jsonl.exists(), f"Missing {filtered_jsonl}"

    print(f"Loading filtered PKL:   {filtered_pkl}")
    df = pd.read_pickle(filtered_pkl)
    print(f"  -> rows: {len(df)}")

    print(f"Loading filtered JSONL: {filtered_jsonl}")
    jsonl_rows = [json.loads(line) for line in open(filtered_jsonl, "r", encoding="utf-8")]
    print(f"  -> rows: {len(jsonl_rows)}")

    # --- Basic schema & invariants ---
    required_cols = {"claim","label","filtered_triples","num_original","num_filtered","approx_tokens"}
    missing = required_cols - set(df.columns)
    print("\nSchema check:")
    print("  Missing columns:", missing)
    assert not missing, "Filtered PKL missing required columns"

    assert df["label"].isin([0,1]).all(), "Labels must be 0/1"
    assert (df["num_filtered"] <= K).all(), "num_filtered cannot exceed K"
    assert (df["num_filtered"] <= df["num_original"]).all(), "num_filtered cannot exceed num_original"
    assert (df["num_filtered"] == df["filtered_triples"].apply(len)).all(), "num_filtered must equal len(filtered_triples)"

    def jsonl_row_ok(r):
        return all(k in r for k in ("claim","label","filtered_triples","k"))
    ok_jsonl = all(jsonl_row_ok(r) for r in jsonl_rows)
    print("  JSONL rows valid schema:", ok_jsonl)

    # --- Tokenization & truncation (pair mode) ---
    print("\nLoading tokenizer: bert-base-uncased")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def lin_triples(triples: List[Tuple[str,str,str]]) -> str:
        return " [TRI] ".join([f"{s} {p} {o}" for (s,p,o) in triples])

    print("Computing token lengths & truncation (filtered)...")
    lens, trunc = [], []
    for _, r in df.iterrows():
        enc = tokenizer(
            r["claim"],
            lin_triples(r["filtered_triples"]),
            add_special_tokens=True,
            truncation="only_second",
            max_length=MAX_LEN,
        )
        L = len(enc["input_ids"])
        lens.append(L)
        trunc.append(int(L > MAX_LEN))

    lens = np.array(lens); trunc = np.array(trunc)
    print(f"Filtered tokens: mean={lens.mean():.1f}  median={np.median(lens):.1f}  max={lens.max()}")
    print(f"Filtered truncation @{MAX_LEN}: {100*trunc.mean():.2f}% (should be ~0%)")

    # --- Distributions ---
    print("\nLabel distribution (prop):")
    print(df["label"].value_counts(normalize=True).rename("prop").round(3))

    if "reasoning_type" in df.columns:
        print("\nTop reasoning types:")
        print(df["reasoning_type"].value_counts().head(10))

    print("\nTriples per example:")
    print(df[["num_original","num_filtered"]].describe().round(2))

    # --- Spot checks ---
    def show_record(idx):
        r = df.iloc[idx]
        print(f"\nIDX {idx} | label={r['label']} | orig={r['num_original']} | kept={r['num_filtered']}")
        print("CLAIM:", r["claim"])
        for t in r["filtered_triples"][:min(10, len(r["filtered_triples"]))]:
            print("  -", t)

    reduced   = df.index[df["num_filtered"] < df["num_original"]].tolist()
    unchanged = df.index[df["num_filtered"] == df["num_original"]].tolist()
    empties   = df.index[df["num_filtered"] == 0].tolist()

    print(f"\nReduced examples: {len(reduced)} | Unchanged: {len(unchanged)} | Empty: {len(empties)}")
    for idx in (reduced[:2] + unchanged[:1] + empties[:1]):
        if len(df) > 0 and idx < len(df):
            show_record(idx)

    # --- Optional: compare against UNFILTERED first N examples ---
    if args.unfiltered_compare_n > 0:
        n = min(args.unfiltered_compare_n, len(df))
        subgraph_pkl = SUBGRAPH_DIR / f"subgraphs_one_hop_{SPLIT}.pkl"
        claims_pkl   = DATA_DIR / f"factkg/factkg_{SPLIT}.pickle"
        if subgraph_pkl.exists() and claims_pkl.exists():
            print(f"\n[Unfiltered] Comparing first {n} examples...")
            subs_df = pd.read_pickle(subgraph_pkl)
            with open(claims_pkl, "rb") as f:
                claims_dict = pickle.load(f)
            claims_list = list(claims_dict.items())[:n]  # (claim_text, meta)

            def clean_text(x):
                x = str(x); x = x.split("/")[-1]
                return x.replace("_"," ").strip()

            def lin_unfiltered(walked):
                triples = []
                if isinstance(walked, dict):
                    triples = (walked.get("walkable", []) or []) + (walked.get("connected", []) or [])
                parts = []
                for t in triples:
                    if isinstance(t, (list,tuple)) and len(t)==3:
                        s,p,o = t
                        parts.append(f"{clean_text(s)} {clean_text(p)} {clean_text(o)}")
                return " [TRI] ".join(parts)

            unf_lens, unf_trunc = [], []
            for i, ((claim_text, _), (_, srow)) in enumerate(zip(claims_list, subs_df.iterrows())):
                subg_text = lin_unfiltered(srow["walked"])
                enc = tokenizer(
                    claim_text,
                    subg_text,
                    add_special_tokens=True,
                    truncation="only_second",
                    max_length=MAX_LEN,
                )
                L = len(enc["input_ids"])
                unf_lens.append(L)
                unf_trunc.append(int(L > MAX_LEN))
            print(f"[UNFILTERED first {n}] Avg tokens: {np.mean(unf_lens):.1f} | trunc %: {100*np.mean(unf_trunc):.1f}%")
            print(f"[FILTERED   all {len(lens)}] Avg tokens: {np.mean(lens):.1f} | trunc %: {100*np.mean(trunc):.1f}%")
        else:
            print("\n[WARN] Skipping unfiltered compare (missing subgraphs or claims pickle).")

    # --- Write a small CSV preview ---
    preview_csv = FILTERED_DIR / f"preview_{SPLIT}_k{K}.csv"
    sample = df.sample(n=min(args.preview_n, len(df)), random_state=7).copy() if len(df) else df.copy()
    sample["filtered_triples_str"] = sample["filtered_triples"].apply(
        lambda ts: " | ".join([f"{s} --{p}--> {o}" for (s,p,o) in ts])
    )
    cols = [c for c in ["claim","label","reasoning_type","num_original","num_filtered","approx_tokens","filtered_triples_str"] if c in sample.columns]
    sample[cols].to_csv(preview_csv, index=False)
    print(f"\nWrote preview CSV: {preview_csv}")

    print("\nAll sanity checks completed.")

if __name__ == "__main__":
    main()
