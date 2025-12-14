# #!/usr/bin/env python3
# """
# LLM-Filtered Single-Step Subgraphs for FACTKG
# ---------------------------------------------

# - Robust alignment by entity seed set (no order assumptions)
# - Deterministic LLM scoring (temperature=0, cached in SQLite)
# - Cheap prefilter (lexical + heuristics) -> LLM re-rank -> rank fusion
# - Outputs: pickle for training + JSONL audit trail
# """

# import os
# import re
# import json
# import time
# import hashlib
# import sqlite3
# import pickle
# from pathlib import Path
# from typing import List, Tuple, Dict

# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from openai import OpenAI

# # -------------------
# # Config (CLI overrides below)
# # -------------------
# DATA_DIR      = Path("data")
# SUBGRAPH_DIR  = DATA_DIR / "subgraphs"
# CACHE_DIR     = Path("cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
# RESULTS_DIR   = Path("results/llm_filtered"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# CACHE_DB      = CACHE_DIR / "triple_scores.db"

# # Default LLM config (can override via CLI)
# MODEL         = "gpt-4o-mini"
# TEMPERATURE   = 0.0
# SEED          = 42

# # Prefilter: # of triples to send to LLM after cheap screening
# M_PREFILTER   = 24

# # -------------------
# # Utilities
# # -------------------
# def norm_text(x: str) -> str:
#     """Normalize a KG string for lexical ops / prompting."""
#     x = str(x)
#     x = x.split("/")[-1]
#     return x.replace("_", " ").strip()

# def label_to_int(lbl):
#     """Normalize FACTKG labels to {0,1}."""
#     if isinstance(lbl, (list, tuple)) and len(lbl) == 1:
#         lbl = lbl[0]
#     s = str(lbl).lower()
#     if s in {"true", "1", "supports", "supported"}:
#         return 1
#     if s in {"false", "0", "refutes", "refuted"}:
#         return 0
#     return 1 if lbl in (True, 1) else 0

# def lexical_score(claim: str, triple: Tuple[str,str,str]) -> float:
#     """Simple Jaccard overlap between claim words and triple tokens."""
#     c = set(norm_text(claim).lower().split())
#     s, p, o = (norm_text(u).lower() for u in triple)
#     toks = set((s + " " + p + " " + o).split())
#     if not c or not toks:
#         return 0.0
#     inter = len(c & toks)
#     uni   = len(c | toks)
#     return inter / max(1, uni)

# def must_keep(claim: str, triple: Tuple[str,str,str]) -> bool:
#     """Heuristic 'hard keep' for obviously relevant predicates / entity mentions."""
#     c = norm_text(claim).lower()
#     s, p, o = (norm_text(u).lower() for u in triple)
#     hints = [
#         "born", "birth", "spouse", "wife", "husband",
#         "success", "predecess", "date", "death", "party",
#         "member", "author", "direct", "director", "team", "club"
#     ]
#     return any(h in p for h in hints) or s in c or o in c

# def _walked_signature(w):
#     """Build a stable, hashable surrogate for sorting subgraph rows."""
#     w = w or {}
#     walk = w.get("walkable", []) or []
#     conn = w.get("connected", []) or []
#     # counts
#     n_walk = len(walk)
#     n_conn = len(conn)
#     # first few predicate names (normalized) for tie-breaker
#     def pred_seq(triples, k=5):
#         preds = []
#         for t in triples[:k]:
#             if isinstance(t, (list, tuple)) and len(t) == 3:
#                 preds.append(str(t[1]).split("/")[-1].replace("_", " ").lower())
#         return "|".join(preds) if preds else ""
#     sig = (n_walk, n_conn, pred_seq(walk, 5), pred_seq(conn, 3))
#     return sig  # tuple is sortable


# def prefilter(claim: str, triples: List[Tuple[str,str,str]], M: int) -> List[Tuple[str,str,str]]:
#     """Cheap prefilter: keep must-keep + best lexical until M."""
#     keep = [t for t in triples if must_keep(claim, t)]
#     rest = sorted([t for t in triples if t not in keep],
#                   key=lambda t: lexical_score(claim, t), reverse=True)
#     return (keep + rest)[:M]

# def rank_fusion(claim: str, triples: List[Tuple[str,str,str]], llm_scores: List[float], alpha=0.7):
#     """Fuse ordinal ranks from LLM scores and lexical scores (1 = best)."""
#     llm_rank = {t: r+1 for r, (t, _) in enumerate(sorted(zip(triples, llm_scores),
#                                                          key=lambda x: -x[1]))}
#     lex_rank = {t: r+1 for r, (t, _) in enumerate(sorted([(t, lexical_score(claim, t)) for t in triples],
#                                                          key=lambda x: -x[1]))}
#     fused = [(t, alpha * llm_rank[t] + (1 - alpha) * lex_rank[t]) for t in triples]
#     return [t for t, _ in sorted(fused, key=lambda x: x[1])]

# # -------------------
# # LLM scoring + cache
# # -------------------

# # def cache_key(claim: str, s: str, p: str, o: str) -> str:
# #     return hashlib.sha1(f"{claim}||{s}||{p}||{o}".encode()).hexdigest()

# def cache_key(claim: str, s: str, p: str, o: str, model: str) -> str:
#     return hashlib.sha1(f"{model}||{claim}||{s}||{p}||{o}".encode()).hexdigest()

# def init_cache():
#     conn = sqlite3.connect(CACHE_DB)
#     # Speed up a bit, keep integrity
#     try:
#         conn.execute("PRAGMA journal_mode=WAL;")
#         conn.execute("PRAGMA synchronous=NORMAL;")
#     except Exception:
#         pass
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS scores (
#             cache_key TEXT PRIMARY KEY,
#             claim TEXT,
#             subject TEXT,
#             relation TEXT,
#             object TEXT,
#             llm_score REAL,
#             timestamp REAL
#         )
#     """)
#     conn.commit()
#     return conn

# def get_cached(conn, key: str):
#     row = conn.execute("SELECT llm_score FROM scores WHERE cache_key=?", (key,)).fetchone()
#     return None if row is None else float(row[0])

# def save_cache(conn, key, claim, s, p, o, score):
#     conn.execute("""INSERT OR REPLACE INTO scores
#         (cache_key, claim, subject, relation, object, llm_score, timestamp)
#         VALUES (?, ?, ?, ?, ?, ?, ?)""",
#         (key, claim, s, p, o, float(score), time.time()))
#     conn.commit()

# def llm_score_triple(client: OpenAI, claim: str, triple: Tuple[str,str,str],
#                      model=MODEL, temperature=TEMPERATURE, seed=SEED) -> float:
#     s, p, o = map(norm_text, triple)
#     prompt = f"""You are scoring knowledge-graph triples for RELEVANCE to a claim.
# Score each triple independently. 1.0 = directly helpful to verify/refute the claim;
# 0.0 = irrelevant or off-topic. Use only the semantics of the triple and the claim.

# Claim:
# {claim}

# Triple:
# subject = {s}
# relation = {p}
# object = {o}

# Return ONLY valid JSON: {{"score": <float 0..1>}}"""
#     for attempt in range(2):
#         resp = client.chat.completions.create(
#             model=model,
#             temperature=temperature,
#             seed=seed,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=20,
#         )
#         txt = resp.choices[0].message.content.strip()
#         # Strict JSON parse
#         try:
#             val = float(json.loads(txt)["score"])
#             return max(0.0, min(1.0, val))
#         except Exception:
#             m = re.search(r'(\d+\.?\d*)', txt)
#             if m:
#                 val = float(m.group(1))
#                 return max(0.0, min(1.0, val))
#         time.sleep(2 ** attempt)
#     return 0.0

# def score_triple(client: OpenAI, conn, claim: str, triple: Tuple[str,str,str],
#                  model=MODEL, temperature=TEMPERATURE, seed=SEED) -> float:
#     key = cache_key(claim, *triple, model=model) # <-- FIX
#     got = get_cached(conn, key)
#     if got is not None:
#         return got
#     val = llm_score_triple(client, claim, triple, model=model, temperature=temperature, seed=seed)
#     save_cache(conn, key, claim, *triple, val)
#     time.sleep(0.03)  # gentle pacing
#     return val

# def filter_subgraph(client: OpenAI, conn, claim: str, triples: List[Tuple[str,str,str]],
#                     k=10, M=M_PREFILTER, alpha=0.7,
#                     model=MODEL, temperature=TEMPERATURE, seed=SEED) -> List[Tuple[str,str,str]]:
#     if not triples:
#         return []
#     triples = [tuple(map(norm_text, t)) for t in triples if isinstance(t, (list, tuple)) and len(t) == 3]
#     pre = prefilter(claim, triples, M=M)
#     if not pre:
#         return []
#     llm_scores = [score_triple(client, conn, claim, t, model=model, temperature=temperature, seed=seed)
#                   for t in pre]
#     ranked = rank_fusion(claim, pre, llm_scores, alpha=alpha)
#     return ranked[:k]

# # -------------------
# # Alignment by entity set signature
# # -------------------
# def canonize_entities(ent_list: List[str]) -> str:
#     """Canonical signature string from a list of entity surface forms."""
#     ents = [str(e).split("/")[-1] for e in (ent_list or [])]
#     ents = [e.replace("_", " ").strip() for e in ents]
#     return " || ".join(sorted(set(ents)))

# def sig_from_subgraph_dict(subg_dict: Dict) -> str:
#     """Signature from the 'subgraph' dict keys (seed entities)."""
#     ents = list((subg_dict or {}).keys())
#     ents = [e.replace("_", " ").strip() for e in ents]
#     return " || ".join(sorted(set(ents)))

# def load_claims(split: str) -> pd.DataFrame:
#     p = DATA_DIR / f"factkg/factkg_{split}.pickle"
#     d = pickle.load(open(p, "rb"))
#     rows = []
#     for claim_text, meta in d.items():
#         rows.append({
#             "claim": claim_text,
#             "Label": meta.get("Label"),
#             "Entity_set": meta.get("Entity_set"),
#             "types": meta.get("types"),
#         })
#     df = pd.DataFrame(rows)
#     df["entity_sig"] = df["Entity_set"].apply(canonize_entities)
#     return df

# def load_subgraphs(split: str) -> pd.DataFrame:
#     p = SUBGRAPH_DIR / f"subgraphs_one_hop_{split}.pkl"
#     df = pd.read_pickle(p).copy()
#     # subgraph column is a dict: {seed_entity: [[...], ...]}
#     df["entity_sig"] = df["subgraph"].apply(sig_from_subgraph_dict)
#     return df

# def resolve_ambiguous_signature(sig: str, subs_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Deterministic positional pairing for a specific ambiguous entity signature.
#     Uses only sortable surrogate keys (no dicts) to avoid unhashable errors.
#     """
#     left  = subs_df[subs_df["entity_sig"] == sig].reset_index(drop=True).copy()
#     right = claims_df[claims_df["entity_sig"] == sig].reset_index(drop=True).copy()

#     # Build sortable surrogate columns
#     left["_n_walk"]  = left["walked"].apply(lambda w: len((w or {}).get("walkable", []) or []))
#     left["_n_conn"]  = left["walked"].apply(lambda w: len((w or {}).get("connected", []) or []))
#     left["_pred_sig"] = left["walked"].apply(_walked_signature)

#     # Sort left by (counts, predicate signature), right by claim text
#     left  = left.sort_values(by=["_n_walk", "_n_conn", "_pred_sig"], kind="mergesort").reset_index(drop=True)
#     right = right.sort_values(by=["claim"], kind="mergesort").reset_index(drop=True)

#     n = min(len(left), len(right))
#     out = []
#     for i in range(n):
#         out.append({
#             "claim":    right.loc[i, "claim"],
#             "Label":    right.loc[i, "Label"],
#             "types":    right.loc[i, "types"],
#             "subgraph": left.loc[i,  "subgraph"],
#             "walked":   left.loc[i,  "walked"],
#             "entity_sig": sig,
#         })
#     return pd.DataFrame(out)

# def align_by_entity_signature(subs_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Align subgraphs to claims by entity seed set signature.
#     Resolves many-to-many collisions deterministically using surrogates.
#     """
#     m = subs_df.merge(claims_df, on="entity_sig", how="inner", suffixes=("_sub", "_clm"))
#     # Count merged rows per signature; >1 implies many-to-many
#     counts = m.groupby("entity_sig").size()
#     ambiguous_sigs = set(counts[counts > 1].index)

#     # Unambiguous (appear exactly once on each side)
#     # We need true 1:1; compute multiplicities on each side separately
#     left_mult  = subs_df["entity_sig"].value_counts()
#     right_mult = claims_df["entity_sig"].value_counts()
#     one_to_one_sigs = {s for s,c in left_mult.items() if c == 1 and right_mult.get(s, 0) == 1}

#     clean_left  = subs_df[subs_df["entity_sig"].isin(one_to_one_sigs)]
#     clean_right = claims_df[claims_df["entity_sig"].isin(one_to_one_sigs)]
#     clean = clean_left.merge(clean_right, on="entity_sig", how="inner")
#     clean = clean[["claim", "Label", "types", "subgraph", "walked", "entity_sig"]]

#     # Resolve ambiguous signatures
#     resolved_chunks = []
#     for sig in sorted(ambiguous_sigs):
#         resolved_chunks.append(resolve_ambiguous_signature(sig, subs_df, claims_df))
#     resolved_df = pd.concat(resolved_chunks, ignore_index=True) if resolved_chunks else pd.DataFrame(
#         columns=["claim","Label","types","subgraph","walked","entity_sig"]
#     )

#     aligned = pd.concat([clean, resolved_df], ignore_index=True)

#     # Sanity note (we proceed even if sizes differ; user can inspect)
#     if len(aligned) != len(subs_df) or len(aligned) != len(claims_df):
#         print(f"[WARN] Alignment sizes differ (aligned={len(aligned)}, subgraphs={len(subs_df)}, claims={len(claims_df)}).")
#         # Optional: show a quick diff summary
#         # print("Unique sigs (subs only):", set(subs_df['entity_sig']) - set(aligned['entity_sig']))
#         # print("Unique sigs (claims only):", set(claims_df['entity_sig']) - set(aligned['entity_sig']))

#     return aligned

# # -------------------
# # Build filtered dataset
# # -------------------
# def build_filtered(split="test", k=10, M=M_PREFILTER, model=MODEL, temperature=TEMPERATURE, seed=SEED, limit=None):
#     print("=" * 80)
#     print(f"LLM-Filtered Subgraphs :: split={split}  k={k}  M={M}  model={model}  temp={temperature}  seed={seed}")
#     print("=" * 80)

#     client = OpenAI()
#     conn = init_cache()

#     # Load
#     print("Loading claims/subgraphs...")
#     claims_df = load_claims(split)
#     subs_df   = load_subgraphs(split)

#     print(f"Claims:   {len(claims_df)}")
#     print(f"Subgraphs:{len(subs_df)}")

#     # Align
#     aligned = align_by_entity_signature(subs_df, claims_df)
    
#     print(f"Aligned rows: {len(aligned)}")
#     # After align_by_entity_signature(...)
#     subs_sigs  = set(subs_df['entity_sig'].tolist())
#     claim_sigs = set(claims_df['entity_sig'].tolist())
#     missing_in_subs  = sorted(list(claim_sigs - subs_sigs))[:5]
#     missing_in_claim = sorted(list(subs_sigs - claim_sigs))[:5]
#     print(f"Unmatched claim signatures: {len(claim_sigs - subs_sigs)} (showing up to 5):", missing_in_subs)
#     print(f"Unmatched subgraph signatures: {len(subs_sigs - claim_sigs)} (showing up to 5):", missing_in_claim)

#     if limit is not None:
#         aligned = aligned.iloc[:limit].copy()
#         print(f"[DEBUG] Limiting to first {len(aligned)} rows")

#     # Iterate & filter
#     records = []
#     # --- ADD THIS FIX ---
#     # Sanitize model name for filename (e.g., 'gpt-4o-mini' -> 'gpt-4o-mini')
#     model_name_safe = model.replace('/', '_').replace(':', '_') 
#     jsonl_path = RESULTS_DIR / f"filtered_{split}_k{k}_{model_name_safe}.jsonl"
#     out_pkl    = RESULTS_DIR / f"filtered_{split}_k{k}_{model_name_safe}.pkl"
#     # --- END FIX ---
#     # jsonl_path = RESULTS_DIR / f"filtered_{split}_k{k}.jsonl"
#     # out_pkl    = RESULTS_DIR / f"filtered_{split}_k{k}.pkl"

#     kept_count = 0
#     orig_total = 0

#     with jsonl_path.open("w") as jf:
#         for _, row in tqdm(aligned.iterrows(), total=len(aligned)):
#             claim_text = row["claim"]
#             walked = row.get("walked") or {}
#             triples = []
#             # Collect triples from both walkable and connected
#             for t in (walked.get("walkable", []) + walked.get("connected", [])):
#                 if isinstance(t, (list, tuple)) and len(t) == 3:
#                     triples.append(tuple(t))

#             # Filter
#             kept = filter_subgraph(client, conn, claim_text, triples, k=k, M=M,
#                                    model=model, temperature=temperature, seed=seed)

#             rec = {
#                 "claim": claim_text,
#                 "label": label_to_int(row.get("Label")),
#                 "reasoning_type": (row.get("types") or ["unknown"])[-1],
#                 "original_triples": triples,
#                 "filtered_triples": kept,
#                 "num_original": len(triples),
#                 "num_filtered": len(kept),
#                 # cheap token estimate: claim ~19, overhead ~3, ~13 tokens/triple
#                 "approx_tokens": int(len(claim_text.split()) + 3 + 13 * len(kept)),
#                 "entity_sig": row.get("entity_sig", ""),
#             }
#             records.append(rec)
#             jf.write(json.dumps({
#                 "claim": rec["claim"],
#                 "label": rec["label"],
#                 "filtered_triples": rec["filtered_triples"],
#                 "k": k
#             }) + "\n")

#             kept_count += len(kept)
#             orig_total += len(triples)

#     # Save pickle
#     pd.DataFrame(records).to_pickle(out_pkl)
#     print(f"Wrote: {out_pkl}")
#     print(f"Also wrote: {jsonl_path}")

#     # Stats
#     df = pd.DataFrame(records)
#     mean_orig = df["num_original"].replace(0, np.nan).mean()
#     reduction = (1 - df["num_filtered"].mean() / mean_orig) * 100 if pd.notna(mean_orig) else 0.0
#     print(f"Avg original triples: {df['num_original'].mean():.1f}")
#     print(f"Avg filtered triples: {df['num_filtered'].mean():.1f}")
#     print(f"Reduction: {reduction:.1f}%")
#     print(f"Approx avg tokens (after filter): {df['approx_tokens'].mean():.1f}")

#     conn.close()
#     return out_pkl, jsonl_path

# # -------------------
# # CLI
# # -------------------
# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--split", choices=["train", "dev", "test"], default="test")
#     ap.add_argument("--k", type=int, default=10, help="Top-k triples to keep")
#     ap.add_argument("--M", type=int, default=M_PREFILTER, help="Prefilter size (triples to send to LLM)")
#     ap.add_argument("--model", type=str, default=MODEL)
#     ap.add_argument("--temperature", type=float, default=TEMPERATURE)
#     ap.add_argument("--seed", type=int, default=SEED)
#     ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug)")
#     args = ap.parse_args()

#     # Allow overrides via env vars (optional)
#     model = os.environ.get("LLM_MODEL", args.model)
#     temperature = float(os.environ.get("LLM_TEMPERATURE", args.temperature))
#     seed = int(os.environ.get("LLM_SEED", args.seed))

#     build_filtered(
#         split=args.split,
#         k=args.k,
#         M=args.M,
#         model=model,
#         temperature=temperature,
#         seed=seed,
#         limit=args.limit
#     )




# #!/usr/bin/env python3
# """
# LLM-Filtered Single-Step Subgraphs for FACTKG
# ---------------------------------------------

# - Robust alignment by entity seed set (no order assumptions)
# - Deterministic LLM scoring (temperature=0, cached in SQLite)
# - Cheap prefilter (lexical + heuristics) -> LLM re-rank -> rank fusion
# - Outputs: pickle for training + JSONL audit trail
# """

# import os
# import re
# import json
# import time
# import hashlib
# import sqlite3
# import pickle
# from pathlib import Path
# from typing import List, Tuple, Dict

# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from openai import OpenAI

# # -------------------
# # Config (CLI overrides below)
# # -------------------
# DATA_DIR = Path("data")
# SUBGRAPH_DIR = DATA_DIR / "subgraphs"
# CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
# RESULTS_DIR = Path("results/llm_filtered"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# CACHE_DB = CACHE_DIR / "triple_scores.db"

# # Default LLM config (can override via CLI)
# MODEL = "gpt-4o-mini"
# TEMPERATURE = 0.0
# SEED = 42

# # Prefilter: # of triples to send to LLM after cheap screening
# M_PREFILTER = 24

# # -------------------
# # Utilities
# # -------------------
# def norm_text(x: str) -> str:
#     """Normalize a KG string for lexical ops / prompting."""
#     x = str(x)
#     x = x.split("/")[-1]
#     return x.replace("_", " ").strip()

# def label_to_int(lbl):
#     """Normalize FACTKG labels to {0,1}."""
#     if isinstance(lbl, (list, tuple)) and len(lbl) == 1:
#         lbl = lbl[0]
#     s = str(lbl).lower()
#     if s in {"true", "1", "supports", "supported"}:
#         return 1
#     if s in {"false", "0", "refutes", "refuted"}:
#         return 0
#     return 1 if lbl in (True, 1) else 0

# def lexical_score(claim: str, triple: Tuple[str,str,str]) -> float:
#     """Simple Jaccard overlap between claim words and triple tokens."""
#     c = set(norm_text(claim).lower().split())
#     s, p, o = (norm_text(u).lower() for u in triple)
#     toks = set((s + " " + p + " " + o).split())
#     if not c or not toks:
#         return 0.0
#     inter = len(c & toks)
#     uni = len(c | toks)
#     return inter / max(1, uni)

# def must_keep(claim: str, triple: Tuple[str,str,str]) -> bool:
#     """Heuristic 'hard keep' for obviously relevant predicates / entity mentions."""
#     c = norm_text(claim).lower()
#     s, p, o = (norm_text(u).lower() for u in triple)
#     hints = [
#         "born", "birth", "spouse", "wife", "husband",
#         "success", "predecess", "date", "death", "party",
#         "member", "author", "direct", "director", "team", "club"
#     ]
#     return any(h in p for h in hints) or s in c or o in c

# def _walked_signature(w):
#     """Build a stable, hashable surrogate for sorting subgraph rows."""
#     w = w or {}
#     walk = w.get("walkable", []) or []
#     conn = w.get("connected", []) or []
#     # counts
#     n_walk = len(walk)
#     n_conn = len(conn)
#     # first few predicate names (normalized) for tie-breaker
#     def pred_seq(triples, k=5):
#         preds = []
#         for t in triples[:k]:
#             if isinstance(t, (list, tuple)) and len(t) == 3:
#                 preds.append(str(t[1]).split("/")[-1].replace("_", " ").lower())
#         return "|".join(preds) if preds else ""
#     sig = (n_walk, n_conn, pred_seq(walk, 5), pred_seq(conn, 3))
#     return sig  # tuple is sortable


# def prefilter(claim: str, triples: List[Tuple[str,str,str]], M: int) -> List[Tuple[str,str,str]]:
#     """Cheap prefilter: keep must-keep + best lexical until M."""
#     keep = [t for t in triples if must_keep(claim, t)]
#     rest = sorted([t for t in triples if t not in keep],
#                   key=lambda t: lexical_score(claim, t), reverse=True)
#     return (keep + rest)[:M]

# def rank_fusion(claim: str, triples: List[Tuple[str,str,str]], llm_scores: List[float], alpha=0.7):
#     """Fuse ordinal ranks from LLM scores and lexical scores (1 = best)."""
#     llm_rank = {t: r+1 for r, (t, _) in enumerate(sorted(zip(triples, llm_scores),
#                                                          key=lambda x: -x[1]))}
#     lex_rank = {t: r+1 for r, (t, _) in enumerate(sorted([(t, lexical_score(claim, t)) for t in triples],
#                                                          key=lambda x: -x[1]))}
#     fused = [(t, alpha * llm_rank[t] + (1 - alpha) * lex_rank[t]) for t in triples]
#     return [t for t, _ in sorted(fused, key=lambda x: x[1])]

# # -------------------
# # LLM scoring + cache
# # -------------------

# def cache_key(claim: str, s: str, p: str, o: str, model: str) -> str:
#     return hashlib.sha1(f"{model}||{claim}||{s}||{p}||{o}".encode()).hexdigest()

# def init_cache():
#     conn = sqlite3.connect(CACHE_DB)
#     # Speed up a bit, keep integrity
#     try:
#         conn.execute("PRAGMA journal_mode=WAL;")
#         conn.execute("PRAGMA synchronous=NORMAL;")
#     except Exception:
#         pass
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS scores (
#             cache_key TEXT PRIMARY KEY,
#             claim TEXT,
#             subject TEXT,
#             relation TEXT,
#             object TEXT,
#             llm_score REAL,
#             timestamp REAL
#         )
#     """)
#     conn.commit()
#     return conn

# def get_cached(conn, key: str):
#     row = conn.execute("SELECT llm_score FROM scores WHERE cache_key=?", (key,)).fetchone()
#     return None if row is None else float(row[0])

# def save_cache(conn, key, claim, s, p, o, score):
#     conn.execute("""INSERT OR REPLACE INTO scores
#         (cache_key, claim, subject, relation, object, llm_score, timestamp)
#         VALUES (?, ?, ?, ?, ?, ?, ?)""",
#         (key, claim, s, p, o, float(score), time.time()))
#     conn.commit()

# def llm_score_triple(client: OpenAI, claim: str, triple: Tuple[str,str,str],
#                      model=MODEL, temperature=TEMPERATURE, seed=SEED) -> float:
#     s, p, o = map(norm_text, triple)
#     prompt = f"""You are scoring knowledge-graph triples for RELEVANCE to a claim.
# Score each triple independently. 1.0 = directly helpful to verify/refute the claim;
# 0.0 = irrelevant or off-topic. Use only the semantics of the triple and the claim.

# Claim:
# {claim}

# Triple:
# subject = {s}
# relation = {p}
# object = {o}

# Return ONLY valid JSON: {{"score": <float 0..1>}}"""
#     for attempt in range(2):
#         resp = client.chat.completions.create(
#             model=model,
#             temperature=temperature,
#             seed=seed,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=20,
#         )
#         txt = resp.choices[0].message.content.strip()
#         # Strict JSON parse
#         try:
#             val = float(json.loads(txt)["score"])
#             return max(0.0, min(1.0, val))
#         except Exception:
#             m = re.search(r'(\d+\.?\d*)', txt)
#             if m:
#                 val = float(m.group(1))
#                 return max(0.0, min(1.0, val))
#         time.sleep(2 ** attempt)
#     return 0.0

# def score_triple(client: OpenAI, conn, claim: str, triple: Tuple[str,str,str],
#                  model=MODEL, temperature=TEMPERATURE, seed=SEED) -> float:
#     key = cache_key(claim, *triple, model=model)
#     got = get_cached(conn, key)
#     if got is not None:
#         return got
#     val = llm_score_triple(client, claim, triple, model=model, temperature=temperature, seed=seed)
#     save_cache(conn, key, claim, *triple, val)
#     time.sleep(0.03)  # gentle pacing
#     return val

# def filter_subgraph(client: OpenAI, conn, claim: str, triples: List[Tuple[str,str,str]],
#                     k=10, M=M_PREFILTER, alpha=0.7,
#                     model=MODEL, temperature=TEMPERATURE, seed=SEED) -> List[Tuple[str,str,str]]:
#     if not triples:
#         return []
#     triples = [tuple(map(norm_text, t)) for t in triples if isinstance(t, (list, tuple)) and len(t) == 3]
#     pre = prefilter(claim, triples, M=M)
#     if not pre:
#         return []
#     llm_scores = [score_triple(client, conn, claim, t, model=model, temperature=temperature, seed=seed)
#                   for t in pre]
#     ranked = rank_fusion(claim, pre, llm_scores, alpha=alpha)
#     return ranked[:k]

# # -------------------
# # Alignment by entity set signature
# # -------------------
# def canonize_entities(ent_list: List[str]) -> str:
#     """Canonical signature string from a list of entity surface forms."""
#     ents = [str(e).split("/")[-1] for e in (ent_list or [])]
#     ents = [e.replace("_", " ").strip() for e in ents]
#     return " || ".join(sorted(set(ents)))

# def sig_from_subgraph_dict(subg_dict: Dict) -> str:
#     """Signature from the 'subgraph' dict keys (seed entities)."""
#     ents = list((subg_dict or {}).keys())
#     ents = [e.replace("_", " ").strip() for e in ents]
#     return " || ".join(sorted(set(ents)))

# def load_claims(split: str) -> pd.DataFrame:
#     p = DATA_DIR / f"factkg/factkg_{split}.pickle"
#     d = pickle.load(open(p, "rb"))
#     rows = []
#     for claim_text, meta in d.items():
#         rows.append({
#             "claim": claim_text,
#             "Label": meta.get("Label"),
#             "Entity_set": meta.get("Entity_set"),
#             "types": meta.get("types"),
#         })
#     df = pd.DataFrame(rows)
#     df["entity_sig"] = df["Entity_set"].apply(canonize_entities)
#     return df

# def load_subgraphs(split: str) -> pd.DataFrame:
#     p = SUBGRAPH_DIR / f"subgraphs_one_hop_{split}.pkl"
#     df = pd.read_pickle(p).copy()
#     # subgraph column is a dict: {seed_entity: [[...], ...]}
#     df["entity_sig"] = df["subgraph"].apply(sig_from_subgraph_dict)
#     return df

# def resolve_ambiguous_signature(sig: str, subs_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Deterministic positional pairing for a specific ambiguous entity signature.
#     Uses only sortable surrogate keys (no dicts) to avoid unhashable errors.
#     """
#     left = subs_df[subs_df["entity_sig"] == sig].reset_index(drop=True).copy()
#     right = claims_df[claims_df["entity_sig"] == sig].reset_index(drop=True).copy()

#     # Build sortable surrogate columns
#     left["_n_walk"] = left["walked"].apply(lambda w: len((w or {}).get("walkable", []) or []))
#     left["_n_conn"] = left["walked"].apply(lambda w: len((w or {}).get("connected", []) or []))
#     left["_pred_sig"] = left["walked"].apply(_walked_signature)

#     # Sort left by (counts, predicate signature), right by claim text
#     left = left.sort_values(by=["_n_walk", "_n_conn", "_pred_sig"], kind="mergesort").reset_index(drop=True)
#     right = right.sort_values(by=["claim"], kind="mergesort").reset_index(drop=True)

#     n = min(len(left), len(right))
#     out = []
#     for i in range(n):
#         out.append({
#             "claim": right.loc[i, "claim"],
#             "Label": right.loc[i, "Label"],
#             "types": right.loc[i, "types"],
#             "subgraph": left.loc[i, "subgraph"],
#             "walked": left.loc[i, "walked"],
#             "entity_sig": sig,
#         })
#     return pd.DataFrame(out)

# def align_by_entity_signature(subs_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Align subgraphs to claims by entity seed set signature.
#     Resolves many-to-many collisions deterministically using surrogates.
#     """
#     m = subs_df.merge(claims_df, on="entity_sig", how="inner", suffixes=("_sub", "_clm"))
#     # Count merged rows per signature; >1 implies many-to-many
#     counts = m.groupby("entity_sig").size()
#     ambiguous_sigs = set(counts[counts > 1].index)

#     # Unambiguous (appear exactly once on each side)
#     # We need true 1:1; compute multiplicities on each side separately
#     left_mult = subs_df["entity_sig"].value_counts()
#     right_mult = claims_df["entity_sig"].value_counts()
#     one_to_one_sigs = {s for s,c in left_mult.items() if c == 1 and right_mult.get(s, 0) == 1}

#     clean_left = subs_df[subs_df["entity_sig"].isin(one_to_one_sigs)]
#     clean_right = claims_df[claims_df["entity_sig"].isin(one_to_one_sigs)]
#     clean = clean_left.merge(clean_right, on="entity_sig", how="inner")
#     clean = clean[["claim", "Label", "types", "subgraph", "walked", "entity_sig"]]

#     # Resolve ambiguous signatures
#     resolved_chunks = []
#     for sig in sorted(ambiguous_sigs):
#         resolved_chunks.append(resolve_ambiguous_signature(sig, subs_df, claims_df))
#     resolved_df = pd.concat(resolved_chunks, ignore_index=True) if resolved_chunks else pd.DataFrame(
#         columns=["claim","Label","types","subgraph","walked","entity_sig"]
#     )

#     aligned = pd.concat([clean, resolved_df], ignore_index=True)

#     # Sanity note (we proceed even if sizes differ; user can inspect)
#     if len(aligned) != len(subs_df) or len(aligned) != len(claims_df):
#         print(f"[WARN] Alignment sizes differ (aligned={len(aligned)}, subgraphs={len(subs_df)}, claims={len(claims_df)}).")
#         # Optional: show a quick diff summary
#         # print("Unique sigs (subs only):", set(subs_df['entity_sig']) - set(aligned['entity_sig']))
#         # print("Unique sigs (claims only):", set(claims_df['entity_sig']) - set(aligned['entity_sig']))

#     return aligned

# # -------------------
# # Build filtered dataset
# # -------------------
# def build_filtered(split="test", k=10, M=M_PREFILTER, model=MODEL, temperature=TEMPERATURE, seed=SEED, limit=None):
#     print("=" * 80)
#     print(f"LLM-Filtered Subgraphs :: split={split} k={k} M={M} model={model} temp={temperature} seed={seed}")
#     print("=" * 80)

#     client = OpenAI()
#     conn = init_cache()

#     # Load
#     print("Loading claims/subgraphs...")
#     claims_df = load_claims(split)
#     subs_df = load_subgraphs(split)

#     print(f"Claims: {len(claims_df)}")
#     print(f"Subgraphs:{len(subs_df)}")

#     # Align
#     aligned = align_by_entity_signature(subs_df, claims_df)
    
#     print(f"Aligned rows: {len(aligned)}")
#     # After align_by_entity_signature(...)
#     subs_sigs = set(subs_df['entity_sig'].tolist())
#     claim_sigs = set(claims_df['entity_sig'].tolist())
#     missing_in_subs = sorted(list(claim_sigs - subs_sigs))[:5]
#     missing_in_claim = sorted(list(subs_sigs - claim_sigs))[:5]
#     print(f"Unmatched claim signatures: {len(claim_sigs - subs_sigs)} (showing up to 5):", missing_in_subs)
#     print(f"Unmatched subgraph signatures: {len(subs_sigs - claim_sigs)} (showing up to 5):", missing_in_claim)

#     if limit is not None:
#         aligned = aligned.iloc[:limit].copy()
#         print(f"[DEBUG] Limiting to first {len(aligned)} rows")

#     # Iterate & filter
#     records = []
#     # Sanitize model name for filename (e.g., 'gpt-4o-mini' -> 'gpt-4o-mini')
#     model_name_safe = model.replace('/', '_').replace(':', '_')
#     jsonl_path = RESULTS_DIR / f"filtered_{split}_k{k}_{model_name_safe}.jsonl"
#     out_pkl = RESULTS_DIR / f"filtered_{split}_k{k}_{model_name_safe}.pkl"

#     kept_count = 0
#     orig_total = 0

#     with jsonl_path.open("w") as jf:
#         for _, row in tqdm(aligned.iterrows(), total=len(aligned)):
#             claim_text = row["claim"]
#             walked = row.get("walked") or {}
#             triples = []
#             # Collect triples from both walkable and connected
#             for t in (walked.get("walkable", []) + walked.get("connected", [])):
#                 if isinstance(t, (list, tuple)) and len(t) == 3:
#                     triples.append(tuple(t))

#             # Filter
#             kept = filter_subgraph(client, conn, claim_text, triples, k=k, M=M,
#                                    model=model, temperature=temperature, seed=seed)

#             rec = {
#                 "claim": claim_text,
#                 "label": label_to_int(row.get("Label")),
#                 "reasoning_type": (row.get("types") or ["unknown"])[-1],
#                 "original_triples": triples,
#                 "filtered_triples": kept,
#                 "num_original": len(triples),
#                 "num_filtered": len(kept),
#                 # cheap token estimate: claim ~19, overhead ~3, ~13 tokens/triple
#                 "approx_tokens": int(len(claim_text.split()) + 3 + 13 * len(kept)),
#                 "entity_sig": row.get("entity_sig", ""),
#             }
#             records.append(rec)
#             jf.write(json.dumps({
#                 "claim": rec["claim"],
#                 "label": rec["label"],
#                 "filtered_triples": rec["filtered_triples"],
#                 "k": k
#             }) + "\n")

#             kept_count += len(kept)
#             orig_total += len(triples)

#     # Save pickle
#     pd.DataFrame(records).to_pickle(out_pkl)
#     print(f"Wrote: {out_pkl}")
#     print(f"Also wrote: {jsonl_path}")

#     # Stats
#     df = pd.DataFrame(records)
#     mean_orig = df["num_original"].replace(0, np.nan).mean()
#     reduction = (1 - df["num_filtered"].mean() / mean_orig) * 100 if pd.notna(mean_orig) else 0.0
#     print(f"Avg original triples: {df['num_original'].mean():.1f}")
#     print(f"Avg filtered triples: {df['num_filtered'].mean():.1f}")
#     print(f"Reduction: {reduction:.1f}%")
#     print(f"Approx avg tokens (after filter): {df['approx_tokens'].mean():.1f}")

#     conn.close()
#     return out_pkl, jsonl_path

# # -------------------
# # CLI
# # -------------------
# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--split", choices=["train", "dev", "test"], default="test")
#     ap.add_argument("--k", type=int, default=10, help="Top-k triples to keep")
#     ap.add_argument("--M", type=int, default=M_PREFILTER, help="Prefilter size (triples to send to LLM)")
#     ap.add_argument("--model", type=str, default=MODEL)
#     ap.add_argument("--temperature", type=float, default=TEMPERATURE)
#     ap.add_argument("--seed", type=int, default=SEED)
#     ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug)")
#     args = ap.parse_args()

#     # Allow overrides via env vars (optional)
#     model = os.environ.get("LLM_MODEL", args.model)
#     temperature = float(os.environ.get("LLM_TEMPERATURE", args.temperature))
#     seed = int(os.environ.get("LLM_SEED", args.seed))

#     build_filtered(
#         split=args.split,
#         k=args.k,
#         M=args.M,
#         model=model,
#         temperature=temperature,
#         seed=seed,
#         limit=args.limit
#     )


#!/usr/bin/env python3
"""
LLM-Filtered Single-Step Subgraphs for FACTKG
---------------------------------------------

- Robust alignment by entity seed set (no order assumptions)
- Deterministic LLM scoring (temperature=0, cached in SQLite)
- Cheap prefilter (lexical + heuristics) -> LLM re-rank -> rank fusion
- Outputs: pickle for training + JSONL audit trail
"""

import os
import re
import json
import time
import hashlib
import sqlite3
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# -------------------
# Config (CLI overrides below)
# -------------------
DATA_DIR = Path("data")
SUBGRAPH_DIR = DATA_DIR / "subgraphs"
CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("results/llm_filtered"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / "triple_scores.db"

# Default LLM config (can override via CLI)
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
SEED = 42

# Prefilter: # of triples to send to LLM after cheap screening
M_PREFILTER = 24

# -------------------
# Utilities
# -------------------
def norm_text(x: str) -> str:
    """Normalize a KG string for lexical ops / prompting."""
    x = str(x)
    x = x.split("/")[-1]
    return x.replace("_", " ").strip()

def label_to_int(lbl):
    """Normalize FACTKG labels to {0,1}."""
    if isinstance(lbl, (list, tuple)) and len(lbl) == 1:
        lbl = lbl[0]
    s = str(lbl).lower()
    if s in {"true", "1", "supports", "supported"}:
        return 1
    if s in {"false", "0", "refutes", "refuted"}:
        return 0
    return 1 if lbl in (True, 1) else 0

def lexical_score(claim: str, triple: Tuple[str,str,str]) -> float:
    """Simple Jaccard overlap between claim words and triple tokens."""
    c = set(norm_text(claim).lower().split())
    s, p, o = (norm_text(u).lower() for u in triple)
    toks = set((s + " " + p + " " + o).split())
    if not c or not toks:
        return 0.0
    inter = len(c & toks)
    uni = len(c | toks)
    return inter / max(1, uni)

def must_keep(claim: str, triple: Tuple[str,str,str]) -> bool:
    """Heuristic 'hard keep' for obviously relevant predicates / entity mentions."""
    c = norm_text(claim).lower()
    s, p, o = (norm_text(u).lower() for u in triple)
    hints = [
        "born", "birth", "spouse", "wife", "husband",
        "success", "predecess", "date", "death", "party",
        "member", "author", "direct", "director", "team", "club"
    ]
    return any(h in p for h in hints) or s in c or o in c

def _walked_signature(w):
    """Build a stable, hashable surrogate for sorting subgraph rows."""
    w = w or {}
    walk = w.get("walkable", []) or []
    conn = w.get("connected", []) or []
    # counts
    n_walk = len(walk)
    n_conn = len(conn)
    # first few predicate names (normalized) for tie-breaker
    def pred_seq(triples, k=5):
        preds = []
        for t in triples[:k]:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                preds.append(str(t[1]).split("/")[-1].replace("_", " ").lower())
        return "|".join(preds) if preds else ""
    sig = (n_walk, n_conn, pred_seq(walk, 5), pred_seq(conn, 3))
    return sig  # tuple is sortable


def prefilter(claim: str, triples: List[Tuple[str,str,str]], M: int) -> List[Tuple[str,str,str]]:
    """Cheap prefilter: keep must-keep + best lexical until M."""
    keep = [t for t in triples if must_keep(claim, t)]
    rest = sorted([t for t in triples if t not in keep],
                  key=lambda t: lexical_score(claim, t), reverse=True)
    return (keep + rest)[:M]

def rank_fusion(claim: str, triples: List[Tuple[str,str,str]], llm_scores: List[float], alpha=0.7):
    """Fuse ordinal ranks from LLM scores and lexical scores (1 = best)."""
    llm_rank = {t: r+1 for r, (t, _) in enumerate(sorted(zip(triples, llm_scores),
                                                        key=lambda x: -x[1]))}
    lex_rank = {t: r+1 for r, (t, _) in enumerate(sorted([(t, lexical_score(claim, t)) for t in triples],
                                                        key=lambda x: -x[1]))}
    fused = [(t, alpha * llm_rank[t] + (1 - alpha) * lex_rank[t]) for t in triples]
    return [t for t, _ in sorted(fused, key=lambda x: x[1])]

# -------------------
# LLM scoring + cache
# -------------------

def cache_key(claim: str, s: str, p: str, o: str, model: str) -> str:
    return hashlib.sha1(f"{model}||{claim}||{s}||{p}||{o}".encode()).hexdigest()

def init_cache(split: str):
    # --- 2. MAKE FILENAME UNIQUE ---
    cache_db_path = CACHE_DIR / f"triple_scores_{split}.db"
    conn = sqlite3.connect(cache_db_path)
    #conn = sqlite3.connect(CACHE_DB)
    # Speed up a bit, keep integrity
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            cache_key TEXT PRIMARY KEY,
            claim TEXT,
            subject TEXT,
            relation TEXT,
            object TEXT,
            llm_score REAL,
            timestamp REAL
        )
    """)
    conn.commit()
    return conn

def get_cached(conn, key: str):
    row = conn.execute("SELECT llm_score FROM scores WHERE cache_key=?", (key,)).fetchone()
    return None if row is None else float(row[0])

def save_cache(conn, key, claim, s, p, o, score):
    conn.execute("""INSERT OR REPLACE INTO scores
        (cache_key, claim, subject, relation, object, llm_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (key, claim, s, p, o, float(score), time.time()))
    conn.commit()

def llm_score_triple(client: OpenAI, claim: str, triple: Tuple[str,str,str],
                     model=MODEL, temperature=TEMPERATURE, seed=SEED) -> float:
    s, p, o = map(norm_text, triple)
    prompt = f"""You are scoring knowledge-graph triples for RELEVANCE to a claim.
Score each triple independently. 1.0 = directly helpful to verify/refute the claim;
0.0 = irrelevant or off-topic. Use only the semantics of the triple and the claim.

Claim:
{claim}

Triple:
subject = {s}
relation = {p}
object = {o}

Return ONLY valid JSON: {{"score": <float 0..1>}}"""
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            seed=seed,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        )
        txt = resp.choices[0].message.content.strip()
        # Strict JSON parse
        try:
            val = float(json.loads(txt)["score"])
            return max(0.0, min(1.0, val))
        except Exception:
            m = re.search(r'(\d+\.?\d*)', txt)
            if m:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
        time.sleep(2 ** attempt)
    return 0.0

def score_triple(client: OpenAI, conn, claim: str, triple: Tuple[str,str,str],
                 model=MODEL, temperature=TEMPERATURE, seed=SEED) -> float:
    key = cache_key(claim, *triple, model=model)
    got = get_cached(conn, key)
    if got is not None:
        return got
    val = llm_score_triple(client, claim, triple, model=model, temperature=temperature, seed=seed)
    save_cache(conn, key, claim, *triple, val)
    time.sleep(0.03)  # gentle pacing
    return val

def filter_subgraph(client: OpenAI, conn, claim: str, triples: List[Tuple[str,str,str]],
                     k=10, M=M_PREFILTER, alpha=0.7,
                     model=MODEL, temperature=TEMPERATURE, seed=SEED) -> List[Tuple[str,str,str]]:
    if not triples:
        return []
    triples = [tuple(map(norm_text, t)) for t in triples if isinstance(t, (list, tuple)) and len(t) == 3]
    pre = prefilter(claim, triples, M=M)
    if not pre:
        return []
    llm_scores = [score_triple(client, conn, claim, t, model=model, temperature=temperature, seed=seed)
                  for t in pre]
    ranked = rank_fusion(claim, pre, llm_scores, alpha=alpha)
    return ranked[:k]

# -------------------
# Alignment by entity set signature
# -------------------
def canonize_entities(ent_list: List[str]) -> str:
    """Canonical signature string from a list of entity surface forms."""
    ents = [str(e).split("/")[-1] for e in (ent_list or [])]
    ents = [e.replace("_", " ").strip() for e in ents]
    return " || ".join(sorted(set(ents)))

def sig_from_subgraph_dict(subg_dict: Dict) -> str:
    """Signature from the 'subgraph' dict keys (seed entities)."""
    ents = list((subg_dict or {}).keys())
    ents = [e.replace("_", " ").strip() for e in ents]
    return " || ".join(sorted(set(ents)))

def load_claims(split: str) -> pd.DataFrame:
    if split == 'train':
        p = DATA_DIR / "factkg/factkg_train_10k_official_sample.pickle"
    else:
        p = DATA_DIR / f"factkg/factkg_{split}.pickle"
    d = pickle.load(open(p, "rb"))
    rows = []
    for claim_text, meta in d.items():
        rows.append({
            "claim": claim_text,
            "Label": meta.get("Label"),
            "Entity_set": meta.get("Entity_set"),
            "types": meta.get("types"),
        })
    df = pd.DataFrame(rows)
    df["entity_sig"] = df["Entity_set"].apply(canonize_entities)
    return df

def load_subgraphs(split: str) -> pd.DataFrame:
    p = SUBGRAPH_DIR / f"subgraphs_one_hop_{split}.pkl"
    df = pd.read_pickle(p).copy()
    # subgraph column is a dict: {seed_entity: [[...], ...]}
    df["entity_sig"] = df["subgraph"].apply(sig_from_subgraph_dict)
    return df

def resolve_ambiguous_signature(sig: str, subs_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic positional pairing for a specific ambiguous entity signature.
    Uses only sortable surrogate keys (no dicts) to avoid unhashable errors.
    """
    left = subs_df[subs_df["entity_sig"] == sig].reset_index(drop=True).copy()
    right = claims_df[claims_df["entity_sig"] == sig].reset_index(drop=True).copy()

    # Build sortable surrogate columns
    left["_n_walk"] = left["walked"].apply(lambda w: len((w or {}).get("walkable", []) or []))
    left["_n_conn"] = left["walked"].apply(lambda w: len((w or {}).get("connected", []) or []))
    left["_pred_sig"] = left["walked"].apply(_walked_signature)

    # Sort left by (counts, predicate signature), right by claim text
    left = left.sort_values(by=["_n_walk", "_n_conn", "_pred_sig"], kind="mergesort").reset_index(drop=True)
    right = right.sort_values(by=["claim"], kind="mergesort").reset_index(drop=True)

    n = min(len(left), len(right))
    out = []
    for i in range(n):
        out.append({
            "claim": right.loc[i, "claim"],
            "Label": right.loc[i, "Label"],
            "types": right.loc[i, "types"],
            "subgraph": left.loc[i, "subgraph"],
            "walked": left.loc[i, "walked"],
            "entity_sig": sig,
        })
    return pd.DataFrame(out)

def align_by_entity_signature(subs_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align subgraphs to claims by entity seed set signature.
    Resolves many-to-many collisions deterministically using surrogates.
    """
    m = subs_df.merge(claims_df, on="entity_sig", how="inner", suffixes=("_sub", "_clm"))
    # Count merged rows per signature; >1 implies many-to-many
    counts = m.groupby("entity_sig").size()
    ambiguous_sigs = set(counts[counts > 1].index)

    # Unambiguous (appear exactly once on each side)
    # We need true 1:1; compute multiplicities on each side separately
    left_mult = subs_df["entity_sig"].value_counts()
    right_mult = claims_df["entity_sig"].value_counts()
    one_to_one_sigs = {s for s,c in left_mult.items() if c == 1 and right_mult.get(s, 0) == 1}

    clean_left = subs_df[subs_df["entity_sig"].isin(one_to_one_sigs)]
    clean_right = claims_df[claims_df["entity_sig"].isin(one_to_one_sigs)]
    clean = clean_left.merge(clean_right, on="entity_sig", how="inner")
    clean = clean[["claim", "Label", "types", "subgraph", "walked", "entity_sig"]]

    # Resolve ambiguous signatures
    resolved_chunks = []
    for sig in sorted(ambiguous_sigs):
        resolved_chunks.append(resolve_ambiguous_signature(sig, subs_df, claims_df))
    resolved_df = pd.concat(resolved_chunks, ignore_index=True) if resolved_chunks else pd.DataFrame(
        columns=["claim","Label","types","subgraph","walked","entity_sig"]
    )

    aligned = pd.concat([clean, resolved_df], ignore_index=True)

    # Sanity note (we proceed even if sizes differ; user can inspect)
    if len(aligned) != len(subs_df) or len(aligned) != len(claims_df):
        print(f"[WARN] Alignment sizes differ (aligned={len(aligned)}, subgraphs={len(subs_df)}, claims={len(claims_df)}).")

    return aligned

# -------------------
# Build filtered dataset
# -------------------
def build_filtered(split="test", k=10, M=M_PREFILTER, model=MODEL, temperature=TEMPERATURE, seed=SEED, limit=None):
    print("=" * 80)
    print(f"LLM-Filtered Subgraphs :: split={split} k={k} M={M} model={model} temp={temperature} seed={seed}")
    print("=" * 80)

    client = OpenAI()
    conn = init_cache(split)

    # Load
    print("Loading claims/subgraphs...")
    claims_df = load_claims(split)
    subs_df = load_subgraphs(split)

    print(f"Claims: {len(claims_df)}")
    print(f"Subgraphs:{len(subs_df)}")

    # Align
    aligned = align_by_entity_signature(subs_df, claims_df)
    
    print(f"Aligned rows: {len(aligned)}")
    # After align_by_entity_signature(...)
    subs_sigs = set(subs_df['entity_sig'].tolist())
    claim_sigs = set(claims_df['entity_sig'].tolist())
    missing_in_subs = sorted(list(claim_sigs - subs_sigs))[:5]
    missing_in_claim = sorted(list(subs_sigs - claim_sigs))[:5]
    print(f"Unmatched claim signatures: {len(claim_sigs - subs_sigs)} (showing up to 5):", missing_in_subs)
    print(f"Unmatched subgraph signatures: {len(subs_sigs - claim_sigs)} (showing up to 5):", missing_in_claim)

    # ---
    # --- FIX 1: STRATIFIED SAMPLING ---
    # ---
    
    # Explode the list so we can sample from ALL tags
    aligned_exploded = aligned.explode("types")
    aligned_exploded["types"] = aligned_exploded["types"].fillna("unknown")
    
    if limit is not None:
        # Stratified sampling: equal samples per reasoning type
        # Use the exploded dataframe here
        reasoning_types = aligned_exploded["types"].value_counts() 
        print(f"\nReasoning type distribution (full dataset):")
        print(reasoning_types)
        
        unique_types = reasoning_types.index.tolist()
        n_types = len(unique_types)
        n_per_type = limit // n_types
        
        print(f"\n[STRATIFIED SAMPLING] Target: {limit} total ({n_per_type} per type)")
        
        sampled_chunks = []
        for rt in unique_types:
            # Sample from the exploded dataframe
            subset = aligned_exploded[aligned_exploded["types"] == rt] 
            n_available = len(subset)
            n_sample = min(n_per_type, n_available)
            
            if n_sample > 0:
                sampled = subset.sample(n=n_sample, random_state=seed)
                sampled_chunks.append(sampled)
                print(f"  {rt:20s}: sampled {n_sample:3d} / {n_available:3d} available")
            else:
                print(f"  {rt:20s}: SKIPPED (no examples available)")
        
        # Re-create 'aligned' from the sampled chunks
        # We must drop duplicates in case one claim had multiple types we sampled
        aligned_indices = pd.concat(sampled_chunks, ignore_index=True).index
        # (Around line 348)
        aligned = aligned.loc[aligned.index.isin(aligned_indices)].drop_duplicates(subset=['claim'])
        print(f"\nTotal sampled: {len(aligned)} rows across {len(sampled_chunks)} reasoning types\n")

    # Iterate & filter
    records = []
    # Sanitize model name for filename (e.g., 'gpt-4o-mini' -> 'gpt-4o-mini')
    model_name_safe = model.replace('/', '_').replace(':', '_')
    jsonl_path = RESULTS_DIR / f"filtered_{split}_k{k}_{model_name_safe}.jsonl"
    out_pkl = RESULTS_DIR / f"filtered_{split}_k{k}_{model_name_safe}.pkl"

    kept_count = 0
    orig_total = 0

    with jsonl_path.open("w") as jf:
        for _, row in tqdm(aligned.iterrows(), total=len(aligned)):
            claim_text = row["claim"]
            walked = row.get("walked") or {}
            triples = []
            # Collect triples from both walkable and connected
            for t in (walked.get("walkable", []) + walked.get("connected", [])):
                if isinstance(t, (list, tuple)) and len(t) == 3:
                    triples.append(tuple(t))

            # Filter
            kept = filter_subgraph(client, conn, claim_text, triples, k=k, M=M,
                                   model=model, temperature=temperature, seed=seed)

            # ---
            # --- FIX 2: SAVE FULL REASONING LIST ---
            # ---
            rec = {
                "claim": claim_text,
                "label": label_to_int(row.get("Label")),
                "reasoning_types": (row.get("types") or ["unknown"]), # <-- SAVES THE FULL LIST
                "original_triples": triples,
                "filtered_triples": kept,
                "num_original": len(triples),
                "num_filtered": len(kept),
                # cheap token estimate: claim ~19, overhead ~3, ~13 tokens/triple
                "approx_tokens": int(len(claim_text.split()) + 3 + 13 * len(kept)),
                "entity_sig": row.get("entity_sig", ""),
            }
            records.append(rec)
            jf.write(json.dumps({
                "claim": rec["claim"],
                "label": rec["label"],
                "filtered_triples": rec["filtered_triples"],
                "k": k
            }) + "\n")

            kept_count += len(kept)
            orig_total += len(triples)

    # Save pickle
    pd.DataFrame(records).to_pickle(out_pkl)
    print(f"Wrote: {out_pkl}")
    print(f"Also wrote: {jsonl_path}")

    # Stats
    df = pd.DataFrame(records)
    mean_orig = df["num_original"].replace(0, np.nan).mean()
    reduction = (1 - df["num_filtered"].mean() / mean_orig) * 100 if pd.notna(mean_orig) else 0.0
    print(f"Avg original triples: {df['num_original'].mean():.1f}")
    print(f"Avg filtered triples: {df['num_filtered'].mean():.1f}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Approx avg tokens (after filter): {df['approx_tokens'].mean():.1f}")

    conn.close()
    return out_pkl, jsonl_path

# -------------------
# CLI
# -------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev", "test"], default="test")
    ap.add_argument("--k", type=int, default=10, help="Top-k triples to keep")
    ap.add_argument("--M", type=int, default=M_PREFILTER, help="Prefilter size (triples to send to LLM)")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug)")
    args = ap.parse_args()

    # Allow overrides via env vars (optional)
    model = os.environ.get("LLM_MODEL", args.model)
    temperature = float(os.environ.get("LLM_TEMPERATURE", args.temperature))
    seed = int(os.environ.get("LLM_SEED", args.seed))

    build_filtered(
        split=args.split,
        k=args.k,
        M=args.M,
        model=model,
        temperature=temperature,
        seed=seed,
        limit=args.limit
    )