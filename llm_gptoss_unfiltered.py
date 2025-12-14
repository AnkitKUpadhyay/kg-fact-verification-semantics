# #!/usr/bin/env python3
# """
# Phase 4 Goal 2 - Part C: Gold Standard Few-Shot with GPT-OSS via HuggingFace

# Uses:
# 1. Gold standard few-shot examples (from fewshot_gold_standard.json)
# 2. GPT-OSS reasoning model (via HuggingFace API router, OpenAI-compatible)
# 3. Original evaluation harness (stratified sampling, faithfulness)
# 4. Unfiltered evidence

# Auth:
#   export HF_TOKEN="hf_xxx"
# """

# import os
# import json
# import pickle
# import time
# import hashlib
# from pathlib import Path
# from collections import defaultdict
# from typing import List, Tuple
# import argparse

# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from openai import OpenAI

# # ---------------- Configuration ----------------
# DATA_DIR = Path("data")
# SUBGRAPH_DIR = DATA_DIR / "subgraphs"
# RESULTS_DIR = Path("results/llm_fewshot")
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# # LLM settings - HF router
# MODEL = "openai/gpt-oss-120b:groq"   # change to your routed model id if needed
# TEMPERATURE = 0
# SEED = 42
# HF_BASE_URL = "https://router.huggingface.co/v1"
# MAX_TOKENS = 1024
# FORCE_JSON_MODE = False  # HF router usually doesn't support OpenAI json mode

# # Evaluation types
# REASONING_TYPES = ["existence", "substitution", "multi hop", "multi claim", "negation"]

# # Minimum triples threshold
# MIN_TRIPLES = 10

# # Gold few-shot file
# GOLD_FEWSHOT_FILE = Path("fewshot_gold_standard.json")


# # ---------------- Few-shot loading / prompt ----------------
# def load_gold_fewshot_examples() -> List[dict]:
#     """Load manually created gold standard few-shot examples."""
#     if not GOLD_FEWSHOT_FILE.exists():
#         raise FileNotFoundError(
#             f"Gold standard file not found: {GOLD_FEWSHOT_FILE}\n"
#             "Please ensure 'fewshot_gold_standard.json' exists."
#         )
#     with open(GOLD_FEWSHOT_FILE, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     examples = data.get("examples", [])
#     print(f"✅ Loaded {len(examples)} gold standard examples")
#     return examples


# def create_concise_fewshot_prompt(
#     gold_examples: List[dict],
#     test_claim: str,
#     test_evidence_text: str,
#     test_num_triples: int
# ) -> str:
#     """
#     Create concise few-shot prompt for GPT-OSS:
#     minimal instructions, clear citations, demonstrate output format.
#     """
#     prompt = (
#         "Task: Determine if the claim is SUPPORTED or REFUTED based ONLY on the provided evidence.\n\n"
#         "Instructions:\n"
#         "• Reason ONLY from evidence - do not use external knowledge\n"
#         "• MUST cite evidence using [ID] format: [0], [3], [7]\n"
#         "• Respond in JSON format shown below\n\n"
#         "Few-Shot Examples:\n"
#     )

#     for i, ex in enumerate(gold_examples, 1):
#         prompt += f"\n--- Example {i} ---\n\n"
#         prompt += f"Claim: {ex['claim']}\n\n"
#         prompt += "Evidence:\n"
#         for ev_line in ex["evidence"]:
#             prompt += f"{ev_line}\n"
#         reasoning_to_show = {
#             "verdict": ex["reasoning"]["verdict"],
#             "explanation": ex["reasoning"]["explanation"],
#             "key_evidence": ex["reasoning"]["key_evidence"],
#             "confidence": ex["reasoning"]["confidence"]
#         }
#         prompt += "\nResponse:\n```json\n"
#         prompt += json.dumps(reasoning_to_show, ensure_ascii=False, indent=2)
#         prompt += "\n```\n"

#     prompt += (
#         f"\n{'='*80}\n\n"
#         "YOUR TASK:\n\n"
#         f"Claim: {test_claim}\n\n"
#         f"Evidence: ({test_num_triples} triples total)\n"
#         f"{test_evidence_text}\n\n"
#         "Response (JSON only):\n```json\n"
#         "{\n"
#         '  "verdict": "SUPPORTED or REFUTED",\n'
#         '  "explanation": "Your reasoning with [ID] citations",\n'
#         '  "key_evidence": [list of IDs],\n'
#         '  "confidence": "high, medium, or low"\n'
#         "}\n"
#         "```"
#     )
#     return prompt


# # ---------------- Data utilities ----------------
# def clean_text(text: str) -> str:
#     """Clean entity/relation text."""
#     if "/" in text:
#         text = text.split("/")[-1]
#     return text.replace("_", " ")


# def linearize_subgraph(
#     walked_dict: dict,
#     add_ids: bool = True,
#     max_triples: int = None
# ) -> Tuple[str, int]:
#     """
#     Linearize subgraph to text.
#     Returns: (linearized_text, total_triples_count)
#     """
#     triples = []
#     if isinstance(walked_dict, dict):
#         triples = (walked_dict.get("walkable", []) or []) + (walked_dict.get("connected", []) or [])
#     if not triples:
#         return "", 0

#     total_triples = len(triples)
#     if max_triples and total_triples > max_triples:
#         triples = triples[:max_triples]

#     lines = []
#     for i, (s, p, o) in enumerate(triples):
#         s_clean = clean_text(str(s))
#         p_clean = clean_text(str(p))
#         o_clean = clean_text(str(o))
#         lines.append(f"[{i}] {s_clean} --{p_clean}--> {o_clean}" if add_ids else f"{s_clean} {p_clean} {o_clean}")
#     return "\n".join(lines), total_triples


# def normalize_label(label) -> bool:
#     """Normalize label to boolean (True=SUPPORTED, False=REFUTED)."""
#     if isinstance(label, bool):
#         return label
#     if isinstance(label, (list, tuple)):
#         return normalize_label(label[0])
#     if isinstance(label, str):
#         v = label.strip().upper()
#         if v in {"SUPPORTED", "TRUE", "1"}:
#             return True
#         if v in {"REFUTED", "FALSE", "0"}:
#             return False
#         raise ValueError(f"Unknown label string: {label}")
#     if isinstance(label, (int, float)):
#         return bool(label)
#     raise ValueError(f"Cannot normalize label type: {type(label)}")


# def load_and_filter_data(split: str, min_triples: int = MIN_TRIPLES) -> List[dict]:
#     """
#     Load data and filter for examples with >= min_triples.
#     Returns: dicts with claim_id, claim, label, types, triples, walked_dict, num_triples
#     """
#     print(f"\nLoading {split} data...")
#     claims_path = DATA_DIR / f"factkg/factkg_{split}.pickle"
#     subgraph_path = SUBGRAPH_DIR / f"subgraphs_one_hop_{split}.pkl"

#     with open(claims_path, "rb") as f:
#         claims_dict = pickle.load(f)
#     subgraphs_df = pd.read_pickle(subgraph_path)
#     print(f"Loaded {len(claims_dict)} claims and {len(subgraphs_df)} subgraphs")

#     data = []
#     claims_items = list(claims_dict.items())

#     for idx, (claim_text, claim_meta) in enumerate(claims_items):
#         if idx >= len(subgraphs_df):
#             print(f"WARNING: No subgraph for claim {idx}, skipping")
#             continue

#         walked_dict = subgraphs_df.iloc[idx]["walked"]
#         triples = []
#         if isinstance(walked_dict, dict):
#             triples = (walked_dict.get("walkable", []) or []) + (walked_dict.get("connected", []) or [])

#         if len(triples) >= min_triples:
#             claim_id = hashlib.md5(claim_text.encode()).hexdigest()[:16]
#             data.append({
#                 "claim_id": claim_id,
#                 "claim": claim_text,
#                 "label": normalize_label(claim_meta["Label"]),
#                 "types": claim_meta.get("types", []),
#                 "triples": triples,
#                 "walked_dict": walked_dict,
#                 "num_triples": len(triples)
#             })

#     print(f"Loaded {len(data)} examples with >={min_triples} triples")
#     return data


# def stratified_sample(
#     data: List[dict],
#     n_per_type: int,
#     reasoning_types: List[str],
#     seed: int = SEED
# ) -> List[dict]:
#     """Sample n_per_type examples for each reasoning type (dedup by claim_id)."""
#     np.random.seed(seed)
#     type_to_examples = defaultdict(list)
#     for ex in data:
#         for rtype in reasoning_types:
#             if rtype in ex.get("types", []):
#                 type_to_examples[rtype].append(ex)

#     sampled_ids = set()
#     sampled = []
#     stats = {}

#     types_by_scarcity = sorted(reasoning_types, key=lambda t: len(type_to_examples[t]))
#     for rtype in types_by_scarcity:
#         examples = type_to_examples[rtype]
#         available = [ex for ex in examples if ex["claim_id"] not in sampled_ids]
#         n_sample = min(len(available), n_per_type)
#         if n_sample > 0:
#             idxs = np.random.choice(len(available), n_sample, replace=False)
#             for i in idxs:
#                 ex = available[i]
#                 sampled.append(ex)
#                 sampled_ids.add(ex["claim_id"])
#             stats[rtype] = f"{n_sample}/{len(examples)} (unique: {len(available)})"
#         else:
#             stats[rtype] = f"0/{len(examples)} (unique: 0)"

#     print("\nSampling statistics (deduplicated):")
#     for rtype in reasoning_types:
#         print(f"  {rtype}: {stats.get(rtype, '0/0')}")
#     print(f"Total unique examples sampled: {len(sampled)}")
#     return sampled


# # ---------------- LLM call + parsing ----------------
# def call_llm(client: OpenAI, prompt: str, max_retries: int = 3) -> dict:
#     """Call the HF-routed model, parse JSON (with fence fallback)."""
#     for attempt in range(max_retries):
#         try:
#             kwargs = dict(
#                 model=MODEL,
#                 temperature=TEMPERATURE,
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=MAX_TOKENS,
#                 seed=SEED
#             )
#             if FORCE_JSON_MODE:
#                 kwargs["response_format"] = {"type": "json_object"}

#             resp = client.chat.completions.create(**kwargs)
#             text = resp.choices[0].message.content.strip()

#             # Strip common markdown fences before json.loads
#             if "```json" in text:
#                 text = text.split("```json", 1)[1].split("```", 1)[0].strip()
#             elif text.startswith("```") and text.endswith("```"):
#                 text = text.strip("`").strip()

#             result = json.loads(text)

#             # Validate required fields
#             v = result.get("verdict", "")
#             verdict = v.upper() if isinstance(v, str) else v
#             if verdict not in {"SUPPORTED", "REFUTED"}:
#                 raise ValueError(f"Invalid verdict: {verdict}")
#             result["verdict"] = verdict
#             return result

#         except Exception as e:
#             if attempt < max_retries - 1:
#                 print(f"  Retry {attempt + 1}/{max_retries} due to: {e}")
#                 time.sleep(2 ** attempt)
#             else:
#                 print(f"  Failed after {max_retries} attempts: {e}")
#                 return {
#                     "verdict": "ERROR",
#                     "explanation": f"Error: {e}",
#                     "key_evidence": [],
#                     "confidence": "error",
#                     "error": True
#                 }


# # ---------------- Faithfulness ----------------
# def evaluate_faithfulness(explanation: str, key_evidence: List[int], triples: List) -> dict:
#     """Check if explanation references provided evidence: valid IDs + mentions."""
#     if not key_evidence:
#         return {"faithful": False, "reason": "No evidence cited"}

#     invalid_ids = [i for i in key_evidence if (i < 0 or i >= len(triples))]
#     if invalid_ids:
#         return {"faithful": False, "reason": f"Invalid IDs: {invalid_ids}"}

#     cited_entities = set()
#     for idx in key_evidence:
#         s, p, o = triples[idx]
#         cited_entities.add(clean_text(s).lower())
#         cited_entities.add(clean_text(p).lower())
#         cited_entities.add(clean_text(o).lower())

#     explanation_lower = (explanation or "").lower()
#     mentioned = [e for e in cited_entities if e in explanation_lower and len(e) > 2]
#     if not mentioned:
#         return {"faithful": False, "reason": "Explanation does not mention cited entities"}

#     return {"faithful": True, "reason": f"Valid citations with {len(mentioned)} entities mentioned"}


# # ---------------- Evaluation loop ----------------
# def run_evaluation(
#     gold_few_shot_examples: List[dict],
#     test_examples: List[dict],
#     output_path: Path
# ):
#     """Run few-shot evaluation with GPT-OSS via HuggingFace API."""
#     print(f"\n{'='*80}")
#     print("Gold Standard Few-Shot Evaluation with GPT-OSS (HuggingFace API)")
#     print(f"{'='*80}")
#     print(f"Model: {MODEL}")
#     print(f"Few-shot examples: {len(gold_few_shot_examples)}")
#     print(f"Test examples: {len(test_examples)}")

#     # Auth + client
#     hf_token = os.environ.get("HF_TOKEN")
#     if not hf_token:
#         raise ValueError(
#             "HF_TOKEN not found in environment variables.\n"
#             "Get token: https://huggingface.co/settings/tokens\n"
#             "export HF_TOKEN='hf_xxx'"
#         )
#     print("✅ HF_TOKEN found.")
#     client = OpenAI(base_url=HF_BASE_URL, api_key=hf_token)

#     results = []
#     metrics = {
#         "overall": {"correct": 0, "total": 0, "errors": 0},
#         "by_type": defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0}),
#         "negation": {"correct": 0, "total": 0, "errors": 0},
#         "by_confidence": defaultdict(lambda: {"correct": 0, "total": 0}),
#         "faithfulness": {"faithful": 0, "total": 0}
#     }

#     print("\nEvaluating...")
#     for ex in tqdm(test_examples):
#         # Prepare evidence (full unfiltered)
#         evidence_text, num_triples = linearize_subgraph(ex["walked_dict"], add_ids=True)

#         # Create concise prompt
#         prompt = create_concise_fewshot_prompt(
#             gold_examples=gold_few_shot_examples,
#             test_claim=ex["claim"],
#             test_evidence_text=evidence_text,
#             test_num_triples=num_triples
#         )

#         # Call LLM
#         llm_result = call_llm(client, prompt)

#         # Error path
#         if llm_result.get("error", False):
#             metrics["overall"]["errors"] += 1
#             metrics["overall"]["total"] += 1
#             for rtype in REASONING_TYPES:
#                 if rtype in ex.get("types", []):
#                     metrics["by_type"][rtype]["errors"] += 1
#                     metrics["by_type"][rtype]["total"] += 1
#             if "negation" in ex.get("types", []):
#                 metrics["negation"]["errors"] += 1
#                 metrics["negation"]["total"] += 1

#             results.append({
#                 "claim": ex["claim"],
#                 "claim_id": ex["claim_id"],
#                 "true_label": ex["label"],
#                 "pred_label": None,
#                 "correct": False,
#                 "num_triples": num_triples,
#                 "types": ex.get("types", []),
#                 "llm_response": llm_result,
#                 "faithfulness": {"faithful": False, "reason": "Error"},
#                 "error": True
#             })
#             continue

#         # Evaluate prediction
#         true_label = ex["label"]
#         pred_label = (llm_result["verdict"] == "SUPPORTED")
#         is_correct = (pred_label == true_label)

#         # Faithfulness
#         faithfulness = evaluate_faithfulness(
#             llm_result.get("explanation", ""),
#             llm_result.get("key_evidence", []),
#             ex["triples"]
#         )

#         # Metrics
#         metrics["overall"]["correct"] += 1 if is_correct else 0
#         metrics["overall"]["total"] += 1

#         for rtype in REASONING_TYPES:
#             if rtype in ex.get("types", []):
#                 metrics["by_type"][rtype]["correct"] += 1 if is_correct else 0
#                 metrics["by_type"][rtype]["total"] += 1

#         if "negation" in ex.get("types", []):
#             metrics["negation"]["correct"] += 1 if is_correct else 0
#             metrics["negation"]["total"] += 1

#         conf = llm_result.get("confidence", "medium")
#         metrics["by_confidence"][conf]["correct"] += 1 if is_correct else 0
#         metrics["by_confidence"][conf]["total"] += 1

#         if faithfulness["faithful"]:
#             metrics["faithfulness"]["faithful"] += 1
#         metrics["faithfulness"]["total"] += 1

#         # Store
#         results.append({
#             "claim": ex["claim"],
#             "claim_id": ex["claim_id"],
#             "true_label": true_label,
#             "pred_label": pred_label,
#             "correct": is_correct,
#             "num_triples": num_triples,
#             "types": ex.get("types", []),
#             "llm_response": llm_result,
#             "faithfulness": faithfulness,
#             "error": False
#         })

#         time.sleep(0.1)  # light rate limiting

#     # Save
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump({
#             "config": {
#                 "model": MODEL,
#                 "temperature": TEMPERATURE,
#                 "seed": SEED,
#                 "n_fewshot": len(gold_few_shot_examples),
#                 "n_test": len(test_examples),
#                 "min_triples": MIN_TRIPLES,
#                 "prompt_type": "concise_fewshot_gptoss_hf",
#                 "api_endpoint": HF_BASE_URL
#             },
#             "results": results,
#             "metrics": metrics
#         }, f, indent=2)

#     print(f"\n✅ Results saved to: {output_path}")
#     return results, metrics


# # ---------------- Reporting ----------------
# def print_metrics(metrics: dict):
#     print(f"\n{'='*80}")
#     print("EVALUATION RESULTS")
#     print(f"{'='*80}")

#     overall = metrics["overall"]
#     total_valid = overall["total"] - overall["errors"]

#     acc_with_errors = (overall["correct"] / overall["total"] * 100) if overall["total"] > 0 else 0.0
#     print(f"\nOverall Accuracy (with errors): {acc_with_errors:.2f}% "
#           f"({overall['correct']}/{overall['total']})")

#     if total_valid > 0:
#         acc_without_errors = overall["correct"] / total_valid * 100
#         print(f"Overall Accuracy (valid only):  {acc_without_errors:.2f}% "
#               f"({overall['correct']}/{total_valid})")

#     if overall["errors"] > 0:
#         error_rate = overall["errors"] / overall["total"] * 100
#         print(f"API Errors: {overall['errors']} ({error_rate:.1f}%)")

#     # Negation (explicit)
#     neg_stats = metrics["negation"]
#     if neg_stats["total"] > 0:
#         neg_valid = neg_stats["total"] - neg_stats["errors"]
#         neg_acc_with_errors = neg_stats["correct"] / neg_stats["total"] * 100
#         print(f"\n🎯 NEGATION Accuracy (with errors): {neg_acc_with_errors:.2f}% "
#               f"({neg_stats['correct']}/{neg_stats['total']})")
#         if neg_valid > 0:
#             neg_acc_without_errors = neg_stats["correct"] / neg_valid * 100
#             print(f"🎯 NEGATION Accuracy (valid only):  {neg_acc_without_errors:.2f}% "
#                   f"({neg_stats['correct']}/{neg_valid})")

#     # By reasoning type
#     print(f"\nAccuracy by Reasoning Type (valid only):")
#     for rtype, stats in metrics["by_type"].items():
#         if stats["total"] == 0:
#             continue
#         valid = stats["total"] - stats["errors"]
#         if valid > 0:
#             acc = stats["correct"] / valid * 100
#             error_note = f" (errors: {stats['errors']})" if stats["errors"] > 0 else ""
#             print(f"  {rtype:20s}: {acc:.2f}% ({stats['correct']}/{valid}){error_note}")

#     # By confidence
#     print(f"\nAccuracy by Confidence:")
#     for conf, stats in metrics["by_confidence"].items():
#         if stats["total"] > 0:
#             acc = stats["correct"] / stats["total"] * 100
#             print(f"  {conf:10s}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

#     # Faithfulness
#     faith_stats = metrics["faithfulness"]
#     if faith_stats["total"] > 0:
#         faith_rate = faith_stats["faithful"] / faith_stats["total"] * 100
#         print(f"\nExplanation Faithfulness: {faith_rate:.2f}% "
#               f"({faith_stats['faithful']}/{faith_stats['total']})")

#     print(f"{'='*80}")


# # ---------------- CLI ----------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_test", type=int, default=200,
#                         help="Total test examples (distributed across types)")
#     parser.add_argument("--min_triples", type=int, default=MIN_TRIPLES,
#                         help="Minimum triples per example")
#     parser.add_argument("--n_fewshot", type=int, default=20,
#                         help="Number of few-shot examples to use (max 20)")
#     args = parser.parse_args()

#     n_types = len(REASONING_TYPES)
#     n_test_per_type = args.n_test // n_types
#     print(f"Target: {n_test_per_type} test examples per type ({args.n_test} total)")

#     gold_examples_all = load_gold_fewshot_examples()
#     gold_examples = gold_examples_all[: args.n_fewshot]
#     print(f"Using {len(gold_examples)} few-shot examples")

#     test_data = load_and_filter_data("test", min_triples=args.min_triples)
#     test_examples = stratified_sample(test_data, n_test_per_type, REASONING_TYPES)

#     output_path = RESULTS_DIR / f"gptoss_gold_n{len(test_examples)}_fewshot{len(gold_examples)}.json"
#     results, metrics = run_evaluation(gold_examples, test_examples, output_path)
#     print_metrics(metrics)

#     print(f"\n✅ Evaluation complete!")
#     print(f"Results saved to: {output_path}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Phase 4 Goal 2 - Part D: Gold Standard Few-Shot with GPT-OSS via Local Ollama

Uses:
1. Gold standard few-shot examples (from fewshot_gold_standard.json)
2. GPT-OSS reasoning model (via local Ollama server)
3. Original evaluation harness (stratified sampling, faithfulness)
4. Unfiltered evidence

Auth:
  This script connects to a local Ollama server. No API key is needed.
"""

import os
import json
import pickle
import time
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ---------------- Configuration ----------------
DATA_DIR = Path("data")
SUBGRAPH_DIR = DATA_DIR / "subgraphs"
RESULTS_DIR = Path("results/llm_fewshot")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LLM settings - Updated for Local Ollama
MODEL = "gpt-oss:20b"  # This must match the model you pulled with ollama_pull
TEMPERATURE = 0
SEED = 42
# This is your assigned port from 'module load ollama'
LOCAL_BASE_URL = "http://localhost:52194/v1"
MAX_TOKENS = 1024
# Set to False, we will parse JSON from markdown
FORCE_JSON_MODE = True

# Evaluation types
REASONING_TYPES = ["existence", "substitution", "multi hop", "multi claim", "negation"]

# Minimum triples threshold
MIN_TRIPLES = 10

# Gold few-shot file
GOLD_FEWSHOT_FILE = Path("fewshot_gold_standard.json")


# ---------------- Few-shot loading / prompt ----------------
def load_gold_fewshot_examples() -> List[dict]:
    """Load manually created gold standard few-shot examples."""
    if not GOLD_FEWSHOT_FILE.exists():
        raise FileNotFoundError(
            f"Gold standard file not found: {GOLD_FEWSHOT_FILE}\n"
            "Please ensure 'fewshot_gold_standard.json' exists."
        )
    with open(GOLD_FEWSHOT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = data.get("examples", [])
    if not examples:
        raise ValueError(f"No examples found in {GOLD_FEWSHOT_FILE}. Check the file format.")
    print(f"✅ Loaded {len(examples)} gold standard examples")
    return examples


def create_concise_fewshot_prompt(
    gold_examples: List[dict],
    test_claim: str,
    test_evidence_text: str,
    test_num_triples: int
) -> str:
    """
    Create concise few-shot prompt for GPT-OSS:
    minimal instructions, clear citations, demonstrate output format.
    """
    prompt = (
        "Task: You are an expert Fact Verifier. Determine if the claim is SUPPORTED or REFUTED based ONLY on the provided evidence.\n\n"
        "Instructions:\n"
        "• Reason ONLY from evidence - do not use external knowledge\n"
        "• MUST cite evidence using [ID] format: [0], [3], [7]\n"
        "• Respond in JSON format shown below\n\n"
        "Few-Shot Examples:\n"
    )

    for i, ex in enumerate(gold_examples, 1):
        prompt += f"\n--- Example {i} ---\n\n"
        prompt += f"Claim: {ex['claim']}\n\n"
        prompt += "Evidence:\n"
        # Use the 'evidence' field which is a list of strings
        for ev_line in ex.get("evidence", []):
            prompt += f"{ev_line}\n"
        
        # Check if reasoning key exists
        if "reasoning" not in ex:
            print(f"Warning: Example {i} is missing 'reasoning' block.")
            continue
            
        reasoning_to_show = {
            "verdict": ex["reasoning"]["verdict"],
            "explanation": ex["reasoning"]["explanation"],
            "key_evidence": ex["reasoning"]["key_evidence"],
            "confidence": ex["reasoning"]["confidence"]
        }
        prompt += "\nResponse:\n```json\n"
        prompt += json.dumps(reasoning_to_show, ensure_ascii=False, indent=2)
        prompt += "\n```\n"

    prompt += (
        f"\n{'='*80}\n\n"
        "YOUR TASK:\n\n"
        f"Claim: {test_claim}\n\n"
        f"Evidence: ({test_num_triples} triples total)\n"
        f"{test_evidence_text}\n\n"
        "Response (JSON only):\n```json\n"
        "{\n"
        '  "verdict": "SUPPORTED or REFUTED",\n'
        '  "explanation": "Your reasoning with [ID] citations",\n'
        '  "key_evidence": [list of IDs],\n'
        '  "confidence": "high, medium, or low"\n'
        "}\n"
        "```"
    )
    return prompt


# ---------------- Data utilities ----------------
def clean_text(text: str) -> str:
    """Clean entity/relation text."""
    if "/" in text:
        text = text.split("/")[-1]
    return text.replace("_", " ")


def linearize_subgraph(
    walked_dict: dict,
    add_ids: bool = True,
    max_triples: int = None
) -> Tuple[str, int]:
    """
    Linearize subgraph to text.
    Returns: (linearized_text, total_triples_count)
    """
    triples = []
    if isinstance(walked_dict, dict):
        triples = (walked_dict.get("walkable", []) or []) + (walked_dict.get("connected", []) or [])
    if not triples:
        return "", 0

    total_triples = len(triples)
    if max_triples and total_triples > max_triples:
        triples = triples[:max_triples]

    lines = []
    for i, (s, p, o) in enumerate(triples):
        s_clean = clean_text(str(s))
        p_clean = clean_text(str(p))
        o_clean = clean_text(str(o))
        lines.append(f"[{i}] {s_clean} --{p_clean}--> {o_clean}" if add_ids else f"{s_clean} {p_clean} {o_clean}")
    return "\n".join(lines), total_triples


def normalize_label(label) -> bool:
    """Normalize label to boolean (True=SUPPORTED, False=REFUTED)."""
    if isinstance(label, bool):
        return label
    if isinstance(label, (list, tuple)):
        return normalize_label(label[0])
    if isinstance(label, str):
        v = label.strip().upper()
        if v in {"SUPPORTED", "TRUE", "1"}:
            return True
        if v in {"REFUTED", "FALSE", "0"}:
            return False
        raise ValueError(f"Unknown label string: {label}")
    if isinstance(label, (int, float)):
        return bool(label)
    raise ValueError(f"Cannot normalize label type: {type(label)}")


def load_and_filter_data(split: str, min_triples: int = MIN_TRIPLES) -> List[dict]:
    """
    Load data and filter for examples with >= min_triples.
    Returns: dicts with claim_id, claim, label, types, triples, walked_dict, num_triples
    """
    print(f"\nLoading {split} data...")
    claims_path = DATA_DIR / f"factkg/factkg_{split}.pickle"
    subgraph_path = SUBGRAPH_DIR / f"subgraphs_one_hop_{split}.pkl"

    with open(claims_path, "rb") as f:
        claims_dict = pickle.load(f)
    subgraphs_df = pd.read_pickle(subgraph_path)
    print(f"Loaded {len(claims_dict)} claims and {len(subgraphs_df)} subgraphs")

    data = []
    claims_items = list(claims_dict.items())

    for idx, (claim_text, claim_meta) in enumerate(claims_items):
        if idx >= len(subgraphs_df):
            print(f"WARNING: No subgraph for claim {idx}, skipping")
            continue

        walked_dict = subgraphs_df.iloc[idx]["walked"]
        triples = []
        if isinstance(walked_dict, dict):
            triples = (walked_dict.get("walkable", []) or []) + (walked_dict.get("connected", []) or [])

        if len(triples) >= min_triples:
            claim_id = hashlib.md5(claim_text.encode()).hexdigest()[:16]
            data.append({
                "claim_id": claim_id,
                "claim": claim_text,
                "label": normalize_label(claim_meta["Label"]),
                "types": claim_meta.get("types", []),
                "triples": triples,
                "walked_dict": walked_dict,
                "num_triples": len(triples)
            })

    print(f"Loaded {len(data)} examples with >={min_triples} triples")
    return data


def stratified_sample(
    data: List[dict],
    n_per_type: int,
    reasoning_types: List[str],
    seed: int = SEED
) -> List[dict]:
    """Sample n_per_type examples for each reasoning type (dedup by claim_id)."""
    np.random.seed(seed)
    type_to_examples = defaultdict(list)
    for ex in data:
        for rtype in reasoning_types:
            if rtype in ex.get("types", []):
                type_to_examples[rtype].append(ex)

    sampled_ids = set()
    sampled = []
    stats = {}

    types_by_scarcity = sorted(reasoning_types, key=lambda t: len(type_to_examples[t]))
    for rtype in types_by_scarcity:
        examples = type_to_examples[rtype]
        available = [ex for ex in examples if ex["claim_id"] not in sampled_ids]
        n_sample = min(len(available), n_per_type)
        if n_sample > 0:
            idxs = np.random.choice(len(available), n_sample, replace=False)
            for i in idxs:
                ex = available[i]
                sampled.append(ex)
                sampled_ids.add(ex["claim_id"])
            stats[rtype] = f"{n_sample}/{len(examples)} (unique: {len(available)})"
        else:
            stats[rtype] = f"0/{len(examples)} (unique: 0)"

    print("\nSampling statistics (deduplicated):")
    for rtype in reasoning_types:
        print(f"  {rtype}: {stats.get(rtype, '0/0')}")
    print(f"Total unique examples sampled: {len(sampled)}")
    return sampled


# ---------------- LLM call + parsing ----------------
def call_llm(client: OpenAI, prompt: str, max_retries: int = 3) -> dict:
    """Call the Ollama-hosted model, parse JSON (with fence fallback)."""
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                seed=SEED
            )
            if FORCE_JSON_MODE:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content.strip()

            # Strip common markdown fences before json.loads
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif text.startswith("```") and text.endswith("```"):
                text = text.strip("`").strip()

            result = json.loads(text)

            # Validate required fields
            v = result.get("verdict", "")
            verdict = v.upper() if isinstance(v, str) else v
            if verdict not in {"SUPPORTED", "REFUTED"}:
                raise ValueError(f"Invalid verdict: {verdict}")
            result["verdict"] = verdict
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} due to: {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {
                    "verdict": "ERROR",
                    "explanation": f"Error: {e}",
                    "key_evidence": [],
                    "confidence": "error",
                    "error": True
                }


# ---------------- Faithfulness ----------------
def evaluate_faithfulness(explanation: str, key_evidence: List[int], triples: List) -> dict:
    """Check if explanation references provided evidence: valid IDs + mentions."""
    if not key_evidence:
        return {"faithful": False, "reason": "No evidence cited"}

    invalid_ids = [i for i in key_evidence if (i < 0 or i >= len(triples))]
    if invalid_ids:
        return {"faithful": False, "reason": f"Invalid IDs: {invalid_ids}"}

    cited_entities = set()
    for idx in key_evidence:
        s, p, o = triples[idx]
        cited_entities.add(clean_text(s).lower())
        cited_entities.add(clean_text(p).lower())
        cited_entities.add(clean_text(o).lower())

    explanation_lower = (explanation or "").lower()
    mentioned = [e for e in cited_entities if e in explanation_lower and len(e) > 2]
    if not mentioned:
        return {"faithful": False, "reason": "Explanation does not mention cited entities"}

    return {"faithful": True, "reason": f"Valid citations with {len(mentioned)} entities mentioned"}

# (Make sure 'import pandas as pd' is at the top of your script)

def load_filtered_test_file(filepath: Path) -> List[dict]:
    """
    Loads a pre-filtered .pkl test file.
    This version is updated to read a Pandas DataFrame.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Filtered test file not found: {filepath}")

    print(f"\nLoading filtered PICKLE (DataFrame) test file from: {filepath}...")
    try:
        # Use pandas to read the pickle file
        df = pd.read_pickle(filepath)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"File {filepath} did not contain a Pandas DataFrame. Found {type(df)}.")
    except Exception as e:
        print(f"Error reading pickle file with pandas: {e}")
        return []

    print(f"Found {len(df)} examples in DataFrame.")

    test_examples_out = []
    # Iterate over the DataFrame rows
    for i, row in df.iterrows():
        
        # 1. Get claim
        claim = row.get("claim")
        if not claim:
            print(f"Warning: Skipping row {i} - no 'claim' column.")
            continue
        
        # 2. Get label
        # The filter script saves the label as 'label' (0 or 1)
        label = row.get("label", row.get("true_label"))
        if label is None:
            print(f"Warning: Skipping row {i} (claim: {claim[:20]}...) - no 'label' or 'true_label' column.")
            continue
        
        # 3. Get triples (The filter script saves them as 'filtered_triples')
        raw_triples = row.get("filtered_triples", row.get("triples", row.get("evidence_triples")))
        
        if not raw_triples:
            # Fallback: check if the 'walked_dict' key exists
            walked_dict_fallback = row.get("walked_dict", {})
            if isinstance(walked_dict_fallback, dict):
                 raw_triples = (walked_dict_fallback.get("walkable", []) or []) + (walked_dict_fallback.get("connected", []) or [])

        if not raw_triples:
            # The filter script might save empty lists, but we need to check
            if isinstance(raw_triples, list) and len(raw_triples) == 0:
                 pass # An empty list is valid (for "no evidence")
            else:
                print(f"Warning: Skipping row {i} (claim: {claim[:20]}...) - no 'filtered_triples', 'triples', 'evidence_triples', or 'walked_dict' key.")
                continue
        
        # 4. Build the 'ex' object that run_evaluation expects
        claim_id = row.get("claim_id", hashlib.md5(claim.encode()).hexdigest()[:16])
        
        test_examples_out.append({
            "claim_id": claim_id,
            "claim": claim,
            "label": normalize_label(label), # normalize_label handles 0/1
            "types": row.get("types", row.get("reasoning_types", [])), # Check for 'types' or 'reasoning_types'
            "triples": raw_triples,  # Used by evaluate_faithfulness
            "walked_dict": {"walkable": raw_triples, "connected": []}, # Used by linearize_subgraph
            "num_triples": len(raw_triples)
        })

    print(f"Successfully loaded {len(test_examples_out)} filtered test examples.")
    return test_examples_out

# ---------------- Evaluation loop ----------------
def run_evaluation(
    gold_few_shot_examples: List[dict],
    test_examples: List[dict],
    output_path: Path
):
    """Run few-shot evaluation with GPT-OSS via Local Ollama API."""
    print(f"\n{'='*80}")
    print("Gold Standard Few-Shot Evaluation with GPT-OSS (Local Ollama API)")
    print(f"{'='*80}")
    print(f"Model: {MODEL}")
    print(f"Endpoint: {LOCAL_BASE_URL}")
    print(f"Few-shot examples: {len(gold_few_shot_examples)}")
    print(f"Test examples: {len(test_examples)}")

    # Auth + client for Local Ollama
    print(f"✅ Connecting to local Ollama server at {LOCAL_BASE_URL}...")
    client = OpenAI(
        base_url=LOCAL_BASE_URL,
        api_key="ollama"  # Ollama doesn't require a key, this is a placeholder
    )

    results = []
    metrics = {
        "overall": {"correct": 0, "total": 0, "errors": 0},
        "by_type": defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0}),
        "negation": {"correct": 0, "total": 0, "errors": 0},
        "by_confidence": defaultdict(lambda: {"correct": 0, "total": 0}),
        "faithfulness": {"faithful": 0, "total": 0}
    }

    print("\nEvaluating...")
    for ex in tqdm(test_examples):
        # Prepare evidence (full unfiltered)
        # We pass the full walked_dict, linearize_subgraph handles creation
        evidence_text, num_triples = linearize_subgraph(ex["walked_dict"], add_ids=True)

        # Create concise prompt
        prompt = create_concise_fewshot_prompt(
            gold_examples=gold_few_shot_examples,
            test_claim=ex["claim"],
            test_evidence_text=evidence_text,
            test_num_triples=num_triples
        )

        # Call LLM
        llm_result = call_llm(client, prompt)

        # Error path
        if llm_result.get("error", False):
            metrics["overall"]["errors"] += 1
            metrics["overall"]["total"] += 1
            for rtype in REASONING_TYPES:
                if rtype in ex.get("types", []):
                    metrics["by_type"][rtype]["errors"] += 1
                    metrics["by_type"][rtype]["total"] += 1
            if "negation" in ex.get("types", []):
                metrics["negation"]["errors"] += 1
                metrics["negation"]["total"] += 1

            results.append({
                "claim": ex["claim"],
                "claim_id": ex["claim_id"],
                "true_label": ex["label"],
                "pred_label": None,
                "correct": False,
                "num_triples": num_triples,
                "types": ex.get("types", []),
                "llm_response": llm_result,
                "faithfulness": {"faithful": False, "reason": "Error"},
                "error": True
            })
            continue

        # Evaluate prediction
        true_label = ex["label"]
        pred_label = (llm_result["verdict"] == "SUPPORTED")
        is_correct = (pred_label == true_label)

        # Faithfulness
        faithfulness = evaluate_faithfulness(
            llm_result.get("explanation", ""),
            llm_result.get("key_evidence", []),
            ex["triples"]
        )

        # Metrics
        metrics["overall"]["correct"] += 1 if is_correct else 0
        metrics["overall"]["total"] += 1

        for rtype in REASONING_TYPES:
            if rtype in ex.get("types", []):
                metrics["by_type"][rtype]["correct"] += 1 if is_correct else 0
                metrics["by_type"][rtype]["total"] += 1

        if "negation" in ex.get("types", []):
            metrics["negation"]["correct"] += 1 if is_correct else 0
            metrics["negation"]["total"] += 1

        conf = llm_result.get("confidence", "medium")
        metrics["by_confidence"][conf]["correct"] += 1 if is_correct else 0
        metrics["by_confidence"][conf]["total"] += 1

        if faithfulness["faithful"]:
            metrics["faithfulness"]["faithful"] += 1
        metrics["faithfulness"]["total"] += 1

        # Store
        results.append({
            "claim": ex["claim"],
            "claim_id": ex["claim_id"],
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": is_correct,
            "num_triples": num_triples,
            "types": ex.get("types", []),
            "llm_response": llm_result,
            "faithfulness": faithfulness,
            "error": False
        })

        time.sleep(0.1)  # light rate limiting

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "n_fewshot": len(gold_few_shot_examples),
                "n_test": len(test_examples),
                "min_triples": MIN_TRIPLES,
                "prompt_type": "concise_fewshot_gptoss_local",
                "api_endpoint": LOCAL_BASE_URL
            },
            "results": results,
            "metrics": metrics
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    return results, metrics


# ---------------- Reporting ----------------
def print_metrics(metrics: dict):
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")

    overall = metrics["overall"]
    total_valid = overall["total"] - overall["errors"]

    acc_with_errors = (overall["correct"] / overall["total"] * 100) if overall["total"] > 0 else 0.0
    print(f"\nOverall Accuracy (with errors): {acc_with_errors:.2f}% "
          f"({overall['correct']}/{overall['total']})")

    if total_valid > 0:
        acc_without_errors = overall["correct"] / total_valid * 100
        print(f"Overall Accuracy (valid only):  {acc_without_errors:.2f}% "
              f"({overall['correct']}/{total_valid})")

    if overall["errors"] > 0:
        error_rate = overall["errors"] / overall["total"] * 100
        print(f"API Errors: {overall['errors']} ({error_rate:.1f}%)")

    # Negation (explicit)
    neg_stats = metrics["negation"]
    if neg_stats["total"] > 0:
        neg_valid = neg_stats["total"] - neg_stats["errors"]
        neg_acc_with_errors = neg_stats["correct"] / neg_stats["total"] * 100
        print(f"\n🎯 NEGATION Accuracy (with errors): {neg_acc_with_errors:.2f}% "
              f"({neg_stats['correct']}/{neg_stats['total']})")
        if neg_valid > 0:
            neg_acc_without_errors = neg_stats["correct"] / neg_valid * 100
            print(f"🎯 NEGATION Accuracy (valid only):  {neg_acc_without_errors:.2f}% "
                  f"({neg_stats['correct']}/{neg_valid})")

    # By reasoning type
    print(f"\nAccuracy by Reasoning Type (valid only):")
    for rtype, stats in metrics["by_type"].items():
        if stats["total"] == 0:
            continue
        valid = stats["total"] - stats["errors"]
        if valid > 0:
            acc = stats["correct"] / valid * 100
            error_note = f" (errors: {stats['errors']})" if stats["errors"] > 0 else ""
            print(f"  {rtype:20s}: {acc:.2f}% ({stats['correct']}/{valid}){error_note}")

    # By confidence
    print(f"\nAccuracy by Confidence:")
    for conf, stats in metrics["by_confidence"].items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {conf:10s}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

    # Faithfulness
    faith_stats = metrics["faithfulness"]
    if faith_stats["total"] > 0:
        faith_rate = faith_stats["faithful"] / faith_stats["total"] * 100
        print(f"\nExplanation Faithfulness: {faith_rate:.2f}% "
              f"({faith_stats['faithful']}/{faith_stats['total']})")

    print(f"{'='*80}")


# ---------------- CLI ----------------
# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Run Gold-Standard Few-Shot with GPT-OSS via Local Ollama"
    )
    # --- Original Args ---
    parser.add_argument("--n_test", type=int, default=200,
                        help="Total test examples to sample (distributed across types)")
    parser.add_argument("--min_triples", type=int, default=MIN_TRIPLES,
                        help="Minimum triples per example (used ONLY if --filtered_test_file is NOT provided)")
    parser.add_argument("--n_fewshot", type=int, default=20,
                        help="Number of few-shot examples to use (max from gold file)")
    
    # --- File Path Arg ---
    parser.add_argument("--filtered_test_file", type=str, default=None,
                        help="Path to a pre-filtered .pkl or .json test file. If provided, will sample from this file.")
    
    args = parser.parse_args()

    # --- 1. Load Gold Few-Shot Examples (Same as before) ---
    gold_examples_all = load_gold_fewshot_examples()
    if args.n_fewshot > len(gold_examples_all):
        print(f"Warning: --n_fewshot={args.n_fewshot} requested, but only {len(gold_examples_all)} available.")
        args.n_fewshot = len(gold_examples_all)
        
    gold_examples = gold_examples_all[: args.n_fewshot]
    print(f"Using {len(gold_examples)} few-shot examples")

    # --- 2. Load and Sample Test Data (NEW LOGIC) ---
    n_types = len(REASONING_TYPES)
    n_test_per_type = args.n_test // n_types
    print(f"Target: {n_test_per_type} test examples per type ({args.n_test} total)")

    if args.filtered_test_file:
        # --- Path A: Load the filtered file AND sample from it ---
        print(f"Loading pre-filtered test file from {args.filtered_test_file}...")
        all_filtered_examples = load_filtered_test_file(Path(args.filtered_test_file))
        if not all_filtered_examples:
            print("Error: No valid examples loaded from filtered test file. Exiting.")
            return
        
        print(f"Loaded {len(all_filtered_examples)} filtered examples. Now sampling {args.n_test} from this set...")
        
        # Now, sample from THIS loaded data
        test_examples = stratified_sample(
            all_filtered_examples,  # Pass in the data we just loaded
            n_test_per_type,
            REASONING_TYPES
        )
        
    else:
        # --- Path B: Load the original unfiltered data and sample from it ---
        print("No filtered test file provided. Using original unfiltered data...")
        
        test_data = load_and_filter_data("test", min_triples=args.min_triples)
        test_examples = stratified_sample(
            test_data, 
            n_test_per_type, 
            REASONING_TYPES
        )

    # --- 3. Run Evaluation (Same as before) ---
    output_file_name = f"gptoss_local_n{len(test_examples)}_fewshot{len(gold_examples)}.json"
    output_path = RESULTS_DIR / output_file_name
    
    results, metrics = run_evaluation(gold_examples, test_examples, output_path)
    print_metrics(metrics)

    print(f"\n✅ Evaluation complete!")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
