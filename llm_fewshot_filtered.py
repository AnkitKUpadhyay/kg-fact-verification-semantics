# """
# Phase 4 Goal 2 - Part B: Few-Shot LLM with Filtered Evidence (MATCHED EXAMPLES)
# Ensures we use the EXACT SAME test examples as the unfiltered run.

# Usage:
#     python llm_fewshot_filtered_matched.py \
#         --unfiltered_results results/llm_fewshot/unfiltered_gpt4_1_n200_fewshot10.json \
#         --model gpt-4.1-mini
# """

# import json
# import pickle
# import time
# from pathlib import Path
# from collections import defaultdict
# from typing import List, Dict, Tuple

# import pandas as pd
# from tqdm import tqdm
# from openai import OpenAI

# # Configuration
# DATA_DIR = Path("data")
# RESULTS_DIR = Path("results/llm_fewshot")
# FILTERED_DIR = Path("results/llm_filtered")
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# # LLM settings
# TEMPERATURE = 0
# SEED = 42

# # Evaluation types
# REASONING_TYPES = ["existence", "substitution", "multi hop", "multi claim", "negation"]


# def clean_text(text: str) -> str:
#     """Clean entity/relation text."""
#     if '/' in text:
#         text = text.split('/')[-1]
#     return text.replace('_', ' ')


# def linearize_filtered_triples(triples: List[Tuple], add_ids: bool = True) -> str:
#     """Linearize filtered triples."""
#     if not triples:
#         return ""
    
#     triple_texts = []
#     for i, (s, p, o) in enumerate(triples):
#         s_clean = clean_text(s)
#         p_clean = clean_text(p)
#         o_clean = clean_text(o)
        
#         if add_ids:
#             triple_texts.append(f"[{i}] {s_clean} --{p_clean}--> {o_clean}")
#         else:
#             triple_texts.append(f"{s_clean} {p_clean} {o_clean}")
    
#     return "\n".join(triple_texts)


# def load_unfiltered_results(filepath: Path) -> dict:
#     """Load unfiltered results to get the exact claim_ids used."""
#     print(f"\n📂 Loading unfiltered results from: {filepath}")
    
#     with open(filepath, 'r') as f:
#         data = json.load(f)
    
#     # Extract claim_ids for test and few-shot examples
#     test_claim_ids = [r['claim_id'] for r in data['results'] if not r.get('error', False)]
    
#     # Try to extract few-shot claim_ids from config if available
#     # If not, we'll need to sample them the same way
    
#     print(f"  Found {len(test_claim_ids)} test claim_ids")
#     print(f"  Config: {data['config']}")
    
#     return {
#         'test_claim_ids': test_claim_ids,
#         'config': data['config'],
#         'n_fewshot': data['config']['n_fewshot']
#     }


# def load_filtered_data_by_ids(split: str, claim_ids: List[str], k: int = 10) -> List[dict]:
#     """
#     Load filtered data and match to specific claim_ids.
    
#     Returns examples in the SAME ORDER as claim_ids.
#     """
#     print(f"\n📥 Loading filtered {split} data (k={k})...")
    
#     # Load filtered pkl
#     filtered_path = FILTERED_DIR / f"filtered_{split}_k{k}_gpt-4.1-mini.pkl"
    
#     if not filtered_path.exists():
#         raise FileNotFoundError(f"Filtered data not found: {filtered_path}")
    
#     df = pd.read_pickle(filtered_path)
#     print(f"  Loaded {len(df)} total filtered examples")
    
#     # Create claim_id -> example mapping
#     import hashlib
    
#     claim_to_example = {}
#     for idx, row in df.iterrows():
#         claim_id = hashlib.md5(row['claim'].encode()).hexdigest()[:16]
        
#         filtered_triples = row.get('filtered_triples', [])
#         if len(filtered_triples) == 0:
#             continue  # Skip empty examples
        
#         claim_to_example[claim_id] = {
#             'claim_id': claim_id,
#             'claim': row['claim'],
#             'label': bool(row['label']),
#             'types': row.get('reasoning_types', []),  # ← Fixed column name
#             'triples': filtered_triples,
#             'num_triples': len(filtered_triples),
#             'num_original': row.get('num_original', 0)
#         }
    
#     # Match to claim_ids (preserve order!)
#     matched_examples = []
#     missing_ids = []
    
#     for claim_id in claim_ids:
#         if claim_id in claim_to_example:
#             matched_examples.append(claim_to_example[claim_id])
#         else:
#             missing_ids.append(claim_id)
    
#     print(f"  ✅ Matched {len(matched_examples)}/{len(claim_ids)} examples")
    
#     if missing_ids:
#         print(f"  ⚠️  Missing {len(missing_ids)} examples (filtered to 0 triples or not in filtered set)")
#         print(f"     First 5 missing: {missing_ids[:5]}")
    
#     return matched_examples


# def sample_fewshot_from_filtered(n_fewshot: int, split: str = 'train', k: int = 10, seed: int = SEED):
#     """
#     Sample few-shot examples from filtered training data.
#     Uses stratified sampling with fallback to ensure we get examples.
#     """
#     import numpy as np
#     np.random.seed(seed)
    
#     print(f"\n🎲 Sampling {n_fewshot} few-shot examples from filtered {split} data...")
    
#     # Load all filtered training data
#     filtered_path = FILTERED_DIR / f"filtered_{split}_k{k}_gpt-4.1-mini.pkl"
    
#     if not filtered_path.exists():
#         print(f"  ❌ Filtered training data not found: {filtered_path}")
#         print(f"  💡 Falling back to random sampling from all filtered data")
#         return []
    
#     df = pd.read_pickle(filtered_path)
#     print(f"  📥 Loaded {len(df)} total examples")
    
#     # Convert to list of dicts
#     import hashlib
    
#     all_examples = []
#     for idx, row in df.iterrows():
#         claim_id = hashlib.md5(row['claim'].encode()).hexdigest()[:16]
#         filtered_triples = row.get('filtered_triples', [])
        
#         if len(filtered_triples) == 0:
#             continue
        
#         all_examples.append({
#             'claim_id': claim_id,
#             'claim': row['claim'],
#             'label': bool(row['label']),
#             'types': row.get('reasoning_types', []),  # ← Fixed column name
#             'triples': filtered_triples,
#             'num_triples': len(filtered_triples),
#             'num_original': row.get('num_original', 0)
#         })
    
#     print(f"  📊 Valid examples (with triples): {len(all_examples)}")
    
#     if len(all_examples) == 0:
#         print(f"  ❌ No valid examples found in filtered data!")
#         return []
    
#     # Group by type
#     type_to_examples = defaultdict(list)
#     for ex in all_examples:
#         for rtype in REASONING_TYPES:
#             if rtype in ex['types']:
#                 type_to_examples[rtype].append(ex)
    
#     # Print distribution
#     print(f"  📋 Distribution by type:")
#     for rtype in REASONING_TYPES:
#         print(f"     {rtype}: {len(type_to_examples[rtype])} examples")
    
#     # Try stratified sampling first
#     n_per_type = max(1, n_fewshot // len(REASONING_TYPES))
#     sampled_ids = set()
#     sampled = []
    
#     types_by_scarcity = sorted(REASONING_TYPES, key=lambda t: len(type_to_examples[t]))
    
#     print(f"  🎯 Attempting stratified sampling ({n_per_type} per type)...")
    
#     for rtype in types_by_scarcity:
#         examples = type_to_examples[rtype]
#         available = [ex for ex in examples if ex['claim_id'] not in sampled_ids]
#         n_sample = min(len(available), n_per_type)
        
#         if n_sample > 0:
#             sampled_indices = np.random.choice(len(available), n_sample, replace=False)
#             for i in sampled_indices:
#                 ex = available[i]
#                 sampled.append(ex)
#                 sampled_ids.add(ex['claim_id'])
#             print(f"     {rtype}: sampled {n_sample}")
#         else:
#             print(f"     {rtype}: ⚠️ no available examples")
    
#     # Fallback: if we didn't get enough, sample randomly from remaining
#     if len(sampled) < n_fewshot:
#         print(f"  ⚠️ Only got {len(sampled)}/{n_fewshot} via stratified sampling")
#         print(f"  🔄 Sampling {n_fewshot - len(sampled)} more from remaining pool...")
        
#         remaining = [ex for ex in all_examples if ex['claim_id'] not in sampled_ids]
#         n_additional = min(len(remaining), n_fewshot - len(sampled))
        
#         if n_additional > 0:
#             additional_indices = np.random.choice(len(remaining), n_additional, replace=False)
#             for i in additional_indices:
#                 sampled.append(remaining[i])
    
#     print(f"  ✅ Final sample: {len(sampled)} few-shot examples")
    
#     if len(sampled) < n_fewshot:
#         print(f"  ⚠️ WARNING: Could only sample {len(sampled)}/{n_fewshot} requested examples")
    
#     return sampled


# def create_fewshot_prompt(
#     few_shot_examples: List[dict],
#     test_claim: str,
#     test_triples: List[Tuple],
#     test_num_original: int
# ) -> str:
#     """Create few-shot prompt with FILTERED examples."""
    
#     prompt = """You are an expert fact verification system using knowledge graph evidence.

# Task: Determine if claims are SUPPORTED or REFUTED based ONLY on the provided evidence triples.

# Instructions:
# • Reason ONLY from the evidence - do not use your pre-trained knowledge
# • Cite specific evidence by triple ID: [0], [3], [7]
# • If evidence is insufficient, make best guess but note uncertainty

# Few-Shot Examples:
# """
    
#     for i, ex in enumerate(few_shot_examples, 1):
#         evidence_text = linearize_filtered_triples(ex['triples'], add_ids=True)
#         verdict = "SUPPORTED" if ex['label'] else "REFUTED"
        
#         prompt += f"""
# Example {i}:
# Claim: {ex['claim']}

# Evidence (top-{ex['num_triples']} most relevant):
# {evidence_text}

# Verdict: {verdict}

# ---
# """
    
#     test_evidence_text = linearize_filtered_triples(test_triples, add_ids=True)
    
#     prompt += f"""
# Now evaluate this claim:

# Claim: {test_claim}

# Evidence (top-{len(test_triples)} most relevant):
# {test_evidence_text}

# Output JSON (respond with ONLY valid JSON, no other text):
# {{
#   "verdict": "SUPPORTED" or "REFUTED",
#   "explanation": "Brief reasoning (2-3 sentences)",
#   "key_evidence": [list of triple IDs used],
#   "confidence": "high", "medium", or "low"
# }}
# """
    
#     return prompt


# def call_llm(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> dict:
#     """Call LLM with retry logic."""
    
#     for attempt in range(max_retries):
#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 temperature=TEMPERATURE,
#                 seed=SEED,
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"}
#             )
            
#             text = response.choices[0].message.content.strip()
#             result = json.loads(text)
            
#             if "verdict" not in result:
#                 raise ValueError("Missing 'verdict' field")
            
#             result['verdict'] = result['verdict'].upper()
#             if result['verdict'] not in ['SUPPORTED', 'REFUTED']:
#                 raise ValueError(f"Invalid verdict: {result['verdict']}")
            
#             return result
            
#         except Exception as e:
#             if attempt < max_retries - 1:
#                 print(f"  Retry {attempt + 1}/{max_retries} due to: {e}")
#                 time.sleep(2 ** attempt)
#             else:
#                 return {
#                     'verdict': 'ERROR',
#                     'explanation': f'Error: {str(e)}',
#                     'key_evidence': [],
#                     'confidence': 'error',
#                     'error': True
#                 }


# def evaluate_faithfulness(explanation: str, key_evidence: List[int], triples: List) -> dict:
#     """Check if explanation references provided evidence."""
#     if not key_evidence or len(key_evidence) == 0:
#         return {'faithful': False, 'reason': 'No evidence cited'}
    
#     invalid_ids = [i for i in key_evidence if i < 0 or i >= len(triples)]
#     if invalid_ids:
#         return {'faithful': False, 'reason': f'Invalid IDs: {invalid_ids}'}
    
#     cited_entities = set()
#     for idx in key_evidence:
#         s, p, o = triples[idx]
#         cited_entities.add(clean_text(s).lower())
#         cited_entities.add(clean_text(p).lower())
#         cited_entities.add(clean_text(o).lower())
    
#     explanation_lower = explanation.lower()
#     entities_mentioned = [e for e in cited_entities if e in explanation_lower]
    
#     if len(entities_mentioned) == 0:
#         return {'faithful': False, 'reason': 'No entities mentioned'}
    
#     return {'faithful': True, 'reason': f'{len(entities_mentioned)} entities mentioned'}


# def run_evaluation(
#     model: str,
#     few_shot_examples: List[dict],
#     test_examples: List[dict],
#     output_path: Path
# ):
#     """Run few-shot evaluation."""
    
#     print(f"\n{'='*80}")
#     print(f"Running Few-Shot Evaluation (Filtered Evidence, MATCHED)")
#     print(f"{'='*80}")
#     print(f"Model: {model}")
#     print(f"Few-shot examples: {len(few_shot_examples)}")
#     print(f"Test examples: {len(test_examples)}")
    
#     client = OpenAI()
#     results = []
    
#     metrics = {
#         'overall': {'correct': 0, 'total': 0, 'errors': 0},
#         'by_type': defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0}),
#         'negation': {'correct': 0, 'total': 0, 'errors': 0},
#         'by_confidence': defaultdict(lambda: {'correct': 0, 'total': 0}),
#         'faithfulness': {'faithful': 0, 'total': 0}
#     }
    
#     print("\nEvaluating...")
#     for ex in tqdm(test_examples):
#         prompt = create_fewshot_prompt(
#             few_shot_examples,
#             ex['claim'],
#             ex['triples'],
#             ex.get('num_original', 0)
#         )
        
#         llm_result = call_llm(client, model, prompt)
        
#         if llm_result.get('error', False):
#             metrics['overall']['errors'] += 1
#             metrics['overall']['total'] += 1
            
#             for rtype in REASONING_TYPES:
#                 if rtype in ex['types']:
#                     metrics['by_type'][rtype]['errors'] += 1
#                     metrics['by_type'][rtype]['total'] += 1
            
#             if 'negation' in ex['types']:
#                 metrics['negation']['errors'] += 1
#                 metrics['negation']['total'] += 1
            
#             results.append({
#                 'claim': ex['claim'],
#                 'claim_id': ex['claim_id'],
#                 'true_label': ex['label'],
#                 'pred_label': None,
#                 'correct': False,
#                 'num_triples': ex['num_triples'],
#                 'num_original': ex.get('num_original', 0),
#                 'types': ex['types'],
#                 'llm_response': llm_result,
#                 'faithfulness': {'faithful': False, 'reason': 'Error'},
#                 'error': True
#             })
#             continue
        
#         true_label = ex['label']
#         pred_label = llm_result['verdict'] == 'SUPPORTED'
#         is_correct = (pred_label == true_label)
        
#         faithfulness = evaluate_faithfulness(
#             llm_result.get('explanation', ''),
#             llm_result.get('key_evidence', []),
#             ex['triples']
#         )
        
#         metrics['overall']['correct'] += is_correct
#         metrics['overall']['total'] += 1
        
#         for rtype in REASONING_TYPES:
#             if rtype in ex['types']:
#                 metrics['by_type'][rtype]['correct'] += is_correct
#                 metrics['by_type'][rtype]['total'] += 1
        
#         if 'negation' in ex['types']:
#             metrics['negation']['correct'] += is_correct
#             metrics['negation']['total'] += 1
        
#         confidence = llm_result.get('confidence', 'medium')
#         metrics['by_confidence'][confidence]['correct'] += is_correct
#         metrics['by_confidence'][confidence]['total'] += 1
        
#         if faithfulness['faithful']:
#             metrics['faithfulness']['faithful'] += 1
#         metrics['faithfulness']['total'] += 1
        
#         results.append({
#             'claim': ex['claim'],
#             'claim_id': ex['claim_id'],
#             'true_label': true_label,
#             'pred_label': pred_label,
#             'correct': is_correct,
#             'num_triples': ex['num_triples'],
#             'num_original': ex.get('num_original', 0),
#             'types': ex['types'],
#             'llm_response': llm_result,
#             'faithfulness': faithfulness,
#             'error': False
#         })
        
#         time.sleep(0.1)
    
#     with open(output_path, 'w') as f:
#         json.dump({
#             'config': {
#                 'model': model,
#                 'temperature': TEMPERATURE,
#                 'seed': SEED,
#                 'n_fewshot': len(few_shot_examples),
#                 'n_test': len(test_examples),
#                 'evidence_type': 'filtered',
#                 'k': 10,
#                 'matched': True
#             },
#             'results': results,
#             'metrics': metrics
#         }, f, indent=2)
    
#     print(f"\n✅ Results saved to: {output_path}")
    
#     return results, metrics


# def print_metrics(metrics: dict):
#     """Print metrics."""
#     print(f"\n{'='*80}")
#     print("EVALUATION RESULTS (FILTERED EVIDENCE - MATCHED)")
#     print(f"{'='*80}")
    
#     overall = metrics['overall']
#     total_valid = overall['total'] - overall['errors']
    
#     if overall['total'] > 0:
#         acc = overall['correct'] / total_valid * 100 if total_valid > 0 else 0
#         print(f"\nOverall Accuracy: {acc:.2f}% ({overall['correct']}/{total_valid})")
    
#     neg_stats = metrics['negation']
#     if neg_stats['total'] > 0:
#         neg_valid = neg_stats['total'] - neg_stats['errors']
#         if neg_valid > 0:
#             neg_acc = neg_stats['correct'] / neg_valid * 100
#             print(f"🎯 NEGATION Accuracy: {neg_acc:.2f}% ({neg_stats['correct']}/{neg_valid})")
    
#     print(f"\nAccuracy by Reasoning Type:")
#     for rtype in REASONING_TYPES:
#         if rtype in metrics['by_type']:
#             stats = metrics['by_type'][rtype]
#             valid = stats['total'] - stats['errors']
#             if valid > 0:
#                 acc = stats['correct'] / valid * 100
#                 print(f"  {rtype:20s}: {acc:.2f}% ({stats['correct']}/{valid})")
    
#     print(f"{'='*80}")


# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--unfiltered_results", type=str, required=True,
#                        help="Path to unfiltered results JSON")
#     parser.add_argument("--model", type=str, default="gpt-4.1-mini",
#                        choices=["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o"],
#                        help="LLM model to use")
#     parser.add_argument("--k", type=int, default=10,
#                        help="Number of filtered triples")
#     args = parser.parse_args()
    
#     # Load unfiltered results to get claim_ids
#     unfiltered_data = load_unfiltered_results(Path(args.unfiltered_results))
    
#     # Load filtered data matching those claim_ids
#     test_examples = load_filtered_data_by_ids(
#         split='test',
#         claim_ids=unfiltered_data['test_claim_ids'],
#         k=args.k
#     )
    
#     # Sample few-shot from filtered training data
#     few_shot_examples = sample_fewshot_from_filtered(
#         n_fewshot=unfiltered_data['n_fewshot'],
#         split='train',
#         k=args.k,
#         seed=SEED
#     )
    
#     # Run evaluation
#     model_safe = args.model.replace('.', '_').replace('-', '_')
#     output_path = RESULTS_DIR / f"filtered_k{args.k}_{model_safe}_MATCHED_n{len(test_examples)}_fewshot{len(few_shot_examples)}.json"
    
#     results, metrics = run_evaluation(
#         args.model,
#         few_shot_examples,
#         test_examples,
#         output_path
#     )
    
#     print_metrics(metrics)
    
#     print(f"\n✅ Evaluation complete!")
#     print(f"Results saved to: {output_path}")


# if __name__ == "__main__":
#     main()



# """
# Phase 4 Goal 2 - Part C: Few-Shot LLM with Chain-of-Thought (Gold Examples)
# Uses manually annotated gold-standard examples with full reasoning chains.

# Usage:
#     python llm_fewshot_with_cot.py \
#         --evidence_type filtered \
#         --model gpt-4o-mini \
#         --n_test 200
# """

# import json
# import pickle
# import time
# from pathlib import Path
# from collections import defaultdict
# from typing import List, Dict, Tuple

# import pandas as pd
# from tqdm import tqdm
# from openai import OpenAI

# # Configuration
# DATA_DIR = Path("data")
# SUBGRAPH_DIR = DATA_DIR / "subgraphs"
# RESULTS_DIR = Path("results/llm_fewshot")
# FILTERED_DIR = Path("results/llm_filtered")
# GOLD_EXAMPLES_PATH = Path("fewshot_gold_examples.json")

# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# # LLM settings
# TEMPERATURE = 0
# SEED = 42

# # Evaluation types
# REASONING_TYPES = ["existence", "substitution", "multi hop", "multi claim", "negation"]


# def clean_text(text: str) -> str:
#     """Clean entity/relation text."""
#     if '/' in text:
#         text = text.split('/')[-1]
#     return text.replace('_', ' ')


# def linearize_triples(triples: List[Tuple], add_ids: bool = True) -> str:
#     """Linearize triples to text format."""
#     if not triples:
#         return ""
    
#     triple_texts = []
#     for i, triple in enumerate(triples):
#         # Handle both list and tuple formats
#         if isinstance(triple, (list, tuple)) and len(triple) == 3:
#             s, p, o = triple
#         else:
#             continue
            
#         s_clean = clean_text(str(s))
#         p_clean = clean_text(str(p))
#         o_clean = clean_text(str(o))
        
#         if add_ids:
#             triple_texts.append(f"[{i}] {s_clean} --{p_clean}--> {o_clean}")
#         else:
#             triple_texts.append(f"{s_clean} {p_clean} {o_clean}")
    
#     return "\n".join(triple_texts)


# def load_gold_examples() -> List[dict]:
#     """Load gold-standard few-shot examples with annotations."""
#     print(f"\n📖 Loading gold-standard few-shot examples from: {GOLD_EXAMPLES_PATH}")
    
#     if not GOLD_EXAMPLES_PATH.exists():
#         raise FileNotFoundError(f"Gold examples not found: {GOLD_EXAMPLES_PATH}")
    
#     with open(GOLD_EXAMPLES_PATH, 'r') as f:
#         data = json.load(f)
    
#     examples = data['examples']
#     print(f"  ✅ Loaded {len(examples)} gold examples with chain-of-thought annotations")
    
#     return examples


# def load_test_data_filtered(n_test: int, k: int = 10):
#     """Load filtered test data."""
#     print(f"\n📥 Loading filtered test data (k={k})...")
    
#     filtered_path = FILTERED_DIR / f"filtered_test_k{k}_gpt-4.1-mini.pkl"
    
#     if not filtered_path.exists():
#         raise FileNotFoundError(f"Filtered data not found: {filtered_path}")
    
#     df = pd.read_pickle(filtered_path)
#     print(f"  Loaded {len(df)} total filtered examples")
    
#     # Convert to list format
#     import hashlib
    
#     test_examples = []
#     for idx, row in df.iterrows():
#         claim_id = hashlib.md5(row['claim'].encode()).hexdigest()[:16]
#         filtered_triples = row.get('filtered_triples', [])
        
#         if len(filtered_triples) == 0:
#             continue
        
#         test_examples.append({
#             'claim_id': claim_id,
#             'claim': row['claim'],
#             'label': bool(row['label']),
#             'types': row.get('reasoning_types', []),
#             'triples': filtered_triples,
#             'num_triples': len(filtered_triples),
#             'num_original': row.get('num_original', 0)
#         })
    
#     # Sample stratified by reasoning type
#     import numpy as np
#     np.random.seed(SEED)
    
#     # Group by type
#     type_to_examples = defaultdict(list)
#     for ex in test_examples:
#         for rtype in REASONING_TYPES:
#             if rtype in ex['types']:
#                 type_to_examples[rtype].append(ex)
    
#     # Sample evenly
#     n_per_type = n_test // len(REASONING_TYPES)
#     sampled_ids = set()
#     sampled = []
    
#     types_by_scarcity = sorted(REASONING_TYPES, key=lambda t: len(type_to_examples[t]))
    
#     for rtype in types_by_scarcity:
#         examples = type_to_examples[rtype]
#         available = [ex for ex in examples if ex['claim_id'] not in sampled_ids]
#         n_sample = min(len(available), n_per_type)
        
#         if n_sample > 0:
#             sampled_indices = np.random.choice(len(available), n_sample, replace=False)
#             for i in sampled_indices:
#                 ex = available[i]
#                 sampled.append(ex)
#                 sampled_ids.add(ex['claim_id'])
    
#     print(f"  ✅ Sampled {len(sampled)} test examples")
    
#     return sampled


# def load_test_data_unfiltered(n_test: int, min_triples: int = 10):
#     """Load unfiltered test data."""
#     print(f"\n📥 Loading unfiltered test data...")
    
#     # Load claims
#     claims_path = DATA_DIR / "factkg/factkg_test.pickle"
#     with open(claims_path, 'rb') as f:
#         claims_dict = pickle.load(f)
    
#     # Load subgraphs
#     subgraph_path = SUBGRAPH_DIR / "subgraphs_one_hop_test.pkl"
#     subgraphs_df = pd.read_pickle(subgraph_path)
    
#     print(f"  Loaded {len(claims_dict)} claims and {len(subgraphs_df)} subgraphs")
    
#     # Process
#     import hashlib
    
#     test_examples = []
#     claims_items = list(claims_dict.items())
    
#     for idx, (claim_text, claim_meta) in enumerate(claims_items):
#         if idx >= len(subgraphs_df):
#             continue
        
#         subgraph_row = subgraphs_df.iloc[idx]
#         walked_dict = subgraph_row['walked']
        
#         triples = []
#         if isinstance(walked_dict, dict):
#             triples = walked_dict.get('walkable', []) + walked_dict.get('connected', [])
        
#         if len(triples) >= min_triples:
#             claim_id = hashlib.md5(claim_text.encode()).hexdigest()[:16]
            
#             # Normalize label
#             label = claim_meta['Label']
#             if isinstance(label, str):
#                 label = label.upper() in ['SUPPORTED', 'TRUE', '1']
#             elif isinstance(label, (list, tuple)):
#                 label = label[0]
#             label = bool(label)
            
#             test_examples.append({
#                 'claim_id': claim_id,
#                 'claim': claim_text,
#                 'label': label,
#                 'types': claim_meta.get('types', []),
#                 'triples': triples,
#                 'walked_dict': walked_dict,
#                 'num_triples': len(triples)
#             })
    
#     # Sample stratified
#     import numpy as np
#     np.random.seed(SEED)
    
#     type_to_examples = defaultdict(list)
#     for ex in test_examples:
#         for rtype in REASONING_TYPES:
#             if rtype in ex['types']:
#                 type_to_examples[rtype].append(ex)
    
#     n_per_type = n_test // len(REASONING_TYPES)
#     sampled_ids = set()
#     sampled = []
    
#     types_by_scarcity = sorted(REASONING_TYPES, key=lambda t: len(type_to_examples[t]))
    
#     for rtype in types_by_scarcity:
#         examples = type_to_examples[rtype]
#         available = [ex for ex in examples if ex['claim_id'] not in sampled_ids]
#         n_sample = min(len(available), n_per_type)
        
#         if n_sample > 0:
#             sampled_indices = np.random.choice(len(available), n_sample, replace=False)
#             for i in sampled_indices:
#                 ex = available[i]
#                 sampled.append(ex)
#                 sampled_ids.add(ex['claim_id'])
    
#     print(f"  ✅ Sampled {len(sampled)} test examples")
    
#     return sampled


# def create_cot_prompt(
#     gold_examples: List[dict],
#     test_claim: str,
#     test_triples: List[Tuple],
#     evidence_type: str
# ) -> str:
#     """
#     Create few-shot prompt with FULL chain-of-thought examples.
    
#     Key difference from previous version:
#     - Shows verdict + explanation + key_evidence + confidence
#     - Demonstrates HOW to reason, not just WHAT to predict
#     """
    
#     prompt = """You are an expert fact verification system using knowledge graph evidence.

# Task: Determine if claims are SUPPORTED or REFUTED based ONLY on the provided evidence triples.

# Instructions:
# • Reason ONLY from the evidence - do not use your pre-trained knowledge
# • Cite specific evidence by triple ID: [0], [3], [7]
# • Explain your reasoning in 2-3 sentences
# • Identify key evidence triples that support your verdict
# • Rate your confidence: high, medium, or low

# Few-Shot Examples (with reasoning):
# """
    
#     # Add few-shot examples WITH full chain-of-thought
#     for i, ex in enumerate(gold_examples, 1):
#         evidence_text = linearize_triples(ex['triples'], add_ids=True)
#         verdict = ex['verdict']
#         explanation = ex['explanation']
#         key_evidence = ex['key_evidence']
#         confidence = ex['confidence']
        
#         prompt += f"""
# {'='*80}
# Example {i}:
# Claim: {ex['claim']}

# Evidence ({len(ex['triples'])} triples):
# {evidence_text}

# Response:
# {{
#   "verdict": "{verdict}",
#   "explanation": "{explanation}",
#   "key_evidence": {json.dumps(key_evidence)},
#   "confidence": "{confidence}"
# }}

# """
    
#     # Add test case
#     test_evidence_text = linearize_triples(test_triples, add_ids=True)
    
#     prompt += f"""
# {'='*80}
# Now evaluate this NEW claim:

# Claim: {test_claim}

# Evidence ({len(test_triples)} triples, {evidence_type}):
# {test_evidence_text}

# Output JSON (respond with ONLY valid JSON, no other text):
# {{
#   "verdict": "SUPPORTED" or "REFUTED",
#   "explanation": "Your reasoning (2-3 sentences, cite triple IDs)",
#   "key_evidence": [list of triple IDs you used],
#   "confidence": "high", "medium", or "low"
# }}
# """
    
#     return prompt


# def call_llm(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> dict:
#     """Call LLM with retry logic."""
    
#     for attempt in range(max_retries):
#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 temperature=TEMPERATURE,
#                 seed=SEED,
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"}
#             )
            
#             text = response.choices[0].message.content.strip()
#             result = json.loads(text)
            
#             if "verdict" not in result:
#                 raise ValueError("Missing 'verdict' field")
            
#             result['verdict'] = result['verdict'].upper()
#             if result['verdict'] not in ['SUPPORTED', 'REFUTED']:
#                 raise ValueError(f"Invalid verdict: {result['verdict']}")
            
#             return result
            
#         except Exception as e:
#             if attempt < max_retries - 1:
#                 print(f"  Retry {attempt + 1}/{max_retries} due to: {e}")
#                 time.sleep(2 ** attempt)
#             else:
#                 return {
#                     'verdict': 'ERROR',
#                     'explanation': f'Error: {str(e)}',
#                     'key_evidence': [],
#                     'confidence': 'error',
#                     'error': True
#                 }


# def run_evaluation(
#     model: str,
#     gold_examples: List[dict],
#     test_examples: List[dict],
#     evidence_type: str,
#     output_path: Path
# ):
#     """Run few-shot evaluation with chain-of-thought."""
    
#     print(f"\n{'='*80}")
#     print(f"Running Few-Shot Evaluation with Chain-of-Thought")
#     print(f"{'='*80}")
#     print(f"Model: {model}")
#     print(f"Evidence type: {evidence_type}")
#     print(f"Gold examples (with CoT): {len(gold_examples)}")
#     print(f"Test examples: {len(test_examples)}")
    
#     client = OpenAI()
#     results = []
    
#     metrics = {
#         'overall': {'correct': 0, 'total': 0, 'errors': 0},
#         'by_type': defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0}),
#         'negation': {'correct': 0, 'total': 0, 'errors': 0},
#         'by_confidence': defaultdict(lambda: {'correct': 0, 'total': 0}),
#         'faithfulness': {'faithful': 0, 'total': 0}
#     }
    
#     print("\nEvaluating...")
#     for ex in tqdm(test_examples):
#         prompt = create_cot_prompt(
#             gold_examples,
#             ex['claim'],
#             ex['triples'],
#             evidence_type
#         )
        
#         llm_result = call_llm(client, model, prompt)
        
#         if llm_result.get('error', False):
#             metrics['overall']['errors'] += 1
#             metrics['overall']['total'] += 1
            
#             for rtype in REASONING_TYPES:
#                 if rtype in ex['types']:
#                     metrics['by_type'][rtype]['errors'] += 1
#                     metrics['by_type'][rtype]['total'] += 1
            
#             if 'negation' in ex['types']:
#                 metrics['negation']['errors'] += 1
#                 metrics['negation']['total'] += 1
            
#             results.append({
#                 'claim': ex['claim'],
#                 'claim_id': ex['claim_id'],
#                 'true_label': ex['label'],
#                 'pred_label': None,
#                 'correct': False,
#                 'num_triples': ex['num_triples'],
#                 'types': ex['types'],
#                 'llm_response': llm_result,
#                 'error': True
#             })
#             continue
        
#         true_label = ex['label']
#         pred_label = llm_result['verdict'] == 'SUPPORTED'
#         is_correct = (pred_label == true_label)
        
#         metrics['overall']['correct'] += is_correct
#         metrics['overall']['total'] += 1
        
#         for rtype in REASONING_TYPES:
#             if rtype in ex['types']:
#                 metrics['by_type'][rtype]['correct'] += is_correct
#                 metrics['by_type'][rtype]['total'] += 1
        
#         if 'negation' in ex['types']:
#             metrics['negation']['correct'] += is_correct
#             metrics['negation']['total'] += 1
        
#         confidence = llm_result.get('confidence', 'medium')
#         metrics['by_confidence'][confidence]['correct'] += is_correct
#         metrics['by_confidence'][confidence]['total'] += 1
        
#         # Simple faithfulness check
#         key_evidence = llm_result.get('key_evidence', [])
#         faithful = len(key_evidence) > 0 and all(0 <= i < len(ex['triples']) for i in key_evidence)
#         if faithful:
#             metrics['faithfulness']['faithful'] += 1
#         metrics['faithfulness']['total'] += 1
        
#         results.append({
#             'claim': ex['claim'],
#             'claim_id': ex['claim_id'],
#             'true_label': true_label,
#             'pred_label': pred_label,
#             'correct': is_correct,
#             'num_triples': ex['num_triples'],
#             'types': ex['types'],
#             'llm_response': llm_result,
#             'error': False
#         })
        
#         time.sleep(0.1)
    
#     with open(output_path, 'w') as f:
#         json.dump({
#             'config': {
#                 'model': model,
#                 'temperature': TEMPERATURE,
#                 'seed': SEED,
#                 'evidence_type': evidence_type,
#                 'n_fewshot': len(gold_examples),
#                 'n_test': len(test_examples),
#                 'chain_of_thought': True,
#                 'gold_examples': True
#             },
#             'results': results,
#             'metrics': metrics
#         }, f, indent=2)
    
#     print(f"\n✅ Results saved to: {output_path}")
    
#     return results, metrics


# def print_metrics(metrics: dict):
#     """Print metrics."""
#     print(f"\n{'='*80}")
#     print("EVALUATION RESULTS (With Chain-of-Thought)")
#     print(f"{'='*80}")
    
#     overall = metrics['overall']
#     total_valid = overall['total'] - overall['errors']
    
#     if overall['total'] > 0:
#         acc = overall['correct'] / total_valid * 100 if total_valid > 0 else 0
#         print(f"\nOverall Accuracy: {acc:.2f}% ({overall['correct']}/{total_valid})")
    
#     neg_stats = metrics['negation']
#     if neg_stats['total'] > 0:
#         neg_valid = neg_stats['total'] - neg_stats['errors']
#         if neg_valid > 0:
#             neg_acc = neg_stats['correct'] / neg_valid * 100
#             print(f"🎯 NEGATION Accuracy: {neg_acc:.2f}% ({neg_stats['correct']}/{neg_valid})")
    
#     print(f"\nAccuracy by Reasoning Type:")
#     for rtype in REASONING_TYPES:
#         if rtype in metrics['by_type']:
#             stats = metrics['by_type'][rtype]
#             valid = stats['total'] - stats['errors']
#             if valid > 0:
#                 acc = stats['correct'] / valid * 100
#                 print(f"  {rtype:20s}: {acc:.2f}% ({stats['correct']}/{valid})")
    
#     faith_stats = metrics['faithfulness']
#     if faith_stats['total'] > 0:
#         faith_rate = faith_stats['faithful'] / faith_stats['total'] * 100
#         print(f"\nExplanation Faithfulness: {faith_rate:.2f}% ({faith_stats['faithful']}/{faith_stats['total']})")
    
#     print(f"{'='*80}")


# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--evidence_type", type=str, required=True,
#                        choices=["filtered", "unfiltered"],
#                        help="Type of evidence to use")
#     parser.add_argument("--model", type=str, default="gpt-4o-mini",
#                        choices=["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o"],
#                        help="LLM model to use")
#     parser.add_argument("--n_test", type=int, default=200,
#                        help="Number of test examples")
#     parser.add_argument("--k", type=int, default=10,
#                        help="Number of filtered triples (for filtered evidence)")
#     args = parser.parse_args()
    
#     # Load gold examples
#     gold_examples = load_gold_examples()
    
#     # Load test data
#     if args.evidence_type == "filtered":
#         test_examples = load_test_data_filtered(args.n_test, k=args.k)
#     else:
#         test_examples = load_test_data_unfiltered(args.n_test)
    
#     # Run evaluation
#     model_safe = args.model.replace('.', '_').replace('-', '_')
#     output_path = RESULTS_DIR / f"{args.evidence_type}_CoT_{model_safe}_n{len(test_examples)}.json"
    
#     results, metrics = run_evaluation(
#         args.model,
#         gold_examples,
#         test_examples,
#         args.evidence_type,
#         output_path
#     )
    
#     print_metrics(metrics)
    
#     print(f"\n✅ Evaluation complete!")
#     print(f"Results saved to: {output_path}")


# if __name__ == "__main__":
#     main()

"""
Complete LLM Fact Verification: Memorization vs. KG-Grounded Reasoning
Runs both experiments on the SAME 200 test examples for fair comparison.

Usage:
    python llm_complete_comparison.py \
        --model gpt-4o-mini \
        --n_test 200
"""

import json
import pickle
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# Configuration
DATA_DIR = Path("data")
SUBGRAPH_DIR = DATA_DIR / "subgraphs"
RESULTS_DIR = Path("results/llm_comparison")
GOLD_EXAMPLES_PATH = Path("fewshot_gold_examples.json")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LLM settings
TEMPERATURE = 0
SEED = 42

# Evaluation types
REASONING_TYPES = ["existence", "substitution", "multi hop", "multi claim", "negation"]


def clean_text(text: str) -> str:
    """Clean entity/relation text."""
    if '/' in text:
        text = text.split('/')[-1]
    return text.replace('_', ' ')


def linearize_triples(triples: List[Tuple], add_ids: bool = True) -> str:
    """Linearize triples to text format."""
    if not triples:
        return ""
    
    triple_texts = []
    for i, triple in enumerate(triples):
        if isinstance(triple, (list, tuple)) and len(triple) == 3:
            s, p, o = triple
        else:
            continue
            
        s_clean = clean_text(str(s))
        p_clean = clean_text(str(p))
        o_clean = clean_text(str(o))
        
        if add_ids:
            triple_texts.append(f"[{i}] {s_clean} --{p_clean}--> {o_clean}")
        else:
            triple_texts.append(f"{s_clean} {p_clean} {o_clean}")
    
    return "\n".join(triple_texts)


def load_gold_examples() -> List[dict]:
    """Load gold-standard few-shot examples."""
    print(f"\n📖 Loading gold-standard examples from: {GOLD_EXAMPLES_PATH}")
    
    if not GOLD_EXAMPLES_PATH.exists():
        raise FileNotFoundError(f"Gold examples not found: {GOLD_EXAMPLES_PATH}")
    
    with open(GOLD_EXAMPLES_PATH, 'r') as f:
        data = json.load(f)
    
    examples = data['examples']
    print(f"  ✅ Loaded {len(examples)} gold examples")
    
    return examples


def load_test_data_unfiltered(n_test: int, min_triples: int = 10):
    """Load unfiltered test data with stratified sampling."""
    print(f"\n📥 Loading unfiltered test data...")
    
    # Load claims
    claims_path = DATA_DIR / "factkg/factkg_test.pickle"
    with open(claims_path, 'rb') as f:
        claims_dict = pickle.load(f)
    
    # Load subgraphs
    subgraph_path = SUBGRAPH_DIR / "subgraphs_one_hop_test.pkl"
    subgraphs_df = pd.read_pickle(subgraph_path)
    
    print(f"  Loaded {len(claims_dict)} claims and {len(subgraphs_df)} subgraphs")
    
    # Process
    import hashlib
    
    test_examples = []
    claims_items = list(claims_dict.items())
    
    for idx, (claim_text, claim_meta) in enumerate(claims_items):
        if idx >= len(subgraphs_df):
            continue
        
        subgraph_row = subgraphs_df.iloc[idx]
        walked_dict = subgraph_row['walked']
        
        triples = []
        if isinstance(walked_dict, dict):
            triples = walked_dict.get('walkable', []) + walked_dict.get('connected', [])
        
        if len(triples) >= min_triples:
            claim_id = hashlib.md5(claim_text.encode()).hexdigest()[:16]
            
            # Normalize label
            label = claim_meta['Label']
            if isinstance(label, str):
                label = label.upper() in ['SUPPORTED', 'TRUE', '1']
            elif isinstance(label, (list, tuple)):
                label = label[0]
            label = bool(label)
            
            test_examples.append({
                'claim_id': claim_id,
                'claim': claim_text,
                'label': label,
                'types': claim_meta.get('types', []),
                'triples': triples,
                'num_triples': len(triples)
            })
    
    # Sample stratified by reasoning type
    np.random.seed(SEED)
    
    type_to_examples = defaultdict(list)
    for ex in test_examples:
        for rtype in REASONING_TYPES:
            if rtype in ex['types']:
                type_to_examples[rtype].append(ex)
    
    n_per_type = n_test // len(REASONING_TYPES)
    sampled_ids = set()
    sampled = []
    
    # Sample from scarcest types first
    types_by_scarcity = sorted(REASONING_TYPES, key=lambda t: len(type_to_examples[t]))
    
    for rtype in types_by_scarcity:
        examples = type_to_examples[rtype]
        available = [ex for ex in examples if ex['claim_id'] not in sampled_ids]
        n_sample = min(len(available), n_per_type)
        
        if n_sample > 0:
            sampled_indices = np.random.choice(len(available), n_sample, replace=False)
            for i in sampled_indices:
                ex = available[i]
                sampled.append(ex)
                sampled_ids.add(ex['claim_id'])
    
    print(f"  ✅ Sampled {len(sampled)} test examples (stratified by reasoning type)")
    
    # Print distribution
    type_counts = defaultdict(int)
    for ex in sampled:
        for rtype in ex['types']:
            type_counts[rtype] += 1
    
    print(f"\n  Distribution:")
    for rtype in REASONING_TYPES:
        print(f"    {rtype:20s}: {type_counts[rtype]:3d} examples")
    
    return sampled


def create_memorization_prompt(
    gold_examples: List[dict],
    test_claim: str
) -> str:
    """
    Create few-shot prompt WITHOUT KG evidence (tests memorization).
    Replicates Fact vs Fiction paper's approach.
    """
    
    prompt = """Task: Determine the truth value (True or False) of the following claims based on information verifiable from Wikipedia, as represented in the DBpedia knowledge graph. Provide your answers without using real-time internet searches or code analysis, relying solely on your pre-trained knowledge.

Instructions:
- Base your answers solely on your knowledge as of your last training cut-off
- Respond with True for verifiable claims, and False otherwise
- Include a brief explanation for each answer, explaining your reasoning based on your pre-training
- If the claim is vague or lacks specific information, please make an educated guess on whether it is likely to be True or False

Output Format: JSON with "verdict" and "explanation"

Few-Shot Examples:
"""
    
    # Add few-shot examples WITHOUT evidence (claim + label only)
    for i, ex in enumerate(gold_examples, 1):
        verdict = "True" if ex['verdict'] == "SUPPORTED" else "False"
        
        prompt += f"""
Example {i}:
Claim: {ex['claim']}
Answer: {verdict}

"""
    
    # Add test case
    prompt += f"""
Now evaluate this claim based on your pre-trained knowledge:

Claim: {test_claim}

Output JSON (respond with ONLY valid JSON, no other text):
{{
  "verdict": "True" or "False",
  "explanation": "Your reasoning based on pre-trained knowledge (2-3 sentences)"
}}
"""
    
    return prompt


def create_kg_grounded_prompt(
    gold_examples: List[dict],
    test_claim: str,
    test_triples: List[Tuple]
) -> str:
    """
    Create few-shot prompt WITH KG evidence and chain-of-thought.
    Tests reasoning over explicit evidence.
    """
    
    prompt = """You are an expert fact verification system using knowledge graph evidence.

Task: Determine if claims are SUPPORTED or REFUTED based ONLY on the provided evidence triples.

Instructions:
- Reason ONLY from the evidence - do not use your pre-trained knowledge
- Cite specific evidence by triple ID: [0], [3], [7]
- Explain your reasoning in 2-3 sentences
- Identify key evidence triples that support your verdict

Few-Shot Examples (with reasoning):
"""
    
    # Add few-shot examples WITH full chain-of-thought
    for i, ex in enumerate(gold_examples, 1):
        evidence_text = linearize_triples(ex['triples'], add_ids=True)
        verdict = ex['verdict']
        explanation = ex['explanation']
        
        prompt += f"""
{'='*80}
Example {i}:
Claim: {ex['claim']}

Evidence ({len(ex['triples'])} triples):
{evidence_text}

Response:
{{
  "verdict": "{verdict}",
  "explanation": "{explanation}"
}}

"""
    
    # Add test case
    test_evidence_text = linearize_triples(test_triples, add_ids=True)
    
    prompt += f"""
{'='*80}
Now evaluate this NEW claim:

Claim: {test_claim}

Evidence ({len(test_triples)} triples, unfiltered):
{test_evidence_text}

Output JSON (respond with ONLY valid JSON, no other text):
{{
  "verdict": "SUPPORTED" or "REFUTED",
  "explanation": "Your reasoning (2-3 sentences, cite triple IDs)"
}}
"""
    
    return prompt


def call_llm(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> dict:
    """Call LLM with retry logic."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                seed=SEED,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            text = response.choices[0].message.content.strip()
            result = json.loads(text)
            
            if "verdict" not in result:
                raise ValueError("Missing 'verdict' field")
            
            result['verdict'] = result['verdict'].upper()
            
            # Normalize verdict
            if result['verdict'] in ['TRUE', 'SUPPORTED']:
                result['verdict'] = 'SUPPORTED'
            elif result['verdict'] in ['FALSE', 'REFUTED']:
                result['verdict'] = 'REFUTED'
            else:
                raise ValueError(f"Invalid verdict: {result['verdict']}")
            
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} due to: {e}")
                time.sleep(2 ** attempt)
            else:
                return {
                    'verdict': 'ERROR',
                    'explanation': f'Error: {str(e)}',
                    'error': True
                }


def run_memorization_baseline(
    model: str,
    gold_examples: List[dict],
    test_examples: List[dict],
    output_path: Path
):
    """Run memorization baseline (no KG evidence)."""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: Memorization Baseline (No KG Evidence)")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Few-shot examples: {len(gold_examples)} (claim + label only)")
    print(f"Test examples: {len(test_examples)}")
    print(f"Replicates: Fact vs Fiction paper's ChatGPT approach")
    
    client = OpenAI()
    results = []
    
    metrics = {
        'overall': {'correct': 0, 'total': 0, 'errors': 0},
        'by_type': defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0}),
    }
    
    print("\nEvaluating (memorization)...")
    for ex in tqdm(test_examples):
        prompt = create_memorization_prompt(
            gold_examples,
            ex['claim']
        )
        
        llm_result = call_llm(client, model, prompt)
        
        if llm_result.get('error', False):
            metrics['overall']['errors'] += 1
            metrics['overall']['total'] += 1
            
            for rtype in REASONING_TYPES:
                if rtype in ex['types']:
                    metrics['by_type'][rtype]['errors'] += 1
                    metrics['by_type'][rtype]['total'] += 1
            
            results.append({
                'claim': ex['claim'],
                'claim_id': ex['claim_id'],
                'true_label': ex['label'],
                'pred_label': None,
                'correct': False,
                'types': ex['types'],
                'llm_response': llm_result,
                'error': True
            })
            continue
        
        true_label = ex['label']
        pred_label = llm_result['verdict'] == 'SUPPORTED'
        is_correct = (pred_label == true_label)
        
        metrics['overall']['correct'] += is_correct
        metrics['overall']['total'] += 1
        
        for rtype in REASONING_TYPES:
            if rtype in ex['types']:
                metrics['by_type'][rtype]['correct'] += is_correct
                metrics['by_type'][rtype]['total'] += 1
        
        results.append({
            'claim': ex['claim'],
            'claim_id': ex['claim_id'],
            'true_label': true_label,
            'pred_label': pred_label,
            'correct': is_correct,
            'types': ex['types'],
            'llm_response': llm_result,
            'error': False
        })
        
        time.sleep(0.1)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'model': model,
                'temperature': TEMPERATURE,
                'seed': SEED,
                'approach': 'memorization',
                'evidence_provided': False,
                'n_fewshot': len(gold_examples),
                'n_test': len(test_examples)
            },
            'results': results,
            'metrics': metrics
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return results, metrics


def run_kg_grounded_reasoning(
    model: str,
    gold_examples: List[dict],
    test_examples: List[dict],
    output_path: Path
):
    """Run KG-grounded reasoning with chain-of-thought."""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 2: KG-Grounded Reasoning (With Evidence + CoT)")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Few-shot examples: {len(gold_examples)} (with full evidence + reasoning)")
    print(f"Test examples: {len(test_examples)}")
    print(f"Approach: Explicit KG evidence + chain-of-thought")
    
    client = OpenAI()
    results = []
    
    metrics = {
        'overall': {'correct': 0, 'total': 0, 'errors': 0},
        'by_type': defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0}),
    }
    
    print("\nEvaluating (KG-grounded)...")
    for ex in tqdm(test_examples):
        prompt = create_kg_grounded_prompt(
            gold_examples,
            ex['claim'],
            ex['triples']
        )
        
        llm_result = call_llm(client, model, prompt)
        
        if llm_result.get('error', False):
            metrics['overall']['errors'] += 1
            metrics['overall']['total'] += 1
            
            for rtype in REASONING_TYPES:
                if rtype in ex['types']:
                    metrics['by_type'][rtype]['errors'] += 1
                    metrics['by_type'][rtype]['total'] += 1
            
            results.append({
                'claim': ex['claim'],
                'claim_id': ex['claim_id'],
                'true_label': ex['label'],
                'pred_label': None,
                'correct': False,
                'num_triples': ex['num_triples'],
                'types': ex['types'],
                'llm_response': llm_result,
                'error': True
            })
            continue
        
        true_label = ex['label']
        pred_label = llm_result['verdict'] == 'SUPPORTED'
        is_correct = (pred_label == true_label)
        
        metrics['overall']['correct'] += is_correct
        metrics['overall']['total'] += 1
        
        for rtype in REASONING_TYPES:
            if rtype in ex['types']:
                metrics['by_type'][rtype]['correct'] += is_correct
                metrics['by_type'][rtype]['total'] += 1
        
        results.append({
            'claim': ex['claim'],
            'claim_id': ex['claim_id'],
            'true_label': true_label,
            'pred_label': pred_label,
            'correct': is_correct,
            'num_triples': ex['num_triples'],
            'types': ex['types'],
            'llm_response': llm_result,
            'error': False
        })
        
        time.sleep(0.1)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'model': model,
                'temperature': TEMPERATURE,
                'seed': SEED,
                'approach': 'kg_grounded',
                'evidence_provided': True,
                'chain_of_thought': True,
                'n_fewshot': len(gold_examples),
                'n_test': len(test_examples)
            },
            'results': results,
            'metrics': metrics
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return results, metrics


def print_comparison(
    memorization_metrics: dict,
    kg_grounded_metrics: dict,
    model: str
):
    """Print side-by-side comparison."""
    
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON: Memorization vs. KG-Grounded Reasoning ({model})")
    print(f"{'='*80}")
    
    # Overall accuracy
    mem_overall = memorization_metrics['overall']
    kg_overall = kg_grounded_metrics['overall']
    
    mem_valid = mem_overall['total'] - mem_overall['errors']
    kg_valid = kg_overall['total'] - kg_overall['errors']
    
    mem_acc = mem_overall['correct'] / mem_valid * 100 if mem_valid > 0 else 0
    kg_acc = kg_overall['correct'] / kg_valid * 100 if kg_valid > 0 else 0
    
    print(f"\nOverall Accuracy:")
    print(f"  Memorization (no evidence):  {mem_acc:.2f}% ({mem_overall['correct']}/{mem_valid})")
    print(f"  KG-Grounded (with evidence): {kg_acc:.2f}% ({kg_overall['correct']}/{kg_valid})")
    print(f"  → Improvement: +{kg_acc - mem_acc:.2f}%")
    
    # Per-reasoning-type comparison
    print(f"\nAccuracy by Reasoning Type:")
    print(f"  {'Type':<20} {'Memorization':<15} {'KG-Grounded':<15} {'Improvement':<15}")
    print(f"  {'-'*65}")
    
    for rtype in REASONING_TYPES:
        mem_stats = memorization_metrics['by_type'][rtype]
        kg_stats = kg_grounded_metrics['by_type'][rtype]
        
        mem_valid_type = mem_stats['total'] - mem_stats['errors']
        kg_valid_type = kg_stats['total'] - kg_stats['errors']
        
        mem_acc_type = mem_stats['correct'] / mem_valid_type * 100 if mem_valid_type > 0 else 0
        kg_acc_type = kg_stats['correct'] / kg_valid_type * 100 if kg_valid_type > 0 else 0
        improvement = kg_acc_type - mem_acc_type
        
        print(f"  {rtype:<20} {mem_acc_type:>6.2f}% {' '*7} {kg_acc_type:>6.2f}% {' '*7} {improvement:>+6.2f}%")
    
    print(f"{'='*80}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete comparison: Memorization vs. KG-Grounded Reasoning"
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       choices=["gpt-4.1-mini", "gpt-4o-mini"],
                       help="LLM model to use")
    parser.add_argument("--n_test", type=int, default=200,
                       help="Number of test examples")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE LLM COMPARISON EXPERIMENT")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Test examples: {args.n_test} (same for both experiments)")
    print(f"Random seed: {SEED}")
    
    # Load gold examples and test data ONCE
    gold_examples = load_gold_examples()
    test_examples = load_test_data_unfiltered(args.n_test)
    
    # Run both experiments on the SAME test examples
    model_safe = args.model.replace('.', '_').replace('-', '_')
    
    # Experiment 1: Memorization baseline
    mem_output = RESULTS_DIR / f"memorization_{model_safe}_n{len(test_examples)}.json"
    mem_results, mem_metrics = run_memorization_baseline(
        args.model,
        gold_examples,
        test_examples,
        mem_output
    )
    
    # Experiment 2: KG-grounded reasoning
    kg_output = RESULTS_DIR / f"kg_grounded_{model_safe}_n{len(test_examples)}.json"
    kg_results, kg_metrics = run_kg_grounded_reasoning(
        args.model,
        gold_examples,
        test_examples,
        kg_output
    )
    
    # Print comparison
    print_comparison(mem_metrics, kg_metrics, args.model)
    
    print(f"\n✅ Complete comparison finished!")
    print(f"\nResults saved:")
    print(f"  Memorization:  {mem_output}")
    print(f"  KG-Grounded:   {kg_output}")


if __name__ == "__main__":
    main()