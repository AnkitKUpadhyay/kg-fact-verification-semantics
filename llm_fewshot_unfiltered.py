"""
Phase 4 Goal 2 - Part A: Few-Shot LLM with Unfiltered Evidence
Focus on examples with >10 triples to show filtering effect.

Usage:
    python llm_fewshot_unfiltered.py --n_test 200 --n_fewshot 10
"""

import json
import pickle
import time
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Configuration
DATA_DIR = Path("data")
SUBGRAPH_DIR = DATA_DIR / "subgraphs"
RESULTS_DIR = Path("results/llm_fewshot")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LLM settings
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0
SEED = 42

# Evaluation types (matching Fact-or-Fiction repo)
REASONING_TYPES = ["existence", "substitution", "multi hop", "multi claim", "negation"]

# Minimum triples threshold
MIN_TRIPLES = 10


def clean_text(text: str) -> str:
    """Clean entity/relation text."""
    if '/' in text:
        text = text.split('/')[-1]
    return text.replace('_', ' ')


def linearize_subgraph(walked_dict: dict, add_ids: bool = True, max_triples: int = None) -> Tuple[str, int]:
    """
    Linearize subgraph to text format.
    
    Args:
        walked_dict: Dictionary with 'walkable' and 'connected' triples
        add_ids: If True, add [ID] prefix for faithful attribution
        max_triples: If set, limit to first N triples (for context safety)
    
    Returns:
        (linearized_text, num_triples_total)
    """
    triples = []
    if isinstance(walked_dict, dict):
        triples = walked_dict.get('walkable', []) + walked_dict.get('connected', [])
    
    if not triples:
        return "", 0
    
    total_triples = len(triples)
    
    # Limit if needed
    if max_triples and len(triples) > max_triples:
        triples = triples[:max_triples]
    
    triple_texts = []
    for i, (s, p, o) in enumerate(triples):
        s_clean = clean_text(s)
        p_clean = clean_text(p)
        o_clean = clean_text(o)
        
        if add_ids:
            triple_texts.append(f"[{i}] {s_clean} --{p_clean}--> {o_clean}")
        else:
            triple_texts.append(f"{s_clean} {p_clean} {o_clean}")
    
    return "\n".join(triple_texts), total_triples


def normalize_label(label) -> bool:
    """Normalize label to boolean (True=SUPPORTED, False=REFUTED)."""
    if isinstance(label, bool):
        return label
    elif isinstance(label, (list, tuple)):
        return normalize_label(label[0])
    elif isinstance(label, str):
        label_upper = label.upper()
        if label_upper in ['SUPPORTED', 'TRUE', '1']:
            return True
        elif label_upper in ['REFUTED', 'FALSE', '0']:
            return False
        else:
            raise ValueError(f"Unknown label: {label}")
    elif isinstance(label, (int, float)):
        return bool(label)
    else:
        raise ValueError(f"Cannot normalize label type: {type(label)}")


def load_and_filter_data(split: str, min_triples: int = MIN_TRIPLES):
    """
    Load data and filter for examples with >= min_triples.
    
    Returns:
        List of dicts with: claim, label, types, triples, num_triples, claim_id
    """
    print(f"\nLoading {split} data...")
    
    # Load claims
    claims_path = DATA_DIR / f"factkg/factkg_{split}.pickle"
    with open(claims_path, 'rb') as f:
        claims_dict = pickle.load(f)
    
    # Load subgraphs
    subgraph_path = SUBGRAPH_DIR / f"subgraphs_one_hop_{split}.pkl"
    subgraphs_df = pd.read_pickle(subgraph_path)
    
    print(f"Loaded {len(claims_dict)} claims and {len(subgraphs_df)} subgraphs")
    
    # Verify alignment
    if len(claims_dict) != len(subgraphs_df):
        print(f"WARNING: Mismatch! {len(claims_dict)} claims vs {len(subgraphs_df)} subgraphs")
    
    # Process with index-based matching (safer than zip)
    data = []
    claims_items = list(claims_dict.items())
    
    for idx, (claim_text, claim_meta) in enumerate(claims_items):
        if idx >= len(subgraphs_df):
            print(f"WARNING: No subgraph for claim {idx}, skipping")
            continue
        
        subgraph_row = subgraphs_df.iloc[idx]
        walked_dict = subgraph_row['walked']
        
        # Extract triples
        triples = []
        if isinstance(walked_dict, dict):
            triples = walked_dict.get('walkable', []) + walked_dict.get('connected', [])
        
        # Filter by min_triples
        if len(triples) >= min_triples:
            # Create stable ID from claim text
            claim_id = hashlib.md5(claim_text.encode()).hexdigest()[:16]
            
            data.append({
                'claim_id': claim_id,
                'claim': claim_text,
                'label': normalize_label(claim_meta['Label']),  # Normalized boolean
                'types': claim_meta.get('types', []),
                'triples': triples,
                'walked_dict': walked_dict,
                'num_triples': len(triples)
            })
    
    print(f"Loaded {len(data)} examples with >={min_triples} triples")
    return data


def stratified_sample(data: List[dict], n_per_type: int, reasoning_types: List[str], seed: int = SEED):
    """
    Sample n_per_type examples for each reasoning type.
    Ensures no duplicates across types - each example appears at most once.
    """
    np.random.seed(seed)
    
    # Group by reasoning type
    type_to_examples = defaultdict(list)
    for ex in data:
        for rtype in reasoning_types:
            if rtype in ex['types']:
                type_to_examples[rtype].append(ex)
    
    # Sample with deduplication
    sampled_ids = set()  # Track claim_ids to avoid duplicates
    sampled = []
    stats = {}
    
    # Sort types by scarcity (sample rarest first)
    types_by_scarcity = sorted(reasoning_types, key=lambda t: len(type_to_examples[t]))
    
    for rtype in types_by_scarcity:
        examples = type_to_examples[rtype]
        n_available = len(examples)
        
        # Filter out already-sampled examples
        available = [ex for ex in examples if ex['claim_id'] not in sampled_ids]
        n_available_unique = len(available)
        n_sample = min(n_available_unique, n_per_type)
        
        if n_sample > 0:
            sampled_indices = np.random.choice(len(available), n_sample, replace=False)
            for i in sampled_indices:
                ex = available[i]
                sampled.append(ex)
                sampled_ids.add(ex['claim_id'])
            
            stats[rtype] = f"{n_sample}/{n_available} (unique: {n_available_unique})"
        else:
            stats[rtype] = f"0/{n_available} (unique: 0)"
    
    print(f"\nSampling statistics (deduplicated):")
    for rtype in reasoning_types:
        print(f"  {rtype}: {stats.get(rtype, '0/0')}")
    print(f"Total unique examples sampled: {len(sampled)}")
    
    return sampled


def create_fewshot_prompt(
    few_shot_examples: List[dict],
    test_claim: str,
    test_evidence: str,
    test_num_triples: int
) -> str:
    """
    Create few-shot prompt with examples.
    
    CRITICAL: Few-shot examples are limited to 10 triples (context safety),
              but TEST evidence is FULL UNFILTERED (all triples).
              This is where the A/B comparison happens!
    """
    
    prompt = """You are an expert fact verification system using knowledge graph evidence.

Task: Determine if claims are SUPPORTED or REFUTED based ONLY on the provided evidence triples.

Instructions:
• Reason ONLY from the evidence - do not use your pre-trained knowledge
• Cite specific evidence by triple ID: [0], [3], [7]
• If evidence is insufficient, make best guess but note uncertainty

Few-Shot Examples:
"""
    
    # Add few-shot examples (LIMIT to 10 triples for context safety)
    for i, ex in enumerate(few_shot_examples, 1):
        # Limit few-shot evidence to 10 triples to save context
        evidence_text, total_triples = linearize_subgraph(
            ex['walked_dict'], 
            add_ids=True,
            max_triples=10  # ← ONLY few-shot is limited!
        )
        verdict = "SUPPORTED" if ex['label'] else "REFUTED"
        
        truncation_note = f" (showing first 10/{total_triples})" if total_triples > 10 else ""
        
        prompt += f"""
Example {i}:
Claim: {ex['claim']}

Evidence{truncation_note}:
{evidence_text}

Verdict: {verdict}

---
"""
    
    # Add test case (FULL UNFILTERED EVIDENCE - THIS IS THE A/B TEST!)
    prompt += f"""
Now evaluate this claim:

Claim: {test_claim}

Evidence (ALL {test_num_triples} triples - UNFILTERED):
{test_evidence}

Output JSON (respond with ONLY valid JSON, no other text):
{{
  "verdict": "SUPPORTED" or "REFUTED",
  "explanation": "Brief reasoning (2-3 sentences)",
  "key_evidence": [list of triple IDs used],
  "confidence": "high", "medium", or "low"
}}
"""
    
    return prompt


def call_llm(client: OpenAI, prompt: str, max_retries: int = 3) -> dict:
    """Call LLM with retry logic."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                seed=SEED,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            text = response.choices[0].message.content.strip()
            result = json.loads(text)
            
            # Validate required fields
            if "verdict" not in result:
                raise ValueError("Missing 'verdict' field")
            
            # Normalize verdict
            result['verdict'] = result['verdict'].upper()
            if result['verdict'] not in ['SUPPORTED', 'REFUTED']:
                raise ValueError(f"Invalid verdict: {result['verdict']}")
            
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} due to: {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                # Return error marker instead of defaulting to SUPPORTED
                return {
                    'verdict': 'ERROR',
                    'explanation': f'Error: {str(e)}',
                    'key_evidence': [],
                    'confidence': 'error',
                    'error': True
                }


def evaluate_faithfulness(explanation: str, key_evidence: List[int], triples: List) -> dict:
    """
    Check if explanation only references provided evidence.
    
    Checks:
    1. Are cited IDs valid?
    2. Are there any citations?
    3. Do entities mentioned in explanation appear in cited triples?
    """
    if not key_evidence or len(key_evidence) == 0:
        return {'faithful': False, 'reason': 'No evidence cited'}
    
    # Check if cited IDs are valid
    invalid_ids = [i for i in key_evidence if i < 0 or i >= len(triples)]
    if invalid_ids:
        return {'faithful': False, 'reason': f'Invalid IDs: {invalid_ids}'}
    
    # Extract entities from cited triples
    cited_entities = set()
    for idx in key_evidence:
        s, p, o = triples[idx]
        # Clean and add entities
        cited_entities.add(clean_text(s).lower())
        cited_entities.add(clean_text(p).lower())
        cited_entities.add(clean_text(o).lower())
    
    # Check if explanation mentions entities from cited triples
    # (Simple heuristic: at least one entity should appear)
    explanation_lower = explanation.lower()
    entities_mentioned = [e for e in cited_entities if e in explanation_lower]
    
    if len(entities_mentioned) == 0:
        return {
            'faithful': False, 
            'reason': 'Explanation does not mention entities from cited triples'
        }
    
    return {
        'faithful': True, 
        'reason': f'Valid citations with {len(entities_mentioned)} entities mentioned'
    }


def run_evaluation(
    few_shot_examples: List[dict],
    test_examples: List[dict],
    output_path: Path
):
    """Run few-shot evaluation."""
    
    print(f"\n{'='*80}")
    print("Running Few-Shot Evaluation (Unfiltered Evidence)")
    print(f"{'='*80}")
    print(f"Few-shot examples: {len(few_shot_examples)}")
    print(f"Test examples: {len(test_examples)}")
    
    # Initialize
    client = OpenAI()
    results = []
    
    # Track metrics
    metrics = {
        'overall': {'correct': 0, 'total': 0, 'errors': 0},
        'by_type': defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0}),
        'negation': {'correct': 0, 'total': 0, 'errors': 0},  # Explicit negation tracking
        'by_confidence': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'faithfulness': {'faithful': 0, 'total': 0}
    }
    
    # Evaluate
    print("\nEvaluating...")
    for ex in tqdm(test_examples):
        # Prepare evidence (FULL unfiltered)
        evidence_text, num_triples = linearize_subgraph(ex['walked_dict'], add_ids=True)
        
        # Create prompt
        prompt = create_fewshot_prompt(
            few_shot_examples,
            ex['claim'],
            evidence_text,
            num_triples
        )
        
        # Call LLM
        llm_result = call_llm(client, prompt)
        
        # Check for error
        if llm_result.get('error', False):
            metrics['overall']['errors'] += 1
            metrics['overall']['total'] += 1
            
            for rtype in REASONING_TYPES:
                if rtype in ex['types']:
                    metrics['by_type'][rtype]['errors'] += 1
                    metrics['by_type'][rtype]['total'] += 1
            
            if 'negation' in ex['types']:
                metrics['negation']['errors'] += 1
                metrics['negation']['total'] += 1
            
            # Store error result
            results.append({
                'claim': ex['claim'],
                'claim_id': ex['claim_id'],
                'true_label': ex['label'],
                'pred_label': None,
                'correct': False,
                'num_triples': num_triples,
                'types': ex['types'],
                'llm_response': llm_result,
                'faithfulness': {'faithful': False, 'reason': 'Error'},
                'error': True
            })
            continue
        
        # Evaluate prediction
        true_label = ex['label']
        pred_label = llm_result['verdict'] == 'SUPPORTED'
        is_correct = (pred_label == true_label)
        
        # Check faithfulness
        faithfulness = evaluate_faithfulness(
            llm_result.get('explanation', ''),
            llm_result.get('key_evidence', []),
            ex['triples']
        )
        
        # Update metrics - overall
        metrics['overall']['correct'] += is_correct
        metrics['overall']['total'] += 1
        
        # Update metrics - by type
        for rtype in REASONING_TYPES:
            if rtype in ex['types']:
                metrics['by_type'][rtype]['correct'] += is_correct
                metrics['by_type'][rtype]['total'] += 1
        
        # Update metrics - explicit negation
        if 'negation' in ex['types']:
            metrics['negation']['correct'] += is_correct
            metrics['negation']['total'] += 1
        
        # Update metrics - by confidence
        confidence = llm_result.get('confidence', 'medium')
        metrics['by_confidence'][confidence]['correct'] += is_correct
        metrics['by_confidence'][confidence]['total'] += 1
        
        # Update metrics - faithfulness
        if faithfulness['faithful']:
            metrics['faithfulness']['faithful'] += 1
        metrics['faithfulness']['total'] += 1
        
        # Store result
        results.append({
            'claim': ex['claim'],
            'claim_id': ex['claim_id'],
            'true_label': true_label,
            'pred_label': pred_label,
            'correct': is_correct,
            'num_triples': num_triples,
            'types': ex['types'],
            'llm_response': llm_result,
            'faithfulness': faithfulness,
            'error': False
        })
        
        # Rate limiting
        time.sleep(0.1)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'model': MODEL,
                'temperature': TEMPERATURE,
                'seed': SEED,
                'n_fewshot': len(few_shot_examples),
                'n_test': len(test_examples),
                'min_triples': MIN_TRIPLES
            },
            'results': results,
            'metrics': metrics
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return results, metrics


def print_metrics(metrics: dict):
    """Print evaluation metrics."""
    
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    
    # Overall
    overall = metrics['overall']
    total_valid = overall['total'] - overall['errors']
    
    # Accuracy including errors (errors count as wrong)
    acc_with_errors = overall['correct'] / overall['total'] * 100 if overall['total'] > 0 else 0
    print(f"\nOverall Accuracy (with errors): {acc_with_errors:.2f}% ({overall['correct']}/{overall['total']})")
    
    # Accuracy excluding errors (only on valid predictions)
    if total_valid > 0:
        acc_without_errors = overall['correct'] / total_valid * 100
        print(f"Overall Accuracy (valid only):  {acc_without_errors:.2f}% ({overall['correct']}/{total_valid})")
    
    if overall['errors'] > 0:
        error_rate = overall['errors'] / overall['total'] * 100
        print(f"API Errors: {overall['errors']} ({error_rate:.1f}%)")
    
    # Negation (explicit)
    neg_stats = metrics['negation']
    if neg_stats['total'] > 0:
        neg_valid = neg_stats['total'] - neg_stats['errors']
        neg_acc_with_errors = neg_stats['correct'] / neg_stats['total'] * 100
        print(f"\n🎯 NEGATION Accuracy (with errors): {neg_acc_with_errors:.2f}% ({neg_stats['correct']}/{neg_stats['total']})")
        
        if neg_valid > 0:
            neg_acc_without_errors = neg_stats['correct'] / neg_valid * 100
            print(f"🎯 NEGATION Accuracy (valid only):  {neg_acc_without_errors:.2f}% ({neg_stats['correct']}/{neg_valid})")
        
        if neg_stats['errors'] > 0:
            print(f"   Negation Errors: {neg_stats['errors']}")
    
    # By reasoning type
    print(f"\nAccuracy by Reasoning Type (valid predictions only):")
    for rtype in REASONING_TYPES:
        if rtype in metrics['by_type']:
            stats = metrics['by_type'][rtype]
            if stats['total'] > 0:
                valid = stats['total'] - stats['errors']
                if valid > 0:
                    acc = stats['correct'] / valid * 100
                    error_note = f" (errors: {stats['errors']})" if stats['errors'] > 0 else ""
                    print(f"  {rtype:20s}: {acc:.2f}% ({stats['correct']}/{valid}){error_note}")
    
    # By confidence
    print(f"\nAccuracy by Confidence:")
    for conf in ['high', 'medium', 'low']:
        if conf in metrics['by_confidence']:
            stats = metrics['by_confidence'][conf]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                print(f"  {conf:10s}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
    
    # Faithfulness
    faith_stats = metrics['faithfulness']
    if faith_stats['total'] > 0:
        faith_rate = faith_stats['faithful'] / faith_stats['total'] * 100
        print(f"\nExplanation Faithfulness: {faith_rate:.2f}% ({faith_stats['faithful']}/{faith_stats['total']})")
    
    print(f"{'='*80}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test", type=int, default=200,
                       help="Total test examples (distributed across types)")
    parser.add_argument("--n_fewshot", type=int, default=10,
                       help="Total few-shot examples (distributed across types)")
    parser.add_argument("--min_triples", type=int, default=MIN_TRIPLES,
                       help="Minimum triples per example")
    args = parser.parse_args()
    
    # Calculate per-type samples
    n_types = len(REASONING_TYPES)
    n_test_per_type = args.n_test // n_types
    n_fewshot_per_type = args.n_fewshot // n_types
    
    print(f"Target: {n_test_per_type} test examples per type ({args.n_test} total)")
    print(f"Target: {n_fewshot_per_type} few-shot examples per type ({args.n_fewshot} total)")
    
    # Load data
    train_data = load_and_filter_data('train', min_triples=args.min_triples)
    test_data = load_and_filter_data('test', min_triples=args.min_triples)
    
    # Sample
    few_shot_examples = stratified_sample(train_data, n_fewshot_per_type, REASONING_TYPES)
    test_examples = stratified_sample(test_data, n_test_per_type, REASONING_TYPES)
    
    # Run evaluation
    output_path = RESULTS_DIR / f"unfiltered_gpt4_1_n{len(test_examples)}_fewshot{len(few_shot_examples)}.json"
    results, metrics = run_evaluation(few_shot_examples, test_examples, output_path)
    
    # Print metrics
    print_metrics(metrics)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()