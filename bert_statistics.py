"""
Analyze BERT input statistics to inform LLM filtering parameters.

Extracts:
- Average tokens per example
- Truncation rate at max_len=512
- Average triples per subgraph
- Token distribution
"""

import pickle
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# Configuration
DATA_DIR = Path("data")
SUBGRAPH_DIR = DATA_DIR / "subgraphs"
MAX_LEN = 512  # BERT's limit

def clean_text(text):
    """Clean entity/relation text for display."""
    if '/' in text:
        text = text.split('/')[-1]
    return text.replace('_', ' ')

def linearize_subgraph(walked_dict, max_triples=None):
    """Linearize subgraph (same as paper's BERT input)."""
    triples = []
    if isinstance(walked_dict, dict):
        triples = walked_dict.get('walkable', []) + walked_dict.get('connected', [])
    
    if max_triples:
        triples = triples[:max_triples]
    
    triple_texts = []
    for s, p, o in triples:
        s_clean = clean_text(s)
        p_clean = clean_text(p)
        o_clean = clean_text(o)
        triple_texts.append(f"{s_clean} {p_clean} {o_clean}")
    
    return " [TRI] ".join(triple_texts) if triple_texts else ""

def analyze_split(split="train"):
    """Analyze token statistics for a split."""
    print(f"\n{'='*80}")
    print(f"Analyzing {split.upper()} split")
    print(f"{'='*80}")
    
    # Load data
    claims_path = DATA_DIR / f"factkg/factkg_{split}.pickle"
    subgraph_path = SUBGRAPH_DIR / f"subgraphs_one_hop_{split}.pkl"
    
    with open(claims_path, 'rb') as f:
        claims_dict = pickle.load(f)
    subgraphs_df = pd.read_pickle(subgraph_path)
    
    # The claim text is the KEY in the dictionary, not a value
    claims_items = list(claims_dict.items())  # Get (claim_text, metadata) pairs
    subgraph_rows = [row for _, row in subgraphs_df.iterrows()]
    
    print(f"\nLoaded {len(claims_items)} examples")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Analyze
    stats = {
        'claim_tokens': [],
        'subgraph_tokens': [],
        'total_tokens': [],
        'num_triples': [],
        'truncated': 0
    }
    
    print("\nProcessing...")
    for (claim_text, claim_metadata), subgraph_row in tqdm(zip(claims_items, subgraph_rows), 
                                                             total=len(claims_items)):
        # claim_text is already the text we need!
        walked_dict = subgraph_row['walked']
        
        # Get triples
        triples = []
        if isinstance(walked_dict, dict):
            triples = walked_dict.get('walkable', []) + walked_dict.get('connected', [])
        
        # Linearize
        subgraph_text = linearize_subgraph(walked_dict)
        
        # Tokenize (as BERT would)
        full_text = f"{claim_text} [SEP] {subgraph_text}"
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        
        claim_tokens = tokenizer.encode(claim_text, add_special_tokens=False)
        subgraph_tokens = tokenizer.encode(subgraph_text, add_special_tokens=False)
        
        # Record stats
        stats['claim_tokens'].append(len(claim_tokens))
        stats['subgraph_tokens'].append(len(subgraph_tokens))
        stats['total_tokens'].append(len(tokens))
        stats['num_triples'].append(len(triples))
        
        if len(tokens) > MAX_LEN:
            stats['truncated'] += 1
    
    # Compute statistics
    n = len(stats['total_tokens'])
    
    print(f"\n{'='*80}")
    print("TOKEN STATISTICS")
    print(f"{'='*80}")
    print(f"Claim tokens:")
    print(f"  Mean: {np.mean(stats['claim_tokens']):.1f}")
    print(f"  Median: {np.median(stats['claim_tokens']):.1f}")
    print(f"  Min: {np.min(stats['claim_tokens'])}")
    print(f"  Max: {np.max(stats['claim_tokens'])}")
    
    print(f"\nSubgraph tokens:")
    print(f"  Mean: {np.mean(stats['subgraph_tokens']):.1f}")
    print(f"  Median: {np.median(stats['subgraph_tokens']):.1f}")
    print(f"  Min: {np.min(stats['subgraph_tokens'])}")
    print(f"  Max: {np.max(stats['subgraph_tokens'])}")
    
    print(f"\nTotal tokens (claim + [SEP] + subgraph):")
    print(f"  Mean: {np.mean(stats['total_tokens']):.1f}")
    print(f"  Median: {np.median(stats['total_tokens']):.1f}")
    print(f"  Min: {np.min(stats['total_tokens'])}")
    print(f"  Max: {np.max(stats['total_tokens'])}")
    
    print(f"\nTruncation:")
    print(f"  Max length: {MAX_LEN}")
    print(f"  Truncated: {stats['truncated']} ({stats['truncated']/n*100:.1f}%)")
    print(f"  Not truncated: {n - stats['truncated']} ({(n-stats['truncated'])/n*100:.1f}%)")
    
    print(f"\nTriples per subgraph:")
    print(f"  Mean: {np.mean(stats['num_triples']):.1f}")
    print(f"  Median: {np.median(stats['num_triples']):.1f}")
    print(f"  Min: {np.min(stats['num_triples'])}")
    print(f"  Max: {np.max(stats['num_triples'])}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR LLM FILTERING")
    print(f"{'='*80}")
    
    avg_tokens = np.mean(stats['total_tokens'])
    avg_triples = np.mean(stats['num_triples'])
    
    # Estimate tokens per triple
    avg_subgraph_tokens = np.mean(stats['subgraph_tokens'])
    tokens_per_triple = avg_subgraph_tokens / avg_triples if avg_triples > 0 else 30
    
    print(f"\nEstimated tokens per triple: ~{tokens_per_triple:.1f}")
    print(f"\nSuggested k values for filtering:")
    
    # k=6: ~40% reduction
    k6_tokens = np.mean(stats['claim_tokens']) + 6 * tokens_per_triple + 10  # +10 for overhead
    print(f"  k=6:  ~{k6_tokens:.0f} tokens/example ({(1 - k6_tokens/avg_tokens)*100:.0f}% reduction)")
    
    # k=10: ~25% reduction
    k10_tokens = np.mean(stats['claim_tokens']) + 10 * tokens_per_triple + 10
    print(f"  k=10: ~{k10_tokens:.0f} tokens/example ({(1 - k10_tokens/avg_tokens)*100:.0f}% reduction)")
    
    # k=14: ~15% reduction
    k14_tokens = np.mean(stats['claim_tokens']) + 14 * tokens_per_triple + 10
    print(f"  k=14: ~{k14_tokens:.0f} tokens/example ({(1 - k14_tokens/avg_tokens)*100:.0f}% reduction)")
    
    print(f"\n💡 Recommendation: Start with k=10")
    print(f"   - Should reduce tokens by ~{(1 - k10_tokens/avg_tokens)*100:.0f}%")
    print(f"   - Keeps most relevant triples")
    print(f"   - Expected to match or beat baseline")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "dev", "test"])
    args = parser.parse_args()
    
    analyze_split(args.split)