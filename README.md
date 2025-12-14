# Evidence Grounding vs. Memorization: Why Neural Semantics Matter for Knowledge Graph Fact Verification

**Course Project - Knowledge-Graph Powered Hybrid AI**  
*Ankit Upadhyay*

## Overview

This repository contains the code and experiments for my research on knowledge graph-based fact verification using the FACTKG dataset. The project systematically compares symbolic, neural, and LLM-based approaches to understand what role semantics and explicit evidence play in fact verification.

## Paper

See [`KG_Project_Paper_Dec_4.pdf`](KG_Project_Paper_Dec_4.pdf) for the complete research paper.

## Dataset

**FACTKG** ([Kim et al., 2023](https://aclanthology.org/2023.acl-long.895.pdf)): 108,675 natural language claims derived from DBpedia, each paired with:
- One-hop knowledge graph subgraphs
- Binary labels (SUPPORTED/REFUTED)
- Reasoning type annotations (single-hop, multi-hop, multi-claim, existence, substitution, negation)

## Experiments

### 1. Symbolic Baselines (Feature Engineering)
- **Approach**: 31 hand-crafted features over graph structure, entity coverage, and evidence overlap
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Best Result**: 63.96% accuracy (Logistic Regression)
- **Key Finding**: Symbolic features fail on negation (40.10%) and multi-hop reasoning (55.48%)

### 2. Neural Encoders (BERT)
- **Approach**: BERT-base fine-tuned on linearized KG subgraphs
- **Result**: 92.68% test accuracy
- **Key Finding**: Token-level neural semantics dramatically outperform symbolic features, especially on negation (91.70%) and existence (98.15%)

### 3. Graph Neural Networks (QA-GNN)
- **Approach**: Message passing over KG structure with claim encoding
- **Models**: 
  - QA-GNN baseline: 69.64%
  - Improved QA-GNN (with cross-attention): 69.74%
- **Key Finding**: Graph structure alone underperforms linearized text encoding, particularly on negation (~50%)

### 4. LLM-Based Semantic Filtering
- **Approach**: Use GPT-4.1-mini to select 10 most relevant triples per claim before BERT training
- **Setup**: Train BERT on 9,706 examples (filtered vs. unfiltered)
- **Results**:
  - **Filtered**: 78.85% accuracy
  - **Unfiltered**: 52.70% accuracy
  - **Gap**: +26.15 points
- **Key Finding**: Semantic quality of training data matters more than quantity

### 5. Memorization vs. KG-Grounded LLM Reasoning
- **Setup**: 300 stratified test claims, comparing two modes:
  - **Memorization**: Claims only (no KG evidence)
  - **KG-Grounded**: Claims + full subgraphs + chain-of-thought prompting
- **Models**: GPT-4o-mini, GPT-4.1-mini

**Results**:

| Model | Memorization | KG-Grounded | Improvement |
|-------|-------------|-------------|-------------|
| GPT-4o-mini | 71.67% | 84.33% | +12.67 |
| GPT-4.1-mini | 74.67% | 84.00% | +9.33 |

**Per-reasoning-type findings** (GPT-4.1-mini):
- Existence: +22.86 points (65.71% → 88.57%)
- Negation: +11.96 points (70.65% → 82.61%)
- Multi-hop: +0.00 points (73.03% → 73.03%)

## Key Files

### Core Experiments
- `train_all_classical.py` - Symbolic baselines (Logistic Regression, Random Forest, XGBoost)
- `feature_extractor.py` - Extract 31 hand-crafted features from KG subgraphs
- `run_stuff_updated.py` - BERT baseline training
- `train_gnn.py` - QA-GNN baseline
- `improved_qagnn_v2.py` - Improved QA-GNN with cross-attention
- `models_cross_attn.py` - Cross-attention architecture

### LLM Experiments
- `llm_fewshot_filtered.py` - LLM filtering experiments (GPT-4.1-mini selects relevant triples)
- `llm_fewshot_unfiltered.py` - Comparison with unfiltered data
- `compare_llms.py` - Memorization vs. KG-grounded comparison (GPT-4o-mini, GPT-4.1-mini)

### Analysis
- `bert_statistics.py` - Token length and truncation analysis
- `all_ml_model_analysis.py` - Compare symbolic model performance

## Main Findings

1. **Symbolic ceiling is ~64%**: Hand-crafted features capture some patterns but fail on compositional semantics
2. **Neural encoders win**: BERT over linearized KG text achieves 92.68%, far exceeding GNNs (~70%)
3. **Semantic filtering works**: LLM-curated training data yields 26-point improvement over random sampling
4. **KG grounding matters**: Explicit evidence improves LLM accuracy by 9-13 points over memorization alone
5. **Multi-hop remains hard**: Even with evidence, LLMs show no improvement on multi-hop reasoning

## Requirements
```bash
# Core dependencies
torch>=1.10
transformers>=4.30
openai>=1.0
pandas
numpy
scikit-learn
networkx
```

## Citation

If you use this code or findings, please cite:
```bibtex
@article{upadhyay2024evidence,
  title={Evidence Grounding vs. Memorization: Why Neural Semantics Matter for Knowledge Graph Fact Verification},
  author={Upadhyay, Ankit},
  year={2024}
}
```

## Acknowledgments

This project builds upon:
- [FACTKG Dataset](https://github.com/jiho283/FactKG) (Kim et al., 2023)
- [Fact or Fiction Repository](https://github.com/Tobias-Opsahl/Fact-or-Fiction) (Opsahl, 2024)

## License

This project is for academic use.
