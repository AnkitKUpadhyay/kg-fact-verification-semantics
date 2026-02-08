# Evidence Grounding vs. Memorization: Why Neural Semantics Matter for Knowledge Graph Fact Verification

**Accepted for Oral Presentation at the FEVER Workshop, EACL 2026** 🏆  
*Ankit Kumar Upadhyay, John S. Erickson, Deborah L. McGuinness* *Rensselaer Polytechnic Institute*

---

## Overview

This repository contains the code and experiments for research on knowledge graph-based fact verification using the FACTKG dataset. This work systematically compares symbolic, neural, and LLM-based approaches to isolate the impact of token-level semantics and explicit KG evidence grounding.

## Paper

The full research paper can be found here: [`FEVER_Camera_Ready.pdf`](FEVER_Camera_Ready.pdf)

---

## Dataset: FACTKG

**FACTKG** ([Kim et al., 2023](https://aclanthology.org/2023.acl-long.895.pdf)) consists of 108,675 natural language claims derived from DBpedia, each paired with:
* **One-hop KG subgraphs**
* **Binary labels** (SUPPORTED / REFUTED)
* **Reasoning types**: Existence, Substitution, Multi-hop, Multi-claim, Negation, and Single-hop.

---

## Experiments & Results

### 1. Symbolic Baselines (Feature Engineering)
* **Approach**: **29 hand-crafted features** spanning graph structure, entity coverage, and semantic relation types.
* **Models**: Logistic Regression, Random Forest, XGBoost.
* **Best Result**: **66.54% accuracy (XGBoost)**.
* **Key Finding**: Symbolic features are competitive on substitution but fail significantly on negation (41.10%) and multi-hop reasoning.

### 2. Neural Encoders (BERT)
* **Approach**: BERT-base fine-tuned on linearized KG subgraphs.
* **Result**: **92.68% test accuracy**.
* **Key Finding**: Token-level neural semantics outperform symbolic features by ~26 points, excelling at negation (91.70%) and existence (98.15%).

### 3. Graph Neural Networks (QA-GNN)
* **Models**: 
    * QA-GNN baseline: 69.64%
    * Cross-attention fusion variant: 69.74%
* **Key Finding**: GNNs lag behind BERT by over 22 points, suggesting that message passing struggles to capture "absence" (negation) compared to token-level attention.

### 4. LLM-Assisted Semantic Filtering
* **Approach**: Use GPT-4.1-mini to select the top $k=10$ most relevant triples to avoid BERT truncation.
* **Results**:
    * **LLM Filtered**: 78.85% accuracy
    * **Heuristic (Jaccard) Control**: 77.54% accuracy
    * **Unfiltered (Truncated)**: 52.70% accuracy
* **Key Finding**: Semantic evidence prioritization yields a consistent gain (+1.31 over heuristics).

### 5. Memorization vs. KG-Grounded LLM Reasoning
* **Setup**: 300 stratified test claims comparing **Memorization** (Claims only) vs. **KG-Grounded** (Claims + Subgraphs + CoT + Citations).

| Model | Memorization | KG-Grounded | Improvement ($\Delta$) |
| :--- | :--- | :--- | :--- |
| GPT-4o-mini | 71.67% | 84.33% | **+12.67** |
| GPT-4.1-mini | 74.67% | 84.00% | **+9.33** |

---

## Main Findings

1.  **Symbolic Ceiling**: Interpretable symbolic features plateau at ~66.54%, failing to model compositional semantics.
2.  **Text Encoding Supremacy**: BERT over linearized text is the current state-of-the-art on FACTKG (92.68%).
3.  **Semantic Filtering**: LLM-curated training data significantly improves the signal-to-noise ratio.
4.  **Attribution & Grounding**: KG-grounding moves models away from "plausibility" guesses toward verifiable triple-based attribution.

---


## Requirements

```bash
torch>=1.10
transformers>=4.30
openai>=1.0
pandas
numpy
scikit-learn
networkx
xgboost
```

---

## Citation

If you use this code or findings, please cite:
```
@inproceedings{upadhyay2026evidence,
  title={Evidence Grounding vs. Memorization: Why Neural Semantics Matter for Knowledge Graph Fact Verification},
  author={Upadhyay, Ankit Kumar and Erickson, John S. and McGuinness, Deborah L.},
  booktitle={Proceedings of the Workshop on Fact Extraction and VERification (FEVER) at EACL},
  year={2026}
}
```

## Acknowledgments

This project builds upon:
- [FACTKG Dataset](https://github.com/jiho283/FactKG) (Kim et al., 2023)
- [Fact or Fiction Repository](https://github.com/Tobias-Opsahl/Fact-or-Fiction) (Opsahl, 2024)

## License

This project is for academic use.
