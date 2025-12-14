"""
Feature Extractor for FactKG Subgraphs
Novel contribution: First interpretable feature-based approach for FactKG
"""

import numpy as np
import pandas as pd
import pickle
import re
from collections import Counter
import networkx as nx
from scipy.stats import entropy
from tqdm import tqdm  # For a nice progress bar
import os
import warnings

class SubgraphFeatureExtractor:
    """
    Extracts interpretable features from FactKG subgraphs (using the 'walked' column)
    for classical ML models.
    """
    
    def __init__(self):
        """
        Defines relation categories for semantic features, including inverses.
        """
        
        # Helper to automatically add inverse relations (e.g., '~successor')
        def add_inverses(rel_set):
            # Also add the lowercased version for matching
            rel_set = {r.lower() for r in rel_set}
            return rel_set.union({f"~{r}" for r in rel_set})

        self.temporal_relations = add_inverses({
            'birthdate', 'deathdate', 'birthyear', 'deathyear', 
            'activeyearsstartyear', 'activeyearsendyear', 'date'
        })
        
        self.location_relations = add_inverses({
            'location', 'country', 'city', 'birthplace', 'deathplace',
            'headquarters', 'ground', 'locatedinarea'
        })
        
        self.person_relations = add_inverses({
            'spouse', 'child', 'parent', 'successor', 'predecessor',
            'relative', 'partner'
        })
        
        self.organizational_relations = add_inverses({
            'team', 'club', 'company', 'employer', 'affiliation',
            'league', 'division'
        })

    def extract_features(self, claim, claim_entities, evidence_dict, walked_dict, label):
        """
        Extracts all features for a single claim-subgraph pair.
        
        Args:
            claim (str): The claim text
            claim_entities (list): List of entities mentioned in claim
            evidence_dict (dict): The ground-truth evidence from metadata
            walked_dict (dict): The dictionary from the 'walked' column 
                                (e.g., {'walkable': [('s', 'p', 'o'), ...]})
            label (bool): Ground truth label
            
        Returns:
            dict: A dictionary of feature names to values
        """
        features = {}
        
        # --- Get the *actual* list of triples ---
        triples_list = []
        if isinstance(walked_dict, dict) and 'walkable' in walked_dict:
            triples_list = walked_dict['walkable']
        
        # --- Get ground-truth evidence triples (s, p) ---
        ground_truth_evidence = []
        if isinstance(evidence_dict, dict):
            for entity, relations in evidence_dict.items():
                for rel_list in relations:
                    if rel_list:
                        ground_truth_evidence.append((entity, rel_list[0]))

        # Basic metadata
        features['claim_length'] = len(claim.split())
        features['num_claim_entities'] = len(claim_entities)
        
        # Pass the correct data (triples_list) to the sub-methods
        structure_features = self._extract_graph_structure(triples_list)
        features.update(structure_features)
        
        matching_features = self._extract_claim_matching(
            claim, set(claim_entities), triples_list, set(ground_truth_evidence)
        )
        features.update(matching_features)
        
        semantic_features = self._extract_semantic_features(triples_list)
        features.update(semantic_features)
        
        # Label (for convenience)
        features['label'] = label
        
        return features

    def _extract_graph_structure(self, triples_list):
        """
        Extracts topological features from the list of (s, p, o) triples.
        """
        features = {}
        
        if not triples_list or len(triples_list) == 0:
            # Return all features as 0 or 0.0
            return {
                'num_nodes': 0, 'num_edges': 0, 'graph_density': 0.0,
                'avg_degree': 0.0, 'max_degree': 0, 'min_degree': 0,
                'num_connected_components': 0, 'has_cycle': 0,
                'degree_std': 0.0
            }

        # --- 1. Build the REAL graph ---
        nodes = set()
        g = nx.DiGraph() # Use a Directed Graph

        for s, p, o in triples_list:
            nodes.add(s)
            nodes.add(o)
            g.add_edge(s, o, relation=p)
            
        # --- 2. Calculate Features ---
        num_nodes = len(nodes)
        num_edges = len(triples_list)

        features['num_nodes'] = num_nodes
        features['num_edges'] = num_edges

        # Graph density
        features['graph_density'] = nx.density(g) if num_nodes > 1 else 0.0

        # Degree statistics (using total degree: in + out)
        degrees = [d for n, d in g.degree()]
        features['avg_degree'] = np.mean(degrees) if degrees else 0.0
        features['max_degree'] = max(degrees) if degrees else 0
        features['min_degree'] = min(degrees) if degrees else 0
        features['degree_std'] = np.std(degrees) if len(degrees) > 1 else 0.0

        # Connectivity (use weakly connected for DiGraph)
        features['num_connected_components'] = nx.number_weakly_connected_components(g)
        
        # Check for cycles (and convert bool to int)
        try:
            features['has_cycle'] = int(not nx.is_directed_acyclic_graph(g))
        except:
            # This can fail if graph has self-loops, which count as cycles
            features['has_cycle'] = 1 

        return features

    def _extract_claim_matching(self, claim, claim_entities_set, triples_list, ground_truth_evidence_set):
        """Extract features about how well subgraph matches the claim"""
        features = {}
        
        if not triples_list:
            return {
                'entity_coverage': 0.0, 'entities_in_subgraph': 0,
                'entities_not_in_subgraph': len(claim_entities_set),
                'avg_edges_per_claim_entity': 0.0,
                'max_edges_for_claim_entity': 0,
                'claim_words_in_relations': 0,
                'claim_relation_overlap': 0.0,
                'exact_evidence_match': 0, # Explicitly 0
                'evidence_overlap_jaccard': 0.0
            }

        # Get all nodes and relations from the subgraph
        subgraph_nodes = set()
        subgraph_relations = []
        subgraph_relation_pairs = set() # For Jaccard
        
        edges_per_claim_entity = {ent: 0 for ent in claim_entities_set}

        for s, p, o in triples_list:
            subgraph_nodes.add(s)
            subgraph_nodes.add(o)
            subgraph_relations.append(p.lower())
            subgraph_relation_pairs.add((s, p))
            
            # Count outgoing edges from claim entities
            if s in edges_per_claim_entity:
                edges_per_claim_entity[s] += 1
        
        # --- Feature 7: entity_coverage ---
        entities_in_subgraph = len(claim_entities_set.intersection(subgraph_nodes))
        features['entity_coverage'] = entities_in_subgraph / len(claim_entities_set) if claim_entities_set else 0.0
        features['entities_in_subgraph'] = entities_in_subgraph
        features['entities_not_in_subgraph'] = len(claim_entities_set) - entities_in_subgraph
        
        # --- Feature: Average edges per claim entity ---
        edge_counts = list(edges_per_claim_entity.values())
        features['avg_edges_per_claim_entity'] = np.mean(edge_counts) if edge_counts else 0.0
        features['max_edges_for_claim_entity'] = max(edge_counts) if edge_counts else 0

        # --- Feature: Check claim word overlap with relations ---
        claim_words = set(claim.lower().split())
        relation_words = set()
        for rel in subgraph_relations:
            # Split camelCase and remove inverse tilde: ~birthDate -> birth date
            words = re.findall('[A-Z][a-z]*|[a-z]+', rel.replace('~', ''))
            relation_words.update(w.lower() for w in words)
        
        overlap = claim_words.intersection(relation_words)
        features['claim_words_in_relations'] = len(overlap)
        features['claim_relation_overlap'] = len(overlap) / len(claim_words) if claim_words else 0.0
        
        # --- Feature 8: exact_evidence_match (Crucial Feature) ---
        evidence_match = 0
        for s_truth, p_truth in ground_truth_evidence_set:
            # Check for direct match (e.g., successor)
            if (s_truth, p_truth) in subgraph_relation_pairs:
                evidence_match = 1
                break
            # Check for inverse match (e.g., ~successor)
            if (s_truth, f"~{p_truth}") in subgraph_relation_pairs:
                evidence_match = 1
                break
        features['exact_evidence_match'] = evidence_match # Will be 0 or 1
        
        # --- Feature 12: evidence_overlap_jaccard ---
        ground_truth_roots = set((s, p.replace('~', '')) for s, p in ground_truth_evidence_set)
        subgraph_roots = set((s, p.replace('~', '')) for s, p in subgraph_relation_pairs)
        
        intersection = len(ground_truth_roots.intersection(subgraph_roots))
        union = len(ground_truth_roots.union(subgraph_roots))
        features['evidence_overlap_jaccard'] = intersection / union if union > 0 else 0.0
        
        return features

    def _extract_semantic_features(self, triples_list):
        """Extracts semantic/domain-specific features from triples list"""
        features = {}
        
        # Get all predicates, lowercased
        all_relations = [p.lower() for s, p, o in triples_list] 
        
        if not all_relations:
            # Return all features as 0 or 0.0
            return {
                'has_temporal_relations': 0, 'num_temporal_relations': 0,
                'has_location_relations': 0, 'num_location_relations': 0,
                'has_person_relations': 0, 'num_person_relations': 0,
                'has_organizational_relations': 0, 'num_organizational_relations': 0,
                'relation_type_diversity': 0.0, 'most_common_relation_freq': 0.0,
                'unique_relation_types': 0, 'inverse_relation_ratio': 0.0
            }

        # --- Feature 13-15: Count relation categories (using exact match) ---
        temporal_count = sum(1 for r in all_relations if r in self.temporal_relations)
        location_count = sum(1 for r in all_relations if r in self.location_relations)
        person_count = sum(1 for r in all_relations if r in self.person_relations)
        org_count = sum(1 for r in all_relations if r in self.organizational_relations)
        
        # Convert booleans to int (0 or 1)
        features['has_temporal_relations'] = int(temporal_count > 0)
        features['num_temporal_relations'] = temporal_count
        features['has_location_relations'] = int(location_count > 0)
        features['num_location_relations'] = location_count
        features['has_person_relations'] = int(person_count > 0)
        features['num_person_relations'] = person_count
        features['has_organizational_relations'] = int(org_count > 0)
        features['num_organizational_relations'] = org_count
        
        # --- Feature 16: relation_type_diversity (Shannon entropy) ---
        relation_counts = Counter(all_relations)
        total = len(all_relations)
        probs = [count / total for count in relation_counts.values()]
        features['relation_type_diversity'] = entropy(probs, base=2)
        
        # --- Feature 17 (from proposal): most_common_relation_freq ---
        most_common_count = relation_counts.most_common(1)[0][1] if relation_counts else 0
        features['most_common_relation_freq'] = most_common_count / total if total > 0 else 0.0
        features['unique_relation_types'] = len(relation_counts)
        
        # --- NEW Feature: Inverse Relation Ratio ---
        inverse_count = sum(1 for r in all_relations if r.startswith('~'))
        features['inverse_relation_ratio'] = inverse_count / total if total > 0 else 0.0
        
        return features

    def get_feature_names(self):
        """Returns a list of all feature names in a consistent order."""
        return [
            # Metadata
            'claim_length', 'num_claim_entities',
            # Graph structure
            'num_nodes', 'num_edges', 'graph_density',
            'avg_degree', 'max_degree', 'min_degree', 'degree_std',
            'num_connected_components', 'has_cycle',
            # Claim matching
            'entity_coverage', 'entities_in_subgraph', 'entities_not_in_subgraph',
            'avg_edges_per_claim_entity', 'max_edges_for_claim_entity',
            'claim_words_in_relations', 'claim_relation_overlap',
            'exact_evidence_match', 'evidence_overlap_jaccard',
            # Semantic
            'has_temporal_relations', 'num_temporal_relations',
            'has_location_relations', 'num_location_relations',
            'has_person_relations', 'num_person_relations',
            'has_organizational_relations', 'num_organizational_relations',
            'relation_type_diversity', 'most_common_relation_freq',
            'unique_relation_types', 'inverse_relation_ratio',
            # Label
            'label'
        ]

# --- Main script to run the feature extraction ---

def process_and_save(dataset_type, extractor, data_dir, save_dir):
    """Loads raw data, processes features, and saves to a CSV file."""
    
    print(f"\n--- Processing {dataset_type} set ---")
    
    # 1. Load claims data
    claims_path = os.path.join(data_dir, f"factkg/factkg_{dataset_type}.pickle")
    with open(claims_path, 'rb') as f:
        claims_dict = pickle.load(f)

    # 2. Load subgraph data
    subgraph_path = os.path.join(data_dir, f"subgraphs/subgraphs_one_hop_{dataset_type}.pkl")
    with open(subgraph_path, 'rb') as f:
        subgraphs_df = pickle.load(f)
        
    print(f"Loaded {len(claims_dict)} claims and {len(subgraphs_df)} subgraphs.")
    
    # 3. ADDED: Robust safety check
    assert len(claims_dict) == len(subgraphs_df), \
        f"❌ Data length mismatch! {len(claims_dict)} claims vs {len(subgraphs_df)} subgraphs"
    print("✅ Data lengths match - proceeding with feature extraction")
        
    all_features = []
    
    # Use tqdm for a progress bar
    for (claim_text, metadata), (_, subgraph_row) in tqdm(
        zip(claims_dict.items(), subgraphs_df.iterrows()), 
        total=len(claims_dict),
        desc=f"Extracting features for {dataset_type}"
    ):
        features = extractor.extract_features(
            claim=claim_text,
            claim_entities=metadata.get('Entity_set', []),
            evidence_dict=metadata.get('Evidence', {}),
            walked_dict=subgraph_row['walked'],
            label=metadata['Label'][0]
        )
        all_features.append(features)
        
    # 4. Create the final DataFrame
    features_df = pd.DataFrame(all_features)
    # Reorder columns to match get_feature_names()
    features_df = features_df[extractor.get_feature_names()]
    
    print(f"Feature extraction complete. Shape: {features_df.shape}")
    
    # 5. Save to file
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"features_one_hop_{dataset_type}.csv")
    features_df.to_csv(save_path, index=False)
    print(f"Successfully saved features to {save_path}")

if __name__ == "__main__":
    # --- Configuration ---
    DATA_DIR = '/users/PAS2136/upadha2/factKG/Fact-or-Fiction/data'
    SAVE_DIR = '/users/PAS2136/upadha2/factKG/Fact-or-Fiction/features'
    # ---
    
    warnings.filterwarnings('ignore')
    
    # Initialize the extractor one time
    feature_extractor = SubgraphFeatureExtractor()
    
    # Process train, validation, and test sets
    process_and_save("train", feature_extractor, DATA_DIR, SAVE_DIR)
    process_and_save("dev", feature_extractor, DATA_DIR, SAVE_DIR)
    process_and_save("test", feature_extractor, DATA_DIR, SAVE_DIR)
    
    print("\nAll processing complete!")