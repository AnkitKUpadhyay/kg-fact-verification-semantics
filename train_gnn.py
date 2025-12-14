"""
Task 3: Train GNN Neural Baseline
- Builds vocabs from all (s,p,o) triples
- Creates a custom PyTorch Geometric Dataset
- Trains a Graph Attention Network (GAT) that uses edge features
- Compares GNN accuracy to the classical model baseline
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import warnings
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Module, Embedding, Linear, Sequential, ReLU, Dropout
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
DATA_DIR = 'data'
SUBGRAPH_DIR = os.path.join(DATA_DIR, 'subgraphs')
VOCAB_DIR = 'models' # Save vocabs with other models
MODEL_SAVE_PATH = os.path.join(VOCAB_DIR, 'gnn_model.pt')

# GNN Hyperparameters
HIDDEN_DIM = 64
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10
# ---

def build_vocabs():
    """
    Runs once. Scans all train/val/test subgraph files to build
    complete entity and relation vocabularies.
    """
    print("Building vocabularies (one-time setup)...")
    ent_vocab_path = os.path.join(VOCAB_DIR, 'entity_to_id.json')
    rel_vocab_path = os.path.join(VOCAB_DIR, 'relation_to_id.json')
    
    if os.path.exists(ent_vocab_path) and os.path.exists(rel_vocab_path):
        print("Vocabularies already exist. Loading them.")
        with open(ent_vocab_path, 'r') as f:
            ent2id = json.load(f)
        with open(rel_vocab_path, 'r') as f:
            rel2id = json.load(f)
        return ent2id, rel2id

    os.makedirs(VOCAB_DIR, exist_ok=True)
    all_entities = set()
    all_relations = set()
    
    files_to_scan = [
        os.path.join(SUBGRAPH_DIR, 'subgraphs_one_hop_train.pkl'),
        os.path.join(SUBGRAPH_DIR, 'subgraphs_one_hop_dev.pkl'),
        os.path.join(SUBGRAPH_DIR, 'subgraphs_one_hop_test.pkl')
    ]
    
    for f_path in files_to_scan:
        print(f"Scanning {f_path}...")
        df = pd.read_pickle(f_path)
        for walked_dict in tqdm(df['walked'], desc="Scanning triples"):
            if isinstance(walked_dict, dict):
                triples = walked_dict.get('walkable', [])
                for s, p, o in triples:
                    all_entities.add(s)
                    all_entities.add(o)
                    all_relations.add(p)
                    
    # Create mappings, adding <UNK> token at index 0
    ent2id = {ent: i+1 for i, ent in enumerate(all_entities)}
    ent2id['<UNK>'] = 0
    
    rel2id = {rel: i+1 for i, rel in enumerate(all_relations)}
    rel2id['<UNK>'] = 0
    
    # Save vocabs
    with open(ent_vocab_path, 'w') as f:
        json.dump(ent2id, f)
    with open(rel_vocab_path, 'w') as f:
        json.dump(rel2id, f)
        
    print(f"Vocabs saved! {len(ent2id)} entities, {len(rel2id)} relations.")
    return ent2id, rel2id


class FactKGDataset(Dataset):
    """Custom PyTorch Geometric Dataset."""
    def __init__(self, split, data_dir, subgraph_dir, ent2id, rel2id, transform=None):
        self.split = split
        self.data_dir = data_dir
        self.subgraph_dir = subgraph_dir
        self.ent2id = ent2id
        self.rel2id = rel2id
        
        # Load the raw data into lists
        claims_path = os.path.join(self.data_dir, f"factkg/factkg_{split}.pickle")
        subgraph_path = os.path.join(self.subgraph_dir, f"subgraphs_one_hop_{split}.pkl")
        
        with open(claims_path, 'rb') as f:
            self.claims_dict = pickle.load(f)
        self.subgraphs_df = pd.read_pickle(subgraph_path)
        
        # Convert to lists for fast __getitem__ access
        self.claims_list = list(self.claims_dict.values())
        self.subgraph_rows = [row for _, row in self.subgraphs_df.iterrows()]
        
        assert len(self.claims_list) == len(self.subgraph_rows), "Data mismatch!"
        
        super().__init__(root=None, transform=transform)

    def len(self):
        return len(self.claims_list)

    def get(self, idx):
        """Builds a single Data object for a given index."""
        
        # 1. Get metadata and label
        metadata = self.claims_list[idx]
        y = torch.tensor([metadata['Label'][0]], dtype=torch.float)
        
        # 2. Get triples
        walked_dict = self.subgraph_rows[idx]['walked']
        triples_list = []
        if isinstance(walked_dict, dict):
            triples_list = walked_dict.get('walkable', [])
            
        # 3. Handle empty graphs
        if not triples_list:
            # Create a graph with a single <UNK> node
            x = torch.tensor([self.ent2id['<UNK>']], dtype=torch.long)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            
        # 4. Build graph for non-empty subgraphs
        node_str_to_local_id = {} # Map 'John_E._Beck' -> 0
        local_id_to_global_id = []  # Map 0 -> 5432 (vocab id)
        
        edge_index_list = []
        edge_attr_list = []
        
        def get_local_id(node_str):
            if node_str not in node_str_to_local_id:
                local_id = len(node_str_to_local_id)
                node_str_to_local_id[node_str] = local_id
                global_id = self.ent2id.get(node_str, self.ent2id['<UNK>'])
                local_id_to_global_id.append(global_id)
            return node_str_to_local_id[node_str]

        for s, p, o in triples_list:
            s_id = get_local_id(s)
            o_id = get_local_id(o)
            p_id = self.rel2id.get(p, self.rel2id['<UNK>'])
            
            edge_index_list.append([s_id, o_id])
            edge_attr_list.append(p_id)
            
        # 5. Convert to Tensors
        x = torch.tensor(local_id_to_global_id, dtype=torch.long)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class GNNModel(Module):
    """
    A Graph Attention Network (GAT) that uses both node and edge embeddings.
    """
    def __init__(self, num_entities, num_relations, hidden_dim):
        super().__init__()
        
        # Embedding layers
        self.node_emb = Embedding(num_entities, hidden_dim)
        self.rel_emb = Embedding(num_relations, hidden_dim)
        
        # GNN layers
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=4, edge_dim=hidden_dim, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=4, edge_dim=hidden_dim, concat=False)
        
        # Readout and Classifier
        self.pool = global_mean_pool
        self.classifier = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(0.5),
            Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        # 1. Get initial embeddings
        # x shape: [num_nodes_in_batch, hidden_dim]
        x = self.node_emb(data.x)
        # edge_attr shape: [num_edges_in_batch, hidden_dim]
        edge_attr = self.rel_emb(data.edge_attr)
        
        # 2. GNN layers
        x = self.conv1(x, data.edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, data.edge_index, edge_attr)
        x = F.elu(x)
        
        # 3. Readout (Aggregate node features to get a graph-level feature)
        # x_pool shape: [batch_size, hidden_dim]
        x_pool = self.pool(x, data.batch)
        
        # 4. Classify
        out = self.classifier(x_pool)
        return out


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(loader, desc="Validating", leave=False):
        batch = batch.to(device)
        out = model(batch)
        
        preds = (torch.sigmoid(out.squeeze()) > 0.5).cpu().numpy()
        labels = batch.y.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        
    return accuracy_score(all_labels, all_preds)

# --- Main Execution ---

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    # --- 1. Build or Load Vocabs ---
    ent2id, rel2id = build_vocabs()
    num_entities = len(ent2id)
    num_relations = len(rel2id)
    
    print(f"Loaded {num_entities} entities and {num_relations} relations.")

    # --- 2. Setup Datasets and Loaders ---
    print("Loading datasets...")
    train_dataset = FactKGDataset("train", DATA_DIR, SUBGRAPH_DIR, ent2id, rel2id)
    val_dataset = FactKGDataset("dev", DATA_DIR, SUBGRAPH_DIR, ent2id, rel2id)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("Datasets ready.")

    # --- 3. Initialize Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = GNNModel(
        num_entities=num_entities,
        num_relations=num_relations,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Binary Cross-Entropy loss, good for 0/1 classification
    loss_fn = torch.nn.BCEWithLogitsLoss() 
    
    # --- 4. Training Loop ---
    print("\nStarting GNN Training...")
    best_val_acc = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🎉 New best model saved to {MODEL_SAVE_PATH} (Acc: {best_val_acc*100:.2f}%)")

    print("\n" + "="*60)
    print("GNN Training Complete")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Classical Model Baseline: 63.96%")
    print("="*60)