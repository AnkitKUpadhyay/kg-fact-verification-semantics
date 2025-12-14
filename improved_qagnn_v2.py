"""
Improved QA-GNN with Cross-Attention

Matches the paper's interface exactly but replaces:
- One-way cosine similarity weighting → Bidirectional learned attention
- Simple concatenation → Cross-attention fusion

Their approach:
  1. Cosine similarity: claim → weights graph nodes
  2. GNN processes weighted graph
  3. Concatenate: [graph_emb, claim_emb]

Your approach:
  1. Cross-attention: claim ↔ graph (bidirectional)
  2. GNN processes attended graph  
  3. Cross-attention fusion: learned combination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models import get_bert_model


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between graph and text."""
    
    def __init__(self, graph_dim, text_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.graph_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.text_to_graph_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, graph_emb, text_emb):
        """
        Args:
            graph_emb: [batch_size, graph_dim]
            text_emb: [batch_size, text_dim]
        Returns:
            fused: [batch_size, hidden_dim * 2]
        """
        g = self.graph_proj(graph_emb).unsqueeze(1)
        t = self.text_proj(text_emb).unsqueeze(1)
        
        # Bidirectional attention
        g_att, _ = self.graph_to_text_attn(g, t, t)
        g_att = self.norm1(g_att + g)
        
        t_att, _ = self.text_to_graph_attn(t, g, g)
        t_att = self.norm2(t_att + t)
        
        return torch.cat([g_att.squeeze(1), t_att.squeeze(1)], dim=1)


class ImprovedQAGNN(nn.Module):
    """
    QA-GNN with cross-attention fusion.
    
    Differences from paper's QA-GNN:
    1. Bidirectional cross-attention (vs one-way cosine similarity)
    2. Learned attention fusion (vs simple concatenation)
    3. Optional dual pooling (mean + max)
    """
    
    def __init__(self, model_name, n_gnn_layers=2, gnn_hidden_dim=256, 
                 gnn_out_features=256, lm_layer_features=None,
                 gnn_batch_norm=True, freeze_base_model=False, 
                 freeze_up_to_pooler=True, gnn_dropout=0.3,
                 classifier_dropout=0.2, lm_layer_dropout=0.4, 
                 use_roberta=False, use_dual_pooling=True):
        
        if n_gnn_layers < 2:
            raise ValueError(f"n_gnn_layers must be >= 2, got {n_gnn_layers}")
        
        super().__init__()
        
        self.name = model_name
        self.use_dual_pooling = use_dual_pooling
        
        # BERT (same as theirs)
        self.bert = get_bert_model(
            "bert_" + model_name, 
            include_classifier=False,
            freeze_base_model=freeze_base_model,
            freeze_up_to_pooler=freeze_up_to_pooler,
            use_roberta=use_roberta
        )
        
        # GNN layers (same structure as theirs)
        self.n_gnn_layers = n_gnn_layers
        self.gnn_layers = nn.ModuleList()
        
        self.gnn_layers.append(
            GATConv(self.bert.config.hidden_size, gnn_hidden_dim, dropout=gnn_dropout)
        )
        for _ in range(n_gnn_layers - 2):
            self.gnn_layers.append(
                GATConv(gnn_hidden_dim, gnn_hidden_dim, dropout=gnn_dropout)
            )
        self.gnn_layers.append(
            GATConv(gnn_hidden_dim, gnn_out_features, dropout=gnn_dropout)
        )
        
        # Batch norm (same as theirs)
        self.gnn_batch_norm = gnn_batch_norm
        if gnn_batch_norm:
            self.gnn_batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(gnn_hidden_dim) 
                for _ in range(n_gnn_layers - 1)
            ])
        
        # Optional LM layer (same as theirs)
        claim_dim = self.bert.config.hidden_size
        self.with_lm_layer = False
        if lm_layer_features is not None:
            self.lm_dropout = nn.Dropout(lm_layer_dropout)
            self.lm_layer = nn.Linear(self.bert.config.hidden_size, lm_layer_features)
            claim_dim = lm_layer_features
            self.with_lm_layer = True
        
        # NEW: Cross-attention fusion
        pooled_gnn_dim = gnn_out_features * 2 if use_dual_pooling else gnn_out_features
        self.cross_attention = CrossAttentionFusion(
            graph_dim=pooled_gnn_dim,
            text_dim=claim_dim,
            hidden_dim=256,
            num_heads=4
        )
        
        # Classifier (takes cross-attention output)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(256 * 2, 1)  # *2 from cross-attention concat
    
    def forward(self, claim_tokens, data_graphs):
        """
        Args:
            claim_tokens: Dict with 'input_ids', 'attention_mask', etc.
            data_graphs: PyG Batch object
            
        Returns:
            logits: [batch_size]
        """
        # Encode claim (same as theirs)
        claim_outputs = self.bert(**claim_tokens)
        claim_embeddings = claim_outputs.last_hidden_state[:, 0]  # [CLS]
        
        batch = data_graphs
        
        # NEW: Use cross-attention instead of cosine similarity
        # Their approach: weighted_node_features = batch.x * cosine_similarity(...)
        # Our approach: Let attention learn the weighting during training
        x = batch.x  # Don't pre-weight, let GNN and attention learn
        
        # GNN layers (same as theirs)
        for i in range(self.n_gnn_layers):
            x = self.gnn_layers[i](x, batch.edge_index)
            if self.gnn_batch_norm and i < (self.n_gnn_layers - 1):
                x = self.gnn_batch_norm_layers[i](x)
            x = F.relu(x)
        
        # Pooling (optionally dual)
        if self.use_dual_pooling:
            pooled_mean = global_mean_pool(x, batch.batch)
            pooled_max = torch.scatter_reduce(
                torch.zeros(batch.batch.max().item() + 1, x.size(1), 
                           device=x.device),
                0,
                batch.batch.unsqueeze(1).expand(-1, x.size(1)),
                x,
                reduce='amax',
                include_self=False
            )
            pooled_gnn_output = torch.cat([pooled_mean, pooled_max], dim=1)
        else:
            pooled_gnn_output = global_mean_pool(x, batch.batch)
        
        # Optional LM layer (same as theirs)
        if self.with_lm_layer:
            claim_embeddings = self.lm_dropout(claim_embeddings)
            claim_embeddings = self.lm_layer(claim_embeddings)
        
        # NEW: Cross-attention fusion instead of simple concatenation
        # Their: combined = torch.cat((pooled_gnn_output, claim_embeddings), dim=1)
        # Ours: combined = cross_attention(pooled_gnn_output, claim_embeddings)
        combined_features = self.cross_attention(pooled_gnn_output, claim_embeddings)
        
        # Classifier (same structure as theirs)
        combined_features = self.classifier_dropout(combined_features)
        out = self.classifier(combined_features)
        
        return out.squeeze(1)


# Alias for drop-in replacement
QAGNN = ImprovedQAGNN


if __name__ == "__main__":
    print("="*60)
    print("Improved QA-GNN with Cross-Attention")
    print("="*60)
    print("\nKey Changes from Paper's QA-GNN:")
    print("1. Bidirectional cross-attention (vs cosine similarity)")
    print("2. Learned attention fusion (vs concatenation)")
    print("3. Optional dual pooling (mean + max)")
    print("\nTo use:")
    print("1. Copy this file to Fact-or-Fiction/")
    print("2. In run_stuff.py, add:")
    print("   from improved_qagnn import ImprovedQAGNN")
    print("3. Replace QAGNN with ImprovedQAGNN")
    print("="*60)