# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv, global_mean_pool
# from models import get_bert_model
# from utils import get_logger

# logger = get_logger(__name__)

# class StabilizedFusionBlock(nn.Module):
#     """
#     Stabilized Fusion Block.
    
#     Improvement: uses a Learnable Gate initialized to near-ZERO.
#     This ensures that at Epoch 0, the noise from the GNN does NOT 
#     corrupt the Text embeddings. The model starts as a clean baseline 
#     and slowly 'turns on' the fusion as it learns.
#     """
#     def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.gat = GATConv(hidden_dim, hidden_dim, dropout=dropout)
#         self.norm_graph = nn.LayerNorm(hidden_dim)
        
#         # Fusion Mechanism
#         self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
#         self.norm_fusion = nn.LayerNorm(hidden_dim)
        
#         # GATE: Learnable parameter to control flow
#         # Linear layer to decide how much to fuse
#         self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.sigmoid = nn.Sigmoid()

#         # CRITICAL: Initialize gate output weights to zero
#         # This ensures the model starts with Gate close to 0.5 or tuned low, 
#         # preventing shock at initialization.
#         nn.init.zeros_(self.gate_linear.weight)
#         nn.init.zeros_(self.gate_linear.bias)

#     def forward(self, x, edge_index, text_context):
#         # 1. Graph Update (Standard GNN)
#         residual = x
#         x_gnn = self.gat(x, edge_index)
#         x_gnn = F.relu(x_gnn)
#         x = self.norm_graph(residual + x_gnn)
        
#         # 2. Interaction (Deep Fusion)
#         # text_context is [Batch, Dim] mapped to nodes [Total_Nodes, Dim]
        
#         # Compute Fusion Candidate
#         # Graph Nodes attend to Text Context (Simulated by direct interaction here for efficiency)
#         fusion_signal = text_context 
        
#         # 3. Gated Update
#         # Compute how much of the text signal to let in based on current node state
#         gate_input = torch.cat([x, fusion_signal], dim=-1)
#         gate_value = self.sigmoid(self.gate_linear(gate_input))
        
#         # Result: Mostly Node State + Scaled Text Context
#         x_fused = (1 - gate_value) * x + gate_value * fusion_signal
        
#         return self.norm_fusion(x_fused)

# class IterativeFusionGNN_V3(nn.Module):
#     """
#     Stabilized SOTA GNN.
#     Fixes the 'Modal Collapse' issue by protecting the BERT signal.
#     """
#     def __init__(self, model_name, n_gnn_layers=2, gnn_hidden_dim=256, gnn_out_features=256,
#                  lm_layer_features=None, gnn_batch_norm=True, freeze_base_model=False,
#                  freeze_up_to_pooler=True, gnn_dropout=0.3, classifier_dropout=0.2,
#                  lm_layer_dropout=0.4, use_roberta=False):
#         super().__init__()
        
#         self.name = "iterative_fusion_v3_" + model_name
        
#         # Text Encoder
#         self.bert = get_bert_model("bert_" + model_name, include_classifier=False, 
#                                    freeze_base_model=freeze_base_model,
#                                    freeze_up_to_pooler=freeze_up_to_pooler, 
#                                    use_roberta=use_roberta)
#         self.hidden_size = self.bert.config.hidden_size
        
#         # Projections
#         self.node_proj = nn.Linear(self.hidden_size, gnn_hidden_dim)
#         self.text_proj = nn.Linear(self.hidden_size, gnn_hidden_dim)
        
#         # Iterative Layers
#         self.layers = nn.ModuleList()
#         for _ in range(n_gnn_layers):
#             self.layers.append(StabilizedFusionBlock(gnn_hidden_dim, dropout=gnn_dropout))
            
#         # Classifier
#         # IMPORTANT: We concat the Original BERT CLS at the end (Global Skip Connection)
#         # This guarantees the model performs AT LEAST as well as BERT.
#         clf_input = gnn_hidden_dim + self.hidden_size
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(classifier_dropout),
#             nn.Linear(clf_input, 1)
#         )

#     def forward(self, claim_tokens, data_graphs):
#         # A. Encode Text
#         bert_output = self.bert(**claim_tokens)
#         text_cls = bert_output.last_hidden_state[:, 0, :] # [Batch, 768]
#         text_ctx = self.text_proj(text_cls) # [Batch, 256]
        
#         # B. Init Nodes
#         batch = data_graphs
#         # Init graph with text context
#         x = text_cls[batch.batch] 
#         x = self.node_proj(x) # [Nodes, 256]
        
#         # C. Iterative Fusion
#         for layer in self.layers:
#             # Map batch text to nodes
#             node_text_ctx = text_ctx[batch.batch] 
#             x = layer(x, batch.edge_index, node_text_ctx)
            
#         # D. Final Prediction
#         x_pool = global_mean_pool(x, batch.batch) # [Batch, 256]
        
#         # GLOBAL SKIP CONNECTION
#         # Concatenate processed graph (x_pool) with pristine BERT signal (text_cls)
#         combined = torch.cat([x_pool, text_cls], dim=1) 
        
#         logits = self.classifier(combined)
#         return logits.squeeze(1)


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, global_mean_pool
from models import get_bert_model
from utils import get_logger

logger = get_logger(__name__)


def global_add_pool(x: torch.Tensor, batch: torch.Tensor, size: int = None) -> torch.Tensor:
    """
    Pure PyTorch implementation of global_add_pool (sum over nodes per graph).

    Args:
        x:     [N, F] node features
        batch: [N] graph indices in [0, num_graphs-1]
        size:  optional number of graphs (rows in output)

    Returns:
        out: [num_graphs, F]
    """
    if batch.numel() == 0:
        # no nodes -> empty tensor
        return x.new_zeros((0, x.size(1)))

    if size is None:
        size = int(batch.max().item()) + 1

    out = x.new_zeros(size, x.size(1))
    index = batch.view(-1, 1).expand(-1, x.size(1))  # [N, F]
    out.scatter_add_(0, index, x)
    return out


class GraphTransformerQAGNN(nn.Module):
    """
    Graph Transformer QA-GNN.

    Key features:
      - Uses precomputed node embeddings from the graph (batch.x),
        reweighted by claim relevance (cosine similarity).
      - Replaces GATConv with TransformerConv, with residual + LayerNorm.
      - Dual pooling: structural (mean) + semantic (claim-attention) pooling.
      - Optional LM projection layer for the claim, like original QAGNN.
    """

    def __init__(
        self,
        model_name: str,
        n_gnn_layers: int = 2,
        gnn_hidden_dim: int = 256,
        gnn_out_features: int = 256,
        lm_layer_features: int = None,
        gnn_batch_norm: bool = True,  # kept for API compatibility, not used
        freeze_base_model: bool = False,
        freeze_up_to_pooler: bool = True,
        gnn_dropout: float = 0.3,
        classifier_dropout: float = 0.2,
        lm_layer_dropout: float = 0.4,
        use_roberta: bool = False,
    ):
        super().__init__()

        if n_gnn_layers < 2:
            raise ValueError(f"`n_gnn_layers` must be at least 2, got {n_gnn_layers}.")

        self.name = "graph_transformer_" + model_name
        self.n_gnn_layers = n_gnn_layers
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_out_features = gnn_out_features

        # 1. Text encoder (BERT / RoBERTa) – no classifier head
        self.bert = get_bert_model(
            "bert_" + model_name,
            include_classifier=False,
            freeze_base_model=freeze_base_model,
            freeze_up_to_pooler=freeze_up_to_pooler,
            use_roberta=use_roberta,
        )
        self.hidden_size = self.bert.config.hidden_size

        # 2. Project node features from LM-dim to GNN hidden dim
        self.node_proj = nn.Linear(self.hidden_size, gnn_hidden_dim)

        # 3. Graph Transformer stack (TransformerConv)
        self.gnn_layers = nn.ModuleList()

        # First (n_gnn_layers - 1) layers: hidden_dim -> hidden_dim
        self.gnn_layers.append(
            TransformerConv(
                in_channels=gnn_hidden_dim,
                out_channels=gnn_hidden_dim,
                heads=4,
                concat=False,
                dropout=gnn_dropout,
            )
        )
        for _ in range(n_gnn_layers - 2):
            self.gnn_layers.append(
                TransformerConv(
                    in_channels=gnn_hidden_dim,
                    out_channels=gnn_hidden_dim,
                    heads=4,
                    concat=False,
                    dropout=gnn_dropout,
                )
            )

        # Last layer: hidden_dim -> out_features
        self.gnn_layers.append(
            TransformerConv(
                in_channels=gnn_hidden_dim,
                out_channels=gnn_out_features,
                heads=4,
                concat=False,
                dropout=gnn_dropout,
            )
        )

        # Norms for intermediate layers (+ residual), and final norm
        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(gnn_hidden_dim) for _ in range(n_gnn_layers - 1)]
        )
        self.gnn_last_norm = nn.LayerNorm(gnn_out_features)

        # 4. Optional LM projection layer for the claim (like original QAGNN)
        self.with_lm_layer = False
        claim_dim = self.hidden_size
        if lm_layer_features is not None:
            self.with_lm_layer = True
            self.lm_dropout = nn.Dropout(lm_layer_dropout)
            self.lm_layer = nn.Linear(self.hidden_size, lm_layer_features)
            claim_dim = lm_layer_features

        # 5. Attention projection for semantic pooling
        # We compare x (gnn_out_features) with text_context_proj (gnn_hidden_dim),
        # so we map x into that space if needed.
        if gnn_out_features != gnn_hidden_dim:
            self.attn_proj = nn.Linear(gnn_out_features, gnn_hidden_dim)
        else:
            self.attn_proj = nn.Identity()

        # 6. Classifier: structural_pool (Dg) + semantic_pool (Dg) + claim (Dc)
        graph_dim = 2 * gnn_out_features
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(graph_dim + claim_dim, 1)

        logger.info(
            f"Initialized GraphTransformerQAGNN with "
            f"{n_gnn_layers} layers, hidden_dim={gnn_hidden_dim}, out_features={gnn_out_features}."
        )

    def forward(self, claim_tokens, data_graphs):
        """
        Args:
            claim_tokens: dict with input_ids, attention_mask, etc. for BERT
            data_graphs:  torch_geometric.data.Batch with fields:
                          - x: node embeddings [N, hidden_size]
                          - edge_index: [2, E]
                          - batch: [N] graph indices
        Returns:
            logits: [batch_size] (BCEWithLogitsLoss-ready)
        """
        # A. Encode claim
        bert_output = self.bert(**claim_tokens)
        text_cls = bert_output.last_hidden_state[:, 0, :]  # [B, hidden_size]

        batch = data_graphs

        # B. Node features from graph (entity embeddings)
        if not hasattr(batch, "x") or batch.x is None:
            raise ValueError(
                "GraphTransformerQAGNN expects `data_graphs.x` "
                "to contain node features."
            )

        node_features = batch.x  # [N, hidden_size]

        # C. Claim-aware reweighting of node features (cosine similarity)
        text_expanded = text_cls[batch.batch]  # [N, hidden_size]
        relevance = F.cosine_similarity(text_expanded, node_features, dim=-1).unsqueeze(-1)  # [N, 1]
        node_features = node_features * relevance  # [N, hidden_size]

        # Project to GNN hidden dim
        x = self.node_proj(node_features)  # [N, gnn_hidden_dim]

        # D. Graph Transformer stack with residual + LayerNorm
        for i, conv in enumerate(self.gnn_layers):
            h = conv(x, batch.edge_index)
            if i < self.n_gnn_layers - 1:
                # Residual connection in hidden_dim space
                h = self.gnn_norms[i](h + x)
            else:
                # Last layer: out_features, no residual from hidden_dim
                h = self.gnn_last_norm(h)
            x = F.relu(h)

        # Now x has shape [N, gnn_out_features]

        # E. Dual pooling

        # 1) Structural pooling: plain mean over nodes
        pool_struct = global_mean_pool(x, batch.batch)  # [B, gnn_out_features]

        # 2) Semantic pooling: attention-weighted by claim
        text_context = text_cls[batch.batch]  # [N, hidden_size]
        text_context_proj = self.node_proj(text_context)  # [N, gnn_hidden_dim]

        x_for_attn = self.attn_proj(x)  # [N, gnn_hidden_dim]
        scores = (x_for_attn * text_context_proj).sum(dim=-1)  # [N]

        scores_exp = scores.exp()
        denom = global_add_pool(scores_exp.unsqueeze(-1), batch.batch).squeeze(-1)  # [B]
        attention_weights = scores_exp / (denom[batch.batch] + 1e-8)  # [N]

        pool_semantic = global_add_pool(x * attention_weights.unsqueeze(-1), batch.batch)  # [B, gnn_out_features]

        # F. Optional LM projection on claim side
        claim_repr = text_cls
        if self.with_lm_layer:
            claim_repr = self.lm_dropout(claim_repr)
            claim_repr = self.lm_layer(claim_repr)

        # G. Final classifier
        graph_repr = torch.cat([pool_struct, pool_semantic], dim=-1)  # [B, 2 * gnn_out_features]
        combined = torch.cat([graph_repr, claim_repr], dim=-1)        # [B, 2Dg + Dc]

        combined = self.classifier_dropout(combined)
        logits = self.classifier(combined)  # [B, 1]

        return logits.squeeze(1)
