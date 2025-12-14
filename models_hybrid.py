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
        return x.new_zeros((0, x.size(1)))

    if size is None:
        size = int(batch.max().item()) + 1

    out = x.new_zeros(size, x.size(1))
    index = batch.view(-1, 1).expand(-1, x.size(1))  # [N, F]
    out.scatter_add_(0, index, x)
    return out


class HybridQAGNN(nn.Module):
    """
    Hybrid QA-GNN:

      • LM branch:
          - BERT/RoBERTa CLS embedding -> lm_classifier -> logit_lm

      • Graph branch:
          - Graph Transformer over one-hop KG subgraph
          - Claim-aware node reweighting
          - Dual pooling (structural + semantic) + claim embedding
          -> graph_classifier -> logit_graph

      • Gating:
          - gate = σ(gate_net(claim_repr)) in [0, 1]
          - final_logit = gate * logit_lm + (1 - gate) * logit_graph
    """

    def __init__(
        self,
        model_name: str,
        n_gnn_layers: int = 2,
        gnn_hidden_dim: int = 256,
        gnn_out_features: int = 256,
        lm_layer_features: int = None,
        gnn_batch_norm: bool = True,  # kept for API compatibility; not used
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

        self.name = "hybrid_qagnn_" + model_name
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

        # 2. Optional LM projection layer for the claim (like original QAGNN)
        self.with_lm_layer = False
        claim_dim = self.hidden_size
        if lm_layer_features is not None:
            self.with_lm_layer = True
            self.lm_dropout = nn.Dropout(lm_layer_dropout)
            self.lm_layer = nn.Linear(self.hidden_size, lm_layer_features)
            claim_dim = lm_layer_features

        # 3. Graph stack: project node features and apply TransformerConv
        self.node_proj = nn.Linear(self.hidden_size, gnn_hidden_dim)
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

        # Norms + residuals in hidden_dim space, final norm in out_features
        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(gnn_hidden_dim) for _ in range(n_gnn_layers - 1)]
        )
        self.gnn_last_norm = nn.LayerNorm(gnn_out_features)

        # 4. Attention projection for semantic pooling
        if gnn_out_features != gnn_hidden_dim:
            self.attn_proj = nn.Linear(gnn_out_features, gnn_hidden_dim)
        else:
            self.attn_proj = nn.Identity()

        # 5. LM-only classifier (uses claim_repr)
        self.lm_classifier = nn.Linear(claim_dim, 1)

        # 6. Graph classifier (uses [pool_struct, pool_semantic, claim_repr])
        graph_dim = 2 * gnn_out_features
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.graph_classifier = nn.Linear(graph_dim + claim_dim, 1)

        # 7. Gating network: gate in [0, 1] from claim_repr
        gate_hidden = max(64, claim_dim // 2)
        self.gate_net = nn.Sequential(
            nn.Linear(claim_dim, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 1),
        )

        logger.info(
            f"Initialized HybridQAGNN with {n_gnn_layers} GNN layers, "
            f"hidden_dim={gnn_hidden_dim}, out_features={gnn_out_features}, claim_dim={claim_dim}."
        )

    def forward(self, claim_tokens, data_graphs):
        """
        Args:
            claim_tokens: dict with input_ids, attention_mask, etc. for BERT
            data_graphs:  torch_geometric.data.Batch with:
                          - x: node embeddings [N, hidden_size]
                          - edge_index: [2, E]
                          - batch: [N] graph indices
        Returns:
            logits: [batch_size] (BCEWithLogitsLoss-ready)
        """
        # ===== LM branch: CLS embedding =====
        bert_output = self.bert(**claim_tokens)
        text_cls = bert_output.last_hidden_state[:, 0, :]  # [B, hidden_size]

        # Claim representation (optionally projected)
        claim_repr = text_cls
        if self.with_lm_layer:
            claim_repr = self.lm_dropout(claim_repr)
            claim_repr = self.lm_layer(claim_repr)  # [B, claim_dim]

        # LM-only logit
        lm_input = self.classifier_dropout(claim_repr)
        logit_lm = self.lm_classifier(lm_input).squeeze(1)  # [B]

        # ===== Graph branch: Graph Transformer over KG subgraph =====
        batch = data_graphs

        if not hasattr(batch, "x") or batch.x is None:
            raise ValueError(
                "HybridQAGNN expects `data_graphs.x` to contain node features."
            )

        node_features = batch.x  # [N, hidden_size]

        # Claim-aware reweighting of node features (cosine similarity)
        text_expanded = text_cls[batch.batch]  # [N, hidden_size]
        relevance = F.cosine_similarity(text_expanded, node_features, dim=-1).unsqueeze(-1)  # [N, 1]
        node_features = node_features * relevance  # [N, hidden_size]

        # Project to GNN hidden dim
        x = self.node_proj(node_features)  # [N, gnn_hidden_dim]

        # Graph Transformer stack
        for i, conv in enumerate(self.gnn_layers):
            h = conv(x, batch.edge_index)
            if i < self.n_gnn_layers - 1:
                # Hidden_dim -> hidden_dim + residual
                h = self.gnn_norms[i](h + x)
            else:
                # Last layer: out_features, no residual from hidden_dim
                h = self.gnn_last_norm(h)
            x = F.relu(h)

        # x has shape [N, gnn_out_features]

        # Structural pooling: mean over nodes per graph
        pool_struct = global_mean_pool(x, batch.batch)  # [B, gnn_out_features]

        # Semantic pooling: attention weighted by claim
        text_context = text_cls[batch.batch]  # [N, hidden_size]
        text_context_proj = self.node_proj(text_context)  # [N, gnn_hidden_dim]

        x_for_attn = self.attn_proj(x)  # [N, gnn_hidden_dim]
        scores = (x_for_attn * text_context_proj).sum(dim=-1)  # [N]

        scores_exp = scores.exp()
        denom = global_add_pool(scores_exp.unsqueeze(-1), batch.batch).squeeze(-1)  # [B]
        attention_weights = scores_exp / (denom[batch.batch] + 1e-8)  # [N]

        pool_semantic = global_add_pool(x * attention_weights.unsqueeze(-1), batch.batch)  # [B, gnn_out_features]

        graph_repr = torch.cat([pool_struct, pool_semantic], dim=-1)  # [B, 2 * gnn_out_features]

        # Graph classifier uses graph_repr + claim_repr
        graph_input = torch.cat([graph_repr, claim_repr], dim=-1)  # [B, 2Dg + Dc]
        graph_input = self.classifier_dropout(graph_input)
        logit_graph = self.graph_classifier(graph_input).squeeze(1)  # [B]

        # ===== Gating: combine LM and Graph logits =====
        gate = torch.sigmoid(self.gate_net(claim_repr)).squeeze(1)  # [B] in [0, 1]
        logits = gate * logit_lm + (1.0 - gate) * logit_graph       # [B]

        return logits
