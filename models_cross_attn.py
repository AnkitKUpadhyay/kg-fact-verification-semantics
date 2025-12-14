import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from models import get_bert_model
from utils import get_logger

logger = get_logger(__name__)

def global_add_pool(x: torch.Tensor, batch: torch.Tensor, size: int = None) -> torch.Tensor:
    if batch.numel() == 0:
        return x.new_zeros((0, x.size(1)))
    if size is None:
        size = int(batch.max().item()) + 1
    out = x.new_zeros(size, x.size(1))
    index = batch.view(-1, 1).expand(-1, x.size(1))
    out.scatter_add_(0, index, x)
    return out

class RGATLayer(MessagePassing):
    """Robust Relational GAT Layer (proven to be stable)."""
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.0):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.head_dim = out_channels // heads
        
        self.lin_q = nn.Linear(in_channels, out_channels)
        self.lin_k = nn.Linear(in_channels, out_channels)
        self.lin_v = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(in_channels, out_channels) 
        self.scale = self.head_dim ** -0.5

    def forward(self, x, edge_index, edge_attr):
        q = self.lin_q(x).view(-1, self.heads, self.head_dim)
        k = self.lin_k(x).view(-1, self.heads, self.head_dim)
        v = self.lin_v(x).view(-1, self.heads, self.head_dim)
        e = self.lin_edge(edge_attr).view(-1, self.heads, self.head_dim)
        
        out = self.propagate(edge_index, q=q, k=k, v=v, e=e)
        return out.view(-1, self.out_channels) 

    def message(self, q_i, k_j, v_j, e, index):
        attn_score = (q_i * (k_j + e)).sum(dim=-1) * self.scale
        attn_weights = softmax(attn_score, index) 
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        return attn_weights.unsqueeze(-1) * (v_j + e)


class ConditionedGraphTransformer(nn.Module):
    """
    Conditioned Graph Transformer (Early Fusion).
    
    Key Innovation: 
    Concatenates the Claim Representation to EVERY Node Representation 
    BEFORE the GNN layers. This forces the GNN to perform reasoning 
    conditioned on the text, similar to how BERT processes [Claim, Triples].
    """
    def __init__(
        self,
        model_name: str,
        n_gnn_layers: int = 2,
        gnn_hidden_dim: int = 256,
        gnn_out_features: int = 256,
        lm_layer_features: int = None,
        num_relations: int = 500, 
        freeze_base_model: bool = False,
        freeze_up_to_pooler: bool = True,
        gnn_dropout: float = 0.3,
        classifier_dropout: float = 0.2,
        use_roberta: bool = False,
    ):
        super().__init__()
        self.name = "cgt_" + model_name
        self.n_gnn_layers = n_gnn_layers
        
        # 1. Text Encoder
        self.bert = get_bert_model(
            "bert_" + model_name,
            include_classifier=False,
            freeze_base_model=freeze_base_model,
            freeze_up_to_pooler=freeze_up_to_pooler,
            use_roberta=use_roberta,
        )
        self.hidden_size = self.bert.config.hidden_size

        # 2. Projections
        # Node Proj: We map 768 -> 256
        self.node_proj = nn.Linear(self.hidden_size, gnn_hidden_dim)
        # Claim Proj: We map 768 -> 256
        self.claim_proj = nn.Linear(self.hidden_size, gnn_hidden_dim)
        
        # Edge Proj: 768 (BERT) -> 256 + 256 (Because nodes are 2x size now!)
        # Actually, we keep edges at gnn_hidden_dim, but project them up inside the layer?
        # Simpler: We project edges to match the 'Conditioned' node size (256+256 = 512)
        # to allow full interaction.
        self.conditioned_dim = gnn_hidden_dim * 2 
        
        self.edge_embedding = nn.Embedding(num_relations, self.conditioned_dim)
        self.edge_proj = nn.Linear(self.hidden_size, self.conditioned_dim)
        
        self.loop_rel_emb = nn.Parameter(torch.Tensor(1, self.conditioned_dim))
        nn.init.xavier_uniform_(self.loop_rel_emb)

        # 3. GNN Stack (Operating on 512-dim features)
        self.gnn_layers = nn.ModuleList()
        for i in range(n_gnn_layers):
            # Output dim shrinks back to 256 at the last layer for classification
            in_dim = self.conditioned_dim
            out_dim = self.conditioned_dim if i < n_gnn_layers - 1 else gnn_out_features
            
            self.gnn_layers.append(
                RGATLayer(in_dim, out_dim, heads=4, dropout=gnn_dropout)
            )

        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(self.conditioned_dim) for _ in range(n_gnn_layers - 1)]
        )
        self.gnn_last_norm = nn.LayerNorm(gnn_out_features)

        # 4. Cross-Attention Fusion (Still useful for final aggregation)
        self.attn_dim = gnn_out_features
        self.claim_attn_q = nn.Linear(self.hidden_size, self.attn_dim)
        self.node_attn_k = nn.Linear(gnn_out_features, self.attn_dim)
        self.node_attn_v = nn.Linear(gnn_out_features, self.attn_dim)
        self.scale = self.attn_dim ** -0.5

        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.attn_dim + self.hidden_size, 1)
        )

    def forward(self, claim_tokens, data_graphs):
        # A. Text
        bert_output = self.bert(**claim_tokens)
        claim_repr = bert_output.last_hidden_state[:, 0, :] # [B, 768]

        # B. Graph Inputs
        batch = data_graphs
        x = batch.x # [N, 768]
        edge_index = batch.edge_index
        
        if hasattr(batch, 'edge_type') and batch.edge_type is not None:
            edge_attr = self.edge_embedding(batch.edge_type) 
        elif hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            edge_attr = self.edge_proj(batch.edge_attr)
        else:
            edge_attr = torch.zeros((edge_index.size(1), self.conditioned_dim), device=x.device)

        # C. EARLY FUSION: Condition Nodes on Claim
        # 1. Project inputs
        x_nodes = self.node_proj(x)   # [N, 256]
        x_claim = self.claim_proj(claim_repr) # [B, 256]
        
        # 2. Expand Claim to match Nodes
        # batch.batch maps nodes to graph_id
        x_claim_expanded = x_claim[batch.batch] # [N, 256]
        
        # 3. Concatenate: New Node Dim is 512
        x_conditioned = torch.cat([x_nodes, x_claim_expanded], dim=-1) # [N, 512]

        # D. Robust Augmentation
        diff = edge_index.size(1) - edge_attr.size(0)
        if diff > 0:
            pad_attr = self.loop_rel_emb.expand(diff, -1)
            edge_attr = torch.cat([edge_attr, pad_attr], dim=0)
        
        num_nodes = x.size(0)
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=x.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1) 
        loop_attr = self.loop_rel_emb.expand(num_nodes, -1) 
        edge_index_aug = torch.cat([edge_index, loop_index], dim=1)
        edge_attr_aug = torch.cat([edge_attr, loop_attr], dim=0)

        # E. GNN (Conditioned Message Passing)
        h = x_conditioned
        for i, conv in enumerate(self.gnn_layers):
            h_out = conv(h, edge_index_aug, edge_attr_aug)
            
            if i < self.n_gnn_layers - 1:
                h = self.gnn_norms[i](h_out + h) # Residual
                h = F.relu(h)
            else:
                h = self.gnn_last_norm(h_out)
        
        # h is now [N, 256]
        
        # F. Cross-Attention Aggregation
        Q = self.claim_attn_q(claim_repr) 
        K = self.node_attn_k(h)           
        V = self.node_attn_v(h)           

        Q_expanded = Q[batch.batch] 
        scores = torch.sum(Q_expanded * K, dim=-1) * self.scale 
        
        scores_exp = scores.exp()
        denom = global_add_pool(scores_exp.unsqueeze(-1), batch.batch, size=Q.size(0)).squeeze(-1)
        weights = scores_exp / (denom[batch.batch] + 1e-8)
        
        graph_context = global_add_pool(V * weights.unsqueeze(-1), batch.batch, size=Q.size(0))

        # G. Classification
        logits = self.classifier(torch.cat([graph_context, claim_repr], dim=-1))
        
        return logits.squeeze(1)