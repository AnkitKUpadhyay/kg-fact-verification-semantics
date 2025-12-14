# import argparse
# import torch
# import torch.nn as nn
# import transformers
# from datasets import get_dataloader, get_graph_dataloader
# from evaluate import evaluate_on_test_set
# from models import QAGNN, get_bert_model
# from train import train
# from utils import get_logger, save_history, seed_everything, set_global_log_level, load_state_dict

# # Import the V3 Model
# from models_iterative import IterativeFusionGNN_V3
# # Import your successful model
# from improved_qagnn_v2 import ImprovedQAGNN

# logger = get_logger(__name__)

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("model_name", type=str)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--learning_rate", type=float, default=0.00001)
#     parser.add_argument("--n_epochs", type=int, default=5)
#     parser.add_argument("--qa_gnn", action="store_true")
    
#     # Selection Flags
#     parser.add_argument("--iterative_fusion", action="store_true", help="Use SOTA Iterative Fusion V3")
#     parser.add_argument("--use_improved", action="store_true", help="Use Improved QAGNN")
#     parser.add_argument("--dual_pooling", action="store_true")

#     # Standard args
#     parser.add_argument("--use_roberta", action="store_true")
#     parser.add_argument("--freeze_base_model", action="store_true")
#     parser.add_argument("--freeze_up_to_pooler", action="store_true")
#     parser.add_argument("--bert_dropout", type=float, default=0)
#     parser.add_argument("--gnn_dropout", type=float, default=0.3)
#     parser.add_argument("--classifier_dropout", type=float, default=0.3)
#     parser.add_argument("--lm_layer_dropout", type=float, default=0.3)
#     parser.add_argument("--n_gnn_layers", type=int, default=2)
#     parser.add_argument("--gnn_batch_norm", action="store_true")
#     parser.add_argument("--gnn_hidden_dim", type=int, default=256)
#     parser.add_argument("--gnn_out_features", type=int, default=256)
#     parser.add_argument("--lm_layer_features", type=int, default=0)
#     parser.add_argument("--evaluate_only", action="store_true")
#     parser.add_argument("--state_dict_path", type=str, default="")
#     parser.add_argument("--vectorized", action="store_true")
#     parser.add_argument("--subgraph_type", type=str, default="none")
#     parser.add_argument("--subgraph_to_use", type=str, default="discovered")
#     parser.add_argument("--online_embeddings", action="store_true")
#     parser.add_argument("--mix_graphs", action="store_true")

#     return parser.parse_args()

# if __name__ == "__main__":
#     set_global_log_level("debug")
#     seed_everything(57)
#     args = get_args()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     if args.lm_layer_features == 0: args.lm_layer_features = None

#     if args.qa_gnn:
#         if args.iterative_fusion:
#             logger.info("Initializing SOTA Iterative Fusion GNN (V3 - Stabilized)...")
#             model = IterativeFusionGNN_V3(
#                 args.model_name, n_gnn_layers=args.n_gnn_layers, gnn_hidden_dim=args.gnn_hidden_dim,
#                 gnn_out_features=args.gnn_out_features, lm_layer_features=args.lm_layer_features,
#                 gnn_batch_norm=args.gnn_batch_norm, freeze_base_model=args.freeze_base_model,
#                 freeze_up_to_pooler=args.freeze_up_to_pooler, gnn_dropout=args.gnn_dropout,
#                 classifier_dropout=args.classifier_dropout, lm_layer_dropout=args.lm_layer_dropout,
#                 use_roberta=args.use_roberta
#             )
#         elif args.use_improved:
#             logger.info("Initializing Improved QAGNN (Late Fusion)...")
#             model = ImprovedQAGNN(
#                 args.model_name, n_gnn_layers=args.n_gnn_layers, gnn_hidden_dim=args.gnn_hidden_dim,
#                 gnn_out_features=args.gnn_out_features, lm_layer_features=args.lm_layer_features,
#                 gnn_batch_norm=args.gnn_batch_norm, freeze_base_model=args.freeze_base_model,
#                 freeze_up_to_pooler=args.freeze_up_to_pooler, gnn_dropout=args.gnn_dropout,
#                 classifier_dropout=args.classifier_dropout, lm_layer_dropout=args.lm_layer_dropout,
#                 use_roberta=args.use_roberta,
#                 use_dual_pooling=args.dual_pooling
#             )
#         else:
#             logger.info("Initializing Standard QAGNN...")
#             model = QAGNN(
#                 args.model_name, n_gnn_layers=args.n_gnn_layers, gnn_hidden_dim=args.gnn_hidden_dim,
#                 gnn_out_features=args.gnn_out_features, lm_layer_features=args.lm_layer_features,
#                 gnn_batch_norm=args.gnn_batch_norm, freeze_base_model=args.freeze_base_model,
#                 freeze_up_to_pooler=args.freeze_up_to_pooler, gnn_dropout=args.gnn_dropout,
#                 classifier_dropout=args.classifier_dropout, lm_layer_dropout=args.lm_layer_dropout,
#                 use_roberta=args.use_roberta
#             )

#         if args.online_embeddings: embedding_model = model.bert
#         else: embedding_model = None
        
#         train_loader = get_graph_dataloader("train", subgraph_type=args.subgraph_type, online_embeddings=args.online_embeddings, model=embedding_model, batch_size=args.batch_size, mix_graphs=args.mix_graphs)
#         val_loader = get_graph_dataloader("val", subgraph_type=args.subgraph_type, online_embeddings=args.online_embeddings, model=embedding_model, batch_size=args.batch_size, mix_graphs=args.mix_graphs)
#         test_loader = get_graph_dataloader("test", subgraph_type=args.subgraph_type, online_embeddings=args.online_embeddings, model=embedding_model, batch_size=args.batch_size // 4, mix_graphs=args.mix_graphs, shuffle=False, drop_last=False)
    
#     else:
#         # Standard BERT logic
#         train_loader = get_dataloader("train", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size)
#         val_loader = get_dataloader("val", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size)
#         test_loader = get_dataloader("test", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size // 4, shuffle=False, drop_last=False)
#         model = get_bert_model(args.model_name, include_classifier=True, num_labels=2, freeze_base_model=args.freeze_base_model, freeze_up_to_pooler=args.freeze_up_to_pooler, dropout_rate=args.bert_dropout, use_roberta=args.use_roberta)

#     criterion = nn.BCEWithLogitsLoss()
#     if not args.evaluate_only:
#         optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
#         lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_loader) * args.n_epochs)
#         history, models_dict = train(model=model, criterion=criterion, optimizer=optimizer, qa_gnn=args.qa_gnn, train_loader=train_loader, val_loader=val_loader, n_epochs=args.n_epochs, scheduler=lr_scheduler)
#         model.load_state_dict(models_dict["best_model_state_dict"])
#     else:
#         model.load_state_dict(load_state_dict(args.state_dict_path, device))
#         history = {}

#     test_results = evaluate_on_test_set(args.qa_gnn, model=model, test_loader=test_loader, criterion=criterion, device=device)
#     test_accuracy = test_results["overall"]["accuracy"]
#     if not args.evaluate_only: save_history(args.model_name, history)
#     else: save_history("eval_" + args.model_name, history)
    
#     logger.info(f"Model {args.model_name}: Test accuracy: {test_accuracy * 100:.4f}%")
#     logger.info(f"{test_results=}")


# import argparse
# import torch
# import torch.nn as nn
# import transformers

# from datasets import get_dataloader, get_graph_dataloader
# from evaluate import evaluate_on_test_set
# from models import QAGNN, get_bert_model
# from models_iterative import GraphTransformerQAGNN
# from models_hybrid import HybridQAGNN
# from train import train
# from utils import (
#     get_logger,
#     save_history,
#     seed_everything,
#     set_global_log_level,
#     load_state_dict,
# )

# logger = get_logger(__name__)


# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("model_name", type=str, help="Name for saving model/history.")

#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--learning_rate", type=float, default=0.00001)
#     parser.add_argument("--n_epochs", type=int, default=5)

#     parser.add_argument("--qa_gnn", action="store_true",
#                         help="Use QA-GNN style model (graph + text).")
#     parser.add_argument(
#         "--graph_transformer",
#         action="store_true",
#         help="Use GraphTransformerQAGNN instead of standard QAGNN."
#     )
#     parser.add_argument(
#         "--hybrid_qagnn",
#         action="store_true",
#         help="Use HybridQAGNN (LM + Graph with gating)."
#     )
#     parser.add_argument("--use_roberta", action="store_true",
#                         help="Use RoBERTa instead of BERT.")

#     parser.add_argument("--freeze_base_model", action="store_true")
#     parser.add_argument("--freeze_up_to_pooler", action="store_true")
#     parser.add_argument("--bert_dropout", type=float, default=0.0)
#     parser.add_argument("--gnn_dropout", type=float, default=0.3)
#     parser.add_argument("--classifier_dropout", type=float, default=0.3)
#     parser.add_argument("--lm_layer_dropout", type=float, default=0.3)

#     parser.add_argument("--n_gnn_layers", type=int, default=2)
#     parser.add_argument("--gnn_batch_norm", action="store_true",
#                         help="Kept for API compatibility; ignored in transformer models.")
#     parser.add_argument("--gnn_hidden_dim", type=int, default=256)
#     parser.add_argument("--gnn_out_features", type=int, default=256)
#     parser.add_argument("--lm_layer_features", type=int, default=0)

#     parser.add_argument("--evaluate_only", action="store_true",
#                         help="Only evaluate a saved model.")
#     parser.add_argument("--state_dict_path", type=str, default="",
#                         help="Path to model weights when using --evaluate_only.")

#     parser.add_argument(
#         "--subgraph_type",
#         type=str,
#         default="none",
#         help="Subgraph type: none, direct, direct_filled, one_hop, relevant",
#     )
#     parser.add_argument(
#         "--subgraph_to_use",
#         type=str,
#         default="discovered",
#         help="For non-QAGNN models: discovered, connected, walkable",
#     )
#     parser.add_argument("--online_embeddings", action="store_true",
#                         help="Compute LM embeddings on the fly.")
#     parser.add_argument("--mix_graphs", action="store_true",
#                         help="Mix connected+walkable graphs (QA-GNN path).")

#     parser.add_argument("--vectorized", action="store_true",
#                         help="Kept for compatibility; not used here.")

#     return parser.parse_args()


# if __name__ == "__main__":
#     set_global_log_level("debug")
#     seed_everything(57)
#     args = get_args()

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Safety checks
#     if args.evaluate_only and args.state_dict_path == "":
#         raise ValueError("Argument --state_dict_path must be provided when --evaluate_only is set.")

#     if args.subgraph_type is not None and args.subgraph_type.lower() == "none":
#         if args.qa_gnn:
#             raise ValueError("Argument --subgraph_type cannot be 'none' when using --qa_gnn.")
#         args.subgraph_type = None

#     if args.lm_layer_features == 0:
#         args.lm_layer_features = None

#     # Make sure we don't select multiple special GNN models at once
#     selected_models = sum([
#         1 if args.graph_transformer else 0,
#         1 if args.hybrid_qagnn else 0,
#     ])
#     if selected_models > 1:
#         raise ValueError("Choose at most one of --graph_transformer or --hybrid_qagnn.")

#     # ---------------------- Model + Dataloaders ---------------------- #
#     if args.qa_gnn:
#         # Model selection within the QA-GNN family
#         if args.hybrid_qagnn:
#             logger.info("Initializing HybridQAGNN (LM + Graph gating) model...")
#             model = HybridQAGNN(
#                 model_name=args.model_name,
#                 n_gnn_layers=args.n_gnn_layers,
#                 gnn_hidden_dim=args.gnn_hidden_dim,
#                 gnn_out_features=args.gnn_out_features,
#                 lm_layer_features=args.lm_layer_features,
#                 gnn_batch_norm=args.gnn_batch_norm,
#                 freeze_base_model=args.freeze_base_model,
#                 freeze_up_to_pooler=args.freeze_up_to_pooler,
#                 gnn_dropout=args.gnn_dropout,
#                 classifier_dropout=args.classifier_dropout,
#                 lm_layer_dropout=args.lm_layer_dropout,
#                 use_roberta=args.use_roberta,
#             )
#         elif args.graph_transformer:
#             logger.info("Initializing GraphTransformerQAGNN (graph transformer) model...")
#             model = GraphTransformerQAGNN(
#                 model_name=args.model_name,
#                 n_gnn_layers=args.n_gnn_layers,
#                 gnn_hidden_dim=args.gnn_hidden_dim,
#                 gnn_out_features=args.gnn_out_features,
#                 lm_layer_features=args.lm_layer_features,
#                 gnn_batch_norm=args.gnn_batch_norm,
#                 freeze_base_model=args.freeze_base_model,
#                 freeze_up_to_pooler=args.freeze_up_to_pooler,
#                 gnn_dropout=args.gnn_dropout,
#                 classifier_dropout=args.classifier_dropout,
#                 lm_layer_dropout=args.lm_layer_dropout,
#                 use_roberta=args.use_roberta,
#             )
#         else:
#             logger.info("Initializing standard QAGNN model...")
#             model = QAGNN(
#                 args.model_name,
#                 n_gnn_layers=args.n_gnn_layers,
#                 gnn_hidden_dim=args.gnn_hidden_dim,
#                 gnn_out_features=args.gnn_out_features,
#                 lm_layer_features=args.lm_layer_features,
#                 gnn_batch_norm=args.gnn_batch_norm,
#                 freeze_base_model=args.freeze_base_model,
#                 freeze_up_to_pooler=args.freeze_up_to_pooler,
#                 gnn_dropout=args.gnn_dropout,
#                 classifier_dropout=args.classifier_dropout,
#                 lm_layer_dropout=args.lm_layer_dropout,
#                 use_roberta=args.use_roberta,
#             )

#         if args.online_embeddings:
#             embedding_model = model.bert
#         else:
#             embedding_model = None

#         train_loader = get_graph_dataloader(
#             "train",
#             subgraph_type=args.subgraph_type,
#             online_embeddings=args.online_embeddings,
#             model=embedding_model,
#             batch_size=args.batch_size,
#             mix_graphs=args.mix_graphs,
#         )
#         val_loader = get_graph_dataloader(
#             "val",
#             subgraph_type=args.subgraph_type,
#             online_embeddings=args.online_embeddings,
#             model=embedding_model,
#             batch_size=args.batch_size,
#             mix_graphs=args.mix_graphs,
#         )
#         test_loader = get_graph_dataloader(
#             "test",
#             subgraph_type=args.subgraph_type,
#             online_embeddings=args.online_embeddings,
#             model=embedding_model,
#             batch_size=max(1, args.batch_size // 4),
#             mix_graphs=args.mix_graphs,
#             shuffle=False,
#             drop_last=False,
#         )
#     else:
#         # Plain BERT / RoBERTa baseline
#         train_loader = get_dataloader(
#             "train",
#             args.subgraph_type,
#             subgraph_to_use=args.subgraph_to_use,
#             batch_size=args.batch_size,
#         )
#         val_loader = get_dataloader(
#             "val",
#             args.subgraph_type,
#             subgraph_to_use=args.subgraph_to_use,
#             batch_size=args.batch_size,
#         )
#         test_loader = get_dataloader(
#             "test",
#             args.subgraph_type,
#             subgraph_to_use=args.subgraph_to_use,
#             batch_size=max(1, args.batch_size // 4),
#             shuffle=False,
#             drop_last=False,
#         )
#         model = get_bert_model(
#             args.model_name,
#             include_classifier=True,
#             num_labels=2,
#             freeze_base_model=args.freeze_base_model,
#             freeze_up_to_pooler=args.freeze_up_to_pooler,
#             dropout_rate=args.bert_dropout,
#             use_roberta=args.use_roberta,
#         )

#     model.to(device)

#     # ---------------------- Train / Evaluate ---------------------- #
#     criterion = nn.BCEWithLogitsLoss()

#     if not args.evaluate_only:
#         optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
#         lr_scheduler = transformers.get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=50,
#             num_training_steps=len(train_loader) * args.n_epochs,
#         )

#         history, models_dict = train(
#             model=model,
#             criterion=criterion,
#             optimizer=optimizer,
#             qa_gnn=args.qa_gnn,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             n_epochs=args.n_epochs,
#             scheduler=lr_scheduler,
#         )

#         logger.info(f"Loading best model state dict from epoch {history['best_epoch']}...")
#         model.load_state_dict(models_dict["best_model_state_dict"])
#         logger.info("Done loading best model state dict.")
#     else:
#         logger.info(f"Loading model state dict from path {args.state_dict_path}...")
#         model.load_state_dict(load_state_dict(args.state_dict_path, device))
#         logger.info("Done loading state dict.")
#         history = {}

#     # ---------------------- Test ---------------------- #
#     test_results = evaluate_on_test_set(
#         args.qa_gnn,
#         model=model,
#         test_loader=test_loader,
#         criterion=criterion,
#         device=device,
#     )
#     test_accuracy = test_results["overall"]["accuracy"]
#     average_test_loss = test_results["overall"]["loss"]

#     history["test_accuracy"] = test_accuracy
#     history["test_results"] = test_results

#     if not args.evaluate_only:
#         save_history(args.model_name, history)
#     else:
#         save_history("eval_" + args.model_name, history)

#     logger.info(
#         f"Model \"{args.model_name}\": "
#         f"Test accuracy: {test_accuracy * 100:.4f}%, average_test_loss={average_test_loss:.4f}."
#     )
#     logger.info(f"{test_results=}")



import argparse
import torch
import torch.nn as nn
import transformers

from datasets import get_dataloader, get_graph_dataloader
from evaluate import evaluate_on_test_set
from models import QAGNN, get_bert_model
from models_cross_attn import ConditionedGraphTransformer # Uses the new NLI version
from train import train
from utils import (
    get_logger,
    save_history,
    seed_everything,
    set_global_log_level,
    load_state_dict,
)

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--qa_gnn", action="store_true")
    parser.add_argument("--rgccam", action="store_true", help="Use Relational GCCAM") # Only one flag needed
    parser.add_argument("--cgt", action="store_true", help="Use Conditioned Graph Transformer (Early Fusion)")
    parser.add_argument("--use_roberta", action="store_true")
    parser.add_argument("--freeze_base_model", action="store_true")
    parser.add_argument("--freeze_up_to_pooler", action="store_true")
    parser.add_argument("--bert_dropout", type=float, default=0.0)
    parser.add_argument("--gnn_dropout", type=float, default=0.3)
    parser.add_argument("--classifier_dropout", type=float, default=0.3)
    parser.add_argument("--lm_layer_dropout", type=float, default=0.3)
    parser.add_argument("--n_gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_batch_norm", action="store_true")
    parser.add_argument("--gnn_hidden_dim", type=int, default=256)
    parser.add_argument("--gnn_out_features", type=int, default=256)
    parser.add_argument("--lm_layer_features", type=int, default=0)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--state_dict_path", type=str, default="")
    parser.add_argument("--subgraph_type", type=str, default="none")
    parser.add_argument("--subgraph_to_use", type=str, default="discovered")
    parser.add_argument("--online_embeddings", action="store_true")
    parser.add_argument("--mix_graphs", action="store_true")
    parser.add_argument("--vectorized", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    set_global_log_level("debug")
    seed_everything(57)
    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.evaluate_only and args.state_dict_path == "":
        raise ValueError("Argument --state_dict_path must be provided when --evaluate_only is set.")

    if args.subgraph_type is not None and args.subgraph_type.lower() == "none":
        if args.qa_gnn:
            raise ValueError("Argument --subgraph_type cannot be 'none' when using --qa_gnn.")
        args.subgraph_type = None

    if args.lm_layer_features == 0:
        args.lm_layer_features = None

    # ---------------------- Model ---------------------- #
    if args.qa_gnn:
        if args.rgccam:
            logger.info("Initializing RelationalGCCAM (NLI-Enhanced)...")
            model = RelationalGCCAM(
                model_name=args.model_name,
                n_gnn_layers=args.n_gnn_layers,
                gnn_hidden_dim=args.gnn_hidden_dim,
                gnn_out_features=args.gnn_out_features,
                freeze_base_model=args.freeze_base_model,
                freeze_up_to_pooler=args.freeze_up_to_pooler,
                gnn_dropout=args.gnn_dropout,
                classifier_dropout=args.classifier_dropout,
                use_roberta=args.use_roberta
            )

            # Model Init
        elif args.cgt:
            logger.info("Initializing ConditionedGraphTransformer (Early Fusion)...")
            model = ConditionedGraphTransformer(
                model_name=args.model_name,
                n_gnn_layers=args.n_gnn_layers,
                gnn_hidden_dim=args.gnn_hidden_dim,
                gnn_out_features=args.gnn_out_features,
                freeze_base_model=args.freeze_base_model,
                freeze_up_to_pooler=args.freeze_up_to_pooler,
                gnn_dropout=args.gnn_dropout,
                classifier_dropout=args.classifier_dropout,
                use_roberta=args.use_roberta
            )
            
        else:
            logger.info("Initializing standard QAGNN model...")
            model = QAGNN(
                args.model_name,
                n_gnn_layers=args.n_gnn_layers,
                gnn_hidden_dim=args.gnn_hidden_dim,
                gnn_out_features=args.gnn_out_features,
                lm_layer_features=args.lm_layer_features,
                gnn_batch_norm=args.gnn_batch_norm,
                freeze_base_model=args.freeze_base_model,
                freeze_up_to_pooler=args.freeze_up_to_pooler,
                gnn_dropout=args.gnn_dropout,
                classifier_dropout=args.classifier_dropout,
                lm_layer_dropout=args.lm_layer_dropout,
                use_roberta=args.use_roberta,
            )

        if args.online_embeddings:
            embedding_model = model.bert
        else:
            embedding_model = None

        train_loader = get_graph_dataloader(
            "train", subgraph_type=args.subgraph_type, online_embeddings=args.online_embeddings,
            model=embedding_model, batch_size=args.batch_size, mix_graphs=args.mix_graphs
        )
        val_loader = get_graph_dataloader(
            "val", subgraph_type=args.subgraph_type, online_embeddings=args.online_embeddings,
            model=embedding_model, batch_size=args.batch_size, mix_graphs=args.mix_graphs
        )
        test_loader = get_graph_dataloader(
            "test", subgraph_type=args.subgraph_type, online_embeddings=args.online_embeddings,
            model=embedding_model, batch_size=max(1, args.batch_size // 4),
            mix_graphs=args.mix_graphs, shuffle=False, drop_last=False
        )
    else:
        # BERT Baseline
        train_loader = get_dataloader(
            "train", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size
        )
        val_loader = get_dataloader(
            "val", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size
        )
        test_loader = get_dataloader(
            "test", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=max(1, args.batch_size // 4),
            shuffle=False, drop_last=False
        )
        model = get_bert_model(
            args.model_name, include_classifier=True, num_labels=2,
            freeze_base_model=args.freeze_base_model, freeze_up_to_pooler=args.freeze_up_to_pooler,
            dropout_rate=args.bert_dropout, use_roberta=args.use_roberta
        )

    model.to(device)

    # ---------------------- Optimizer (Differential LR) ---------------------- #
    criterion = nn.BCEWithLogitsLoss()

    if not args.evaluate_only:
        # Separate BERT params from GNN params
        bert_params = list(map(id, model.bert.parameters()))
        gnn_params = filter(lambda p: id(p) not in bert_params, model.parameters())
        
        # Apply 1x LR to BERT, 20x LR to GNN (Standard QA-GNN trick)
        optimizer = torch.optim.AdamW([
            {'params': model.bert.parameters(), 'lr': args.learning_rate},
            {'params': gnn_params, 'lr': args.learning_rate * 20} 
        ])
        
        logger.info(f"Using Differential LR: BERT={args.learning_rate}, GNN={args.learning_rate*20}")

        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=50,
            num_training_steps=len(train_loader) * args.n_epochs,
        )

        history, models_dict = train(
            model=model, criterion=criterion, optimizer=optimizer,
            qa_gnn=args.qa_gnn, train_loader=train_loader,
            val_loader=val_loader, n_epochs=args.n_epochs, scheduler=lr_scheduler
        )

        logger.info(f"Loading best model state dict from epoch {history['best_epoch']}...")
        model.load_state_dict(models_dict["best_model_state_dict"])
        logger.info("Done loading best model state dict.")
    else:
        logger.info(f"Loading model state dict from path {args.state_dict_path}...")
        model.load_state_dict(load_state_dict(args.state_dict_path, device))
        history = {}

    test_results = evaluate_on_test_set(
        args.qa_gnn, model=model, test_loader=test_loader, criterion=criterion, device=device
    )
    test_accuracy = test_results["overall"]["accuracy"]
    average_test_loss = test_results["overall"]["loss"]

    history["test_accuracy"] = test_accuracy
    history["test_results"] = test_results

    if not args.evaluate_only:
        save_history(args.model_name, history)
    else:
        save_history("eval_" + args.model_name, history)

    logger.info(
        f"Model \"{args.model_name}\": Test accuracy: {test_accuracy * 100:.4f}%, avg_test_loss={average_test_loss:.4f}."
    )
    logger.info(f"{test_results=}")