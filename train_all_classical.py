# """
# Task 2: Train ALL Classical ML Models on Extracted Features
# - Model 1: Logistic Regression (Scaled Baseline)
# - Model 2: Random Forest (Interpretable)
# - Model 3: XGBoost (Performance + Early Stopping)
# - Saves all models, predictions, feature importances, and comparison table.
# """

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import xgboost as xgb
# import pickle
# import os
# import warnings

# # --- Configuration ---
# FEATURE_DIR = '/users/PAS2136/upadha2/factKG/Fact-or-Fiction/features'
# MODEL_SAVE_DIR = 'models'
# RESULT_SAVE_DIR = 'results'
# # ---

# def print_feature_importance(model, feature_names, top_n=15):
#     """Prints the most important features for tree-based models."""
#     if not hasattr(model, 'feature_importances_'):
#         return
        
#     print(f"\n--- Top {top_n} Features ---")
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[-top_n:]
    
#     # Create DataFrame for saving
#     importance_df = pd.DataFrame({
#         'feature': feature_names,
#         'importance': importances
#     }).sort_values('importance', ascending=False)

#     print(importance_df.head(top_n).to_string(index=False))
#     return importance_df

# def print_lr_coefficients(model, feature_names, top_n=10):
#     """Prints the most impactful coefficients for Logistic Regression."""
#     if not hasattr(model, 'coef_'):
#         return
        
#     print(f"\n--- Top {top_n} Most Impactful Features (Coefficients) ---")
#     coeffs = model.coef_[0]
#     coeff_df = pd.DataFrame({
#         'feature': feature_names,
#         'coefficient': coeffs,
#         'abs_coeff': np.abs(coeffs)
#     }).sort_values('abs_coeff', ascending=False)
    
#     print("Most Positive (Predicts 'Supported'):")
#     print(coeff_df.sort_values('coefficient', ascending=False).head(top_n).to_string(index=False))
#     print("\nMost Negative (Predicts 'Refuted'):")
#     print(coeff_df.sort_values('coefficient', ascending=True).head(top_n).to_string(index=False))
#     return coeff_df.sort_values('abs_coeff', ascending=False)

# def print_confusion_matrix(y_true, y_pred, model_name=""):
#     """Prints a formatted confusion matrix."""
#     print(f"\nConfusion Matrix ({model_name}):")
#     cm = confusion_matrix(y_true, y_pred)
#     print(cm)
#     print(f"True Negatives (Refuted):  {cm[0,0]}")
#     print(f"False Positives (Refuted): {cm[0,1]}")
#     print(f"False Negatives (Supported): {cm[1,0]}")
#     print(f"True Positives (Supported):  {cm[1,1]}")


# if __name__ == "__main__":
#     warnings.filterwarnings('ignore')
#     os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
#     os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

#     print("="*60)
#     print("CLASSICAL MODEL TRAINING PIPELINE")
#     print("="*60)

#     # 1. Load Data
#     print("\nLoading features...")
#     train_df = pd.read_csv(os.path.join(FEATURE_DIR, 'features_one_hop_train.csv'))
#     val_df = pd.read_csv(os.path.join(FEATURE_DIR, 'features_one_hop_dev.csv'))
#     test_df = pd.read_csv(os.path.join(FEATURE_DIR, 'features_one_hop_test.csv'))

#     # 2. Pre-process Data
    
#     # Smart feature cleaning
#     cols_to_drop = ['has_cycle']
#     train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
#     val_df = val_df.drop(columns=cols_to_drop, errors='ignore')
#     test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

#     # Separate features and labels
#     feature_cols = [col for col in train_df.columns if col not in ['label', 'claim']]
    
#     X_train = train_df[feature_cols]
#     y_train = train_df['label']
#     X_val = val_df[feature_cols]
#     y_val = val_df['label']
#     X_test = test_df[feature_cols]
#     y_test = test_df['label']

#     print(f"Training samples: {len(X_train)}")
#     print(f"Validation samples: {len(X_val)}")
#     print(f"Test samples: {len(X_test)}")
#     print(f"Features: {len(feature_cols)}")

#     # 3. Scale Data (for Logistic Regression)
#     print("\nScaling features for Logistic Regression...")
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)
    
#     scaler_path = os.path.join(MODEL_SAVE_DIR, 'scaler.pkl')
#     with open(scaler_path, 'wb') as f:
#         pickle.dump(scaler, f)
#     print(f"✅ Scaler saved to {scaler_path}")

#     # --- Model 1: Logistic Regression (Baseline) ---
#     print("\n" + "="*60)
#     print("MODEL 1: LOGISTIC REGRESSION")
#     print("="*60)
    
#     lr_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1, class_weight='balanced')
#     lr_model.fit(X_train_scaled, y_train)

#     print("\n--- Test Set Results ---")
#     y_test_pred_lr = lr_model.predict(X_test_scaled)
#     test_acc_lr = accuracy_score(y_test, y_test_pred_lr)
#     print(f"Accuracy: {test_acc_lr:.4f} ({test_acc_lr*100:.2f}%)")
#     print(classification_report(y_test, y_test_pred_lr, target_names=['Refuted', 'Supported']))
#     print_confusion_matrix(y_test, y_test_pred_lr, "Logistic Regression")

#     lr_coeffs = print_lr_coefficients(lr_model, feature_cols)
    
#     # Save artifacts
#     with open(os.path.join(MODEL_SAVE_DIR, 'lr_baseline_one_hop.pkl'), 'wb') as f:
#         pickle.dump(lr_model, f)
#     lr_coeffs.to_csv(os.path.join(RESULT_SAVE_DIR, 'lr_feature_importance_one_hop.csv'), index=False)
#     pd.DataFrame({'true_label': y_test, 'predicted_label': y_test_pred_lr}).to_csv(
#         os.path.join(RESULT_SAVE_DIR, 'lr_test_predictions_one_hop.csv'), index=False
#     )
#     print("\n✅ Logistic Regression model and results saved.")

#     # --- Model 2: Random Forest (Interpretable) ---
#     print("\n" + "="*60)
#     print("MODEL 2: RANDOM FOREST")
#     print("="*60)
#     print("\nTraining Random Forest...")
    
#     rf_model = RandomForestClassifier(
#         n_estimators=100,
#         max_depth=20,
#         min_samples_split=10,
#         min_samples_leaf=5,
#         random_state=42,
#         n_jobs=-1,
#         verbose=0,
#         class_weight='balanced'
#     )
#     rf_model.fit(X_train, y_train)

#     print("\n--- Test Set Results ---")
#     y_test_pred_rf = rf_model.predict(X_test)
#     test_acc_rf = accuracy_score(y_test, y_test_pred_rf)
#     print(f"Accuracy: {test_acc_rf:.4f} ({test_acc_rf*100:.2f}%)")
#     print(classification_report(y_test, y_test_pred_rf, target_names=['Refuted', 'Supported']))
#     print_confusion_matrix(y_test, y_test_pred_rf, "Random Forest")
    
#     rf_importance = print_feature_importance(rf_model, feature_cols)

#     # Save artifacts
#     with open(os.path.join(MODEL_SAVE_DIR, 'rf_baseline_one_hop.pkl'), 'wb') as f:
#         pickle.dump(rf_model, f)
#     rf_importance.to_csv(os.path.join(RESULT_SAVE_DIR, 'rf_feature_importance_one_hop.csv'), index=False)
#     pd.DataFrame({'true_label': y_test, 'predicted_label': y_test_pred_rf}).to_csv(
#         os.path.join(RESULT_SAVE_DIR, 'rf_test_predictions_one_hop.csv'), index=False
#     )
#     print("\n✅ Random Forest model and results saved.")

#     # --- Model 3: XGBoost (Performance + Early Stopping) ---
#     print("\n" + "="*60)
#     print("MODEL 3: XGBOOST")
#     print("="*60)
#     print("\nTraining XGBoost with Early Stopping...")

#     xgb_model = xgb.XGBClassifier(
#         n_estimators=200,      # Increased n_estimators
#         max_depth=10,
#         learning_rate=0.1,
#         random_state=42,
#         n_jobs=-1,
#         use_label_encoder=False,
#         eval_metric='logloss',
#         scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
#         early_stopping_rounds=10  # <-- ADDED
#     )
    
#     # Train with evaluation set for early stopping
#     xgb_model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)], # <-- ADDED
#         verbose=False             # <-- ADDED
#     )
    
#     print(f"Best iteration: {xgb_model.best_iteration}")

#     print("\n--- Test Set Results ---")
#     y_test_pred_xgb = xgb_model.predict(X_test)
#     test_acc_xgb = accuracy_score(y_test, y_test_pred_xgb)
#     print(f"Accuracy: {test_acc_xgb:.4f} ({test_acc_xgb*100:.2f}%)")
#     print(classification_report(y_test, y_test_pred_xgb, target_names=['Refuted', 'Supported']))
#     print_confusion_matrix(y_test, y_test_pred_xgb, "XGBoost")
    
#     xgb_importance = print_feature_importance(xgb_model, feature_cols)

#     # Save artifacts
#     with open(os.path.join(MODEL_SAVE_DIR, 'xgb_baseline_one_hop.pkl'), 'wb') as f:
#         pickle.dump(xgb_model, f)
#     xgb_importance.to_csv(os.path.join(RESULT_SAVE_DIR, 'xgb_feature_importance_one_hop.csv'), index=False)
#     pd.DataFrame({'true_label': y_test, 'predicted_label': y_test_pred_xgb}).to_csv(
#         os.path.join(RESULT_SAVE_DIR, 'xgb_test_predictions_one_hop.csv'), index=False
#     )
#     print("\n✅ XGBoost model and results saved.")
    
#     # --- 6. Final Summary Table ---
#     print("\n" + "="*60)
#     print("FINAL SUMMARY - TEST SET ACCURACY")
#     print("="*60)
    
#     # Create and save the final comparison table
#     results_df = pd.DataFrame({
#         'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Paper BERT'],
#         'Test Accuracy': [test_acc_lr*100, test_acc_rf*100, test_acc_xgb*100, 93.49]
#     })
    
#     results_df.to_csv(os.path.join(RESULT_SAVE_DIR, 'model_comparison.csv'), index=False)
    
#     # Print the table to the console
#     print(results_df.to_string(index=False))
#     print("="*60)





"""
Task 2 (v3): Train ALL Classical ML Models on Extracted Features
- FIXES: Removes incorrect class balancing.
- TUNES: Uses more regularized hyperparameters.
- Model 1: Logistic Regression (Scaled Baseline)
- Model 2: Random Forest (Interpretable)
- Model 3: XGBoost (Performance + Early Stopping)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import os
import warnings

# --- Configuration ---
FEATURE_DIR = '/users/PAS2136/upadha2/factKG/Fact-or-Fiction/features'
MODEL_SAVE_DIR = 'models'
RESULT_SAVE_DIR = 'results'
# ---

def print_feature_importance(model, feature_names, top_n=15):
    """Prints the most important features for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return
        
    print(f"\n--- Top {top_n} Features ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    # Create DataFrame for saving
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(importance_df.head(top_n).to_string(index=False))
    return importance_df

def print_lr_coefficients(model, feature_names, top_n=10):
    """Prints the most impactful coefficients for Logistic Regression."""
    if not hasattr(model, 'coef_'):
        return
        
    print(f"\n--- Top {top_n} Most Impactful Features (Coefficients) ---")
    coeffs = model.coef_[0]
    coeff_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coeffs,
        'abs_coeff': np.abs(coeffs)
    }).sort_values('abs_coeff', ascending=False)
    
    print("Most Positive (Predicts 'Supported'):")
    print(coeff_df.sort_values('coefficient', ascending=False).head(top_n).to_string(index=False))
    print("\nMost Negative (Predicts 'Refuted'):")
    print(coeff_df.sort_values('coefficient', ascending=True).head(top_n).to_string(index=False))
    return coeff_df.sort_values('abs_coeff', ascending=False)

def print_confusion_matrix(y_true, y_pred, model_name=""):
    """Prints a formatted confusion matrix."""
    print(f"\nConfusion Matrix ({model_name}):")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(f"True Negatives (Refuted):  {cm[0,0]}")
    print(f"False Positives (Refuted): {cm[0,1]}")
    print(f"False Negatives (Supported): {cm[1,0]}")
    print(f"True Positives (Supported):  {cm[1,1]}")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

    print("="*60)
    print("CLASSICAL MODEL TRAINING PIPELINE (v3 - Tuned)")
    print("="*60)

    # 1. Load Data
    print("\nLoading features...")
    train_df = pd.read_csv(os.path.join(FEATURE_DIR, 'features_one_hop_train.csv'))
    val_df = pd.read_csv(os.path.join(FEATURE_DIR, 'features_one_hop_dev.csv'))
    test_df = pd.read_csv(os.path.join(FEATURE_DIR, 'features_one_hop_test.csv'))

    # 2. Pre-process Data
    
    # Smart feature cleaning
    cols_to_drop = ['has_cycle']
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    val_df = val_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

    # Separate features and labels
    feature_cols = [col for col in train_df.columns if col not in ['label', 'claim']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_val = val_df[feature_cols]
    y_val = val_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")

    # 3. Scale Data (for Logistic Regression)
    print("\nScaling features for Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_path = os.path.join(MODEL_SAVE_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to {scaler_path}")

    # --- Model 1: Logistic Regression (Baseline) ---
    print("\n" + "="*60)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("="*60)
    
    # NOTE: We keep 'class_weight=balanced' for LR, as it was our best model.
    # It seems to handle it correctly, or the data has a slight
    # imbalance that this helps with. Let's keep this as our control.
    lr_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)

    print("\n--- Test Set Results ---")
    y_test_pred_lr = lr_model.predict(X_test_scaled)
    test_acc_lr = accuracy_score(y_test, y_test_pred_lr)
    print(f"Accuracy: {test_acc_lr:.4f} ({test_acc_lr*100:.2f}%)")
    print(classification_report(y_test, y_test_pred_lr, target_names=['Refuted', 'Supported']))
    print_confusion_matrix(y_test, y_test_pred_lr, "Logistic Regression")

    lr_coeffs = print_lr_coefficients(lr_model, feature_cols)
    
    # Save artifacts
    with open(os.path.join(MODEL_SAVE_DIR, 'lr_baseline_one_hop.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
    lr_coeffs.to_csv(os.path.join(RESULT_SAVE_DIR, 'lr_feature_importance_one_hop.csv'), index=False)
    pd.DataFrame({'true_label': y_test, 'predicted_label': y_test_pred_lr}).to_csv(
        os.path.join(RESULT_SAVE_DIR, 'lr_test_predictions_one_hop.csv'), index=False
    )
    print("\n✅ Logistic Regression model and results saved.")

    # --- Model 2: Random Forest (Interpretable) ---
    print("\n" + "="*60)
    print("MODEL 2: RANDOM FOREST (Tuned)")
    print("="*60)
    print("\nTraining Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,        # More trees
        max_depth=10,            # Shallower (was 20)
        min_samples_split=50,    # More conservative (was 10)
        min_samples_leaf=20,     # More conservative (was 5)
        random_state=42,
        n_jobs=-1,
        verbose=0
        # REMOVED: class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    print("\n--- Test Set Results ---")
    y_test_pred_rf = rf_model.predict(X_test)
    test_acc_rf = accuracy_score(y_test, y_test_pred_rf)
    print(f"Accuracy: {test_acc_rf:.4f} ({test_acc_rf*100:.2f}%)")
    print(classification_report(y_test, y_test_pred_rf, target_names=['Refuted', 'Supported']))
    print_confusion_matrix(y_test, y_test_pred_rf, "Random Forest")
    
    rf_importance = print_feature_importance(rf_model, feature_cols)

    # Save artifacts
    with open(os.path.join(MODEL_SAVE_DIR, 'rf_baseline_one_hop.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    rf_importance.to_csv(os.path.join(RESULT_SAVE_DIR, 'rf_feature_importance_one_hop.csv'), index=False)
    pd.DataFrame({'true_label': y_test, 'predicted_label': y_test_pred_rf}).to_csv(
        os.path.join(RESULT_SAVE_DIR, 'rf_test_predictions_one_hop.csv'), index=False
    )
    print("\n✅ Random Forest model and results saved.")

    # --- Model 3: XGBoost (Performance + Early Stopping) ---
    print("\n" + "="*60)
    print("MODEL 3: XGBOOST (Tuned)")
    print("="*60)
    print("\nTraining XGBoost with Early Stopping...")

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,        # Fewer iterations (was 200)
        max_depth=6,             # Shallower (was 10)
        learning_rate=0.05,      # Slower learning (was 0.1)
        min_child_weight=3,      # Add regularization
        subsample=0.8,           # Add sampling
        colsample_bytree=0.8,    # Add feature sampling
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=10
        # REMOVED: scale_pos_weight=...
    )
    
    # Train with evaluation set for early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"Best iteration: {xgb_model.best_iteration}")

    print("\n--- Test Set Results ---")
    y_test_pred_xgb = xgb_model.predict(X_test)
    test_acc_xgb = accuracy_score(y_test, y_test_pred_xgb)
    print(f"Accuracy: {test_acc_xgb:.4f} ({test_acc_xgb*100:.2f}%)")
    print(classification_report(y_test, y_test_pred_xgb, target_names=['Refuted', 'Supported']))
    print_confusion_matrix(y_test, y_test_pred_xgb, "XGBoost")
    
    xgb_importance = print_feature_importance(xgb_model, feature_cols)

    # Save artifacts
    with open(os.path.join(MODEL_SAVE_DIR, 'xgb_baseline_one_hop.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    xgb_importance.to_csv(os.path.join(RESULT_SAVE_DIR, 'xgb_feature_importance_one_hop.csv'), index=False)
    pd.DataFrame({'true_label': y_test, 'predicted_label': y_test_pred_xgb}).to_csv(
        os.path.join(RESULT_SAVE_DIR, 'xgb_test_predictions_one_hop.csv'), index=False
    )
    print("\n✅ XGBoost model and results saved.")
    
    # --- 6. Final Summary Table ---
    print("\n" + "="*60)
    print("FINAL SUMMARY - TEST SET ACCURACY")
    print("="*60)
    
    # Create and save the final comparison table
    results_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest (Tuned)', 'XGBoost (Tuned)', 'Paper BERT'],
        'Test Accuracy': [
            round(test_acc_lr*100, 2), 
            round(test_acc_rf*100, 2), 
            round(test_acc_xgb*100, 2), 
            93.49
        ]
    })
    
    results_df.to_csv(os.path.join(RESULT_SAVE_DIR, 'model_comparison.csv'), index=False)
    
    # Print the table to the console
    print(results_df.to_string(index=False))
    print("="*60)