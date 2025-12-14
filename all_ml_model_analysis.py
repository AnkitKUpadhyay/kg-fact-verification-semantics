import pandas as pd
import pickle
import os
import warnings
from sklearn.metrics import accuracy_score

# --- Configuration ---
FEATURE_DIR = '/users/PAS2136/upadha2/factKG/Fact-or-Fiction/features'
MODEL_DIR = 'models'
DATA_DIR = 'data' # Assumes 'data/factkg/factkg_test.pickle'
RESULT_DIR = 'results'
# ---

def load_data(split='test'):
    """Loads features, labels, and the original claims data."""
    
    # 1. Load features
    feature_path = os.path.join(FEATURE_DIR, f"features_one_hop_{split}.csv")
    features_df = pd.read_csv(feature_path)
    
    y_true = features_df['label'].astype(int)
    X = features_df.drop(columns=['label'])
    
    # 2. Load original claims data (to get reasoning types)
    claims_path = os.path.join(DATA_DIR, f"factkg/factkg_{split}.pickle")
    with open(claims_path, 'rb') as f:
        claims_dict = pickle.load(f)
        
    # 3. Extract reasoning types in the *same order*
    reasoning_types = []
    for claim_text, metadata in claims_dict.items():
        # Get all types for this claim
        types_list = metadata.get('types', ['unknown'])
        reasoning_types.append(types_list)
        
    features_df['reasoning_types'] = reasoning_types
    
    return X, y_true, features_df, X.columns.to_list()

def analyze_by_reasoning_type(df, model_name):
    """Calculates and prints accuracy for each reasoning type."""
    
    print("\n" + "="*60)
    print(f"REASONING TYPE ANALYSIS ({model_name})")
    print("="*60)
    
    # Explode the list of types into separate rows
    # A claim with ['negation', 'multi hop'] will become two rows
    df_exploded = df.explode('reasoning_types')
    
    # Calculate accuracy for each type
    results = {}
    for r_type in df_exploded['reasoning_types'].unique():
        subset_df = df_exploded[df_exploded['reasoning_types'] == r_type]
        if len(subset_df) > 0:
            accuracy = accuracy_score(subset_df['true_label'], subset_df['predicted_label'])
            results[r_type] = {
                'accuracy': accuracy,
                'count': len(subset_df)
            }
            
    # Format and print
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.sort_values('accuracy', ascending=False)
    results_df['accuracy'] = results_df['accuracy'].apply(lambda x: f"{x*100:.1f}%")
    
    print(results_df.to_string())
    
    # Save to file
    save_path = os.path.join(RESULT_DIR, f"{model_name}_reasoning_type_analysis.csv")
    results_df.to_csv(save_path)
    print(f"\n✅ Analysis saved to {save_path}")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    # --- 1. Load Test Data and Scaler ---
    print("Loading test data and scaler...")
    X_test, y_test, test_df, feature_names = load_data('test')
    
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # --- THIS IS THE FIX ---
    # Manually drop the 'has_cycle' column if it exists to match
    # the data the scaler was trained on.
    if 'has_cycle' in X_test.columns:
        X_test = X_test.drop(columns=['has_cycle'])
        print("Dropped 'has_cycle' column from test set to match scaler.")
    # --- END FIX ---
        
    # Scale test data. Now X_test has the correct columns.
    X_test_scaled = scaler.transform(X_test)
    
    # --- 2. Load Best Model (Logistic Regression) ---
    print("Loading best classical model (LogisticRegression)...")
    model_path = os.path.join(MODEL_DIR, 'lr_baseline_one_hop.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # --- 3. Make Predictions ---
    predictions = model.predict(X_test_scaled)
    test_df['true_label'] = y_test
    test_df['predicted_label'] = predictions
    
    # --- 4. Run Analysis ---
    analyze_by_reasoning_type(test_df, "LogisticRegression")
    
    print("\nAnalysis complete.")