# predict.py
# This script simulates the real-world use of the trained IDS model on new data.
# FINAL VERSION: Includes proper fix for the feature name mismatch error.

import pandas as pd
import numpy as np
import joblib
import os
import argparse
import json

def predict_traffic(input_file, model_dir='saved_model'):
    """
    Loads a trained model and its components to predict on new network traffic data.

    Args:
        input_file (str): Path to the new CSV data to be analyzed.
        model_dir (str): Directory where the model components are saved.
    """
    # --- 1. Define Paths and Load Components ---
    print("[*] Loading trained model and preprocessors...")
    try:
        model_path = os.path.join(model_dir, 'ids_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        selector_path = os.path.join(model_dir, 'ids_model_feature_selector.pkl')
        metadata_path = os.path.join(model_dir, 'ids_model_metadata.json')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_selector = joblib.load(selector_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    except FileNotFoundError as e:
        print(f"[!] ERROR: A required model file was not found: {e}")
        print("[!] Please run the training script first to generate the model files.")
        return

    # --- 2. Load and Prepare New Data ---
    print(f"[*] Loading new traffic data from '{input_file}'...")
    try:
        new_data = pd.read_csv(input_file, low_memory=False)
        original_data = new_data.copy() # Keep for the final report
    except FileNotFoundError:
        print(f"[!] ERROR: Input file not found at '{input_file}'")
        return

    # --- 3. Preprocess New Data (Must match the training pipeline) ---
    print("[*] Preprocessing new data...")
    
    # Clean column names to match training
    new_data.columns = new_data.columns.str.strip()
    
    # Handle infinite values and fill NaNs
    new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_data.fillna(0, inplace=True)

    # Drop the 'Label' column if it exists in the input file
    if 'Label' in new_data.columns:
        new_data = new_data.drop(columns=['Label'])

    # --- PROPER FIX: Get the original training feature names from metadata ---
    if 'training_features' in metadata:
        training_features = metadata['training_features']
        print(f"[*] Found {len(training_features)} training features in metadata.")
    else:
        # Fallback: try to get feature names from the selector (this may not work for all selectors)
        print("[!] WARNING: No training features found in metadata. Attempting fallback method...")
        if hasattr(feature_selector, 'feature_names_in_'):
            training_features = feature_selector.feature_names_in_.tolist()
        else:
            print("[!] ERROR: Cannot determine original training features. Please ensure training script saves feature names to metadata.")
            return
    
    # Check for missing columns in the new data and add them with a value of 0
    missing_cols = set(training_features) - set(new_data.columns)
    if missing_cols:
        print(f"[!] WARNING: Missing {len(missing_cols)} features in new data. Adding them with value 0.")
        for c in missing_cols:
            new_data[c] = 0
    
    # Check for extra columns in new data
    extra_cols = set(new_data.columns) - set(training_features)
    if extra_cols:
        print(f"[!] INFO: Found {len(extra_cols)} extra features in new data. They will be ignored.")
    
    # Reorder the columns of the new data to match the exact order from training
    new_data = new_data[training_features]
    print(f"[*] Data aligned with training features: {new_data.shape}")
    # --- END PROPER FIX ---

    # a) Apply Feature Selection using the loaded selector
    print(f"[*] Applying feature selection...")
    new_data_selected = feature_selector.transform(new_data)
    
    # b) Apply Scaling using the loaded scaler
    print(f"[*] Scaling selected features...")
    new_data_scaled = scaler.transform(new_data_selected)

    # --- 4. Make Predictions ---
    print("[*] Making predictions...")
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)[:, 1] # Probability of being an ATTACK

    # --- 5. Report the Verdict ---
    print("\n" + "="*50)
    print("✅ PREDICTION SUMMARY")
    print("="*50)
    
    predicted_labels = ['ATTACK' if p == 1 else 'BENIGN' for p in predictions]
    
    original_data['Predicted_Label'] = predicted_labels
    original_data['Attack_Confidence'] = probabilities
    
    print(original_data['Predicted_Label'].value_counts())
    
    output_filename = 'prediction_results.csv'
    original_data.to_csv(output_filename, index=False)
    print(f"\n[+] Detailed results saved to '{output_filename}'")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict network intrusions using a trained model.")
    parser.add_argument(
        '--input', 
        required=True, 
        help="Path to the input CSV file with new traffic data."
    )
    args = parser.parse_args()

    predict_traffic(args.input)