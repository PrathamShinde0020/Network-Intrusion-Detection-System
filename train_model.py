# train_model.py
# This script runs the full training pipeline for the IDS model.

from ids_detector import ImprovedNetworkIntrusionDetector
import os

def main():
    """
    Main function to execute the training pipeline.
    """
    print("="*70)
    print("🚀 STARTING INTRUSION DETECTION SYSTEM TRAINING PIPELINE")
    print("="*70)

    # --- Configuration ---
    # Use the absolute path to your dataset file
    DATASET_PATH = r'C:\Users\Prathamesh\Downloads\Data_Science\IDS\dataset\ids_data.csv'
    
    # Verify file exists before starting
    if not os.path.exists(DATASET_PATH):
        print(f"\n[!] CRITICAL ERROR: The dataset file was not found at '{DATASET_PATH}'")
        return

    # --- Pipeline Execution ---
    # 1. Initialize the detector
    detector = ImprovedNetworkIntrusionDetector(dataset_path=DATASET_PATH, random_state=42)
    
    # 2. Load and preprocess data
    if detector.load_and_analyze_data():
        if detector.clean_and_preprocess_data():
            # 3. Prepare data and train the model
            detector.prepare_data_with_validation()
            detector.train_conservative_model()
            
            # 4. Save the final model and components
            detector.save_model()
            
            print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            print("\n❌ PIPELINE FAILED: Preprocessing step failed.")
    else:
        print("\n❌ PIPELINE FAILED: Data loading step failed.")

if __name__ == "__main__":
    main()
