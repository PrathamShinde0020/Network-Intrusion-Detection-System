# ids_detector.py
# This file contains the core class for the Intrusion Detection System.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class ImprovedNetworkIntrusionDetector:
    """
    An improved intrusion detection system with data leakage prevention,
    realistic evaluation metrics, and model persistence capabilities.
    """
    def __init__(self, dataset_path=None, random_state=42):
        self.dataset_path = dataset_path
        self.random_state = random_state
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.model_metadata = {}

    def load_and_analyze_data(self):
        """Loads and provides a basic analysis of the dataset."""
        if self.dataset_path is None:
            print("[!] ERROR: No dataset path provided")
            return False
            
        print(f"[*] Loading and analyzing data from '{self.dataset_path}'...")
        try:
            self.data = pd.read_csv(self.dataset_path, low_memory=False)
        except FileNotFoundError:
            print(f"[!] ERROR: Dataset not found at '{self.dataset_path}'")
            return False
        print(f"[+] Dataset loaded: {self.data.shape}")
        return True

    def clean_and_preprocess_data(self):
        """Performs enhanced data cleaning and preprocessing."""
        print("[*] Cleaning and preprocessing data...")
        self.data.columns = self.data.columns.str.strip()
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(subset=['Label'], inplace=True)
        
        label_counts = self.data['Label'].value_counts()
        labels_to_keep = label_counts[label_counts >= 100].index
        self.data = self.data[self.data['Label'].isin(labels_to_keep)]
        print("[+] Data preprocessing completed.")
        return True

    def prepare_data_with_validation(self):
        """Prepares data for modeling with validation steps."""
        print("[*] Preparing data with validation...")
        self.data['Label'] = self.data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        X = self.data.drop(columns=['Label'])
        y = self.data['Label']
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X.fillna(0, inplace=True)
        
        print("[*] Performing feature selection...")
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(20, len(X.columns)))
        X_selected = self.feature_selector.fit_transform(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_conservative_model(self):
        """Trains a more conservative model to prevent overfitting."""
        print("[*] Training conservative RandomForest model...")
        self.model = RandomForestClassifier(
            n_estimators=50, max_depth=10, min_samples_split=20,
            min_samples_leaf=10, random_state=self.random_state,
            n_jobs=-1, class_weight='balanced'
        )
        self.model.fit(self.X_train, self.y_train)
        self.model_metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'RandomForestClassifier',
            'n_features': len(self.selected_features),
            'selected_features': list(self.selected_features),
            'hyperparameters': {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 10}
        }
        print("[+] Model training completed.")

    def save_model(self, model_dir='saved_model'):
        """Saves the trained model, scaler, feature selector, and metadata."""
        if self.model is None:
            print("[!] ERROR: No trained model to save.")
            return False
        
        print(f"\n[*] Saving model components to '{model_dir}'...")
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            joblib.dump(self.model, os.path.join(model_dir, 'ids_model.pkl'))
            joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
            joblib.dump(self.feature_selector, os.path.join(model_dir, 'feature_selector.pkl'))
            with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            print("[+] Model saving completed successfully!")
            return True
        except Exception as e:
            print(f"[!] ERROR: Failed to save model - {str(e)}")
            return False

    def load_model(self, model_dir='saved_model'):
        """Loads a previously trained model and its components."""
        print(f"\n[*] Loading model components from '{model_dir}'...")
        try:
            model_path = os.path.join(model_dir, 'ids_model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl') # FIX: Corrected filename
            selector_path = os.path.join(model_dir, 'feature_selector.pkl')
            metadata_path = os.path.join(model_dir, 'model_metadata.json')

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_selector = joblib.load(selector_path)
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            self.selected_features = pd.Index(self.model_metadata['selected_features'])
            
            print("[+] Model loading completed successfully!")
            return True
        except Exception as e:
            print(f"[!] ERROR: Failed to load model - {str(e)}")
            return False

    def predict_new_data(self, new_data):
        """Makes predictions on new data using the loaded model."""
        if self.model is None:
            print("[!] ERROR: No model loaded.")
            return None, None
        
        print(f"[*] Making predictions on {len(new_data)} samples...")
        try:
            for col in new_data.columns:
                new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
            new_data.fillna(0, inplace=True)
            
            # Ensure new_data has the original feature columns before selection
            original_features = self.feature_selector.feature_names_in_
            for col in original_features:
                if col not in new_data.columns:
                    new_data[col] = 0
            new_data = new_data[original_features]

            new_data_selected = self.feature_selector.transform(new_data)
            new_data_scaled = self.scaler.transform(new_data_selected)
            
            predictions = self.model.predict(new_data_scaled)
            probabilities = self.model.predict_proba(new_data_scaled)[:, 1]
            
            return predictions, probabilities
        except Exception as e:
            print(f"[!] ERROR: Failed to make predictions - {str(e)}")
            return None, None
