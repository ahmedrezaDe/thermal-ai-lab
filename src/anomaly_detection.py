"""
Anomaly Detection Module

This module provides algorithms for detecting anomalies in thermal system data,
including statistical methods, machine learning approaches, and real-time detection.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Ensure robust path resolution relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# The physics-informed feature set. 
# NOTE: 'cooling_coefficient' is intentionally excluded to prevent data leakage.
# The model must infer the fault from observable physics, not the hidden mechanical state.
FEATURES = ['temperature_c', 'power_w', 'dTdt']

def train_baseline_model():
    """
    Trains the Isolation Forest and Scaler entirely on normal operating data.
    """
    df_normal = pd.read_csv(os.path.join(RESULTS_DIR, 'normal_data.csv'))
    
    # Standardize features for consistent preprocessing.
    # Isolation Forest itself is not strongly dependent on feature scale.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_normal[FEATURES])
    
    # Initialize Isolation Forest. 
    # Contamination is very low (0.5%) because the training data is fundamentally healthy.
    model = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
    model.fit(X_train)
    
    return model, scaler

def score_anomalies(df, model, scaler):
    """
    Applies the pre-trained scaler and model to a new dataset.
    """
    # Apply the training scaler without refitting on the evaluation data.
    X_test = scaler.transform(df[FEATURES])
    
    # Predict: IsolationForest returns 1 for normal, -1 for anomaly
    raw_predictions = model.predict(X_test)
    
    # Map to our standard schema: 0 for normal, 1 for anomaly
    df['ml_prediction'] = np.where(raw_predictions == -1, 1, 0)
    
    # Get the raw anomaly score (lower/negative means more anomalous)
    df['anomaly_score'] = model.decision_function(X_test)
    
    return df

def main():
    print("Initializing Stage 3: ML Anomaly Detection...")
    
    # 1. Train on Normal Data
    print("Training Isolation Forest on baseline data...")
    model, scaler = train_baseline_model()
    
    # 2. Evaluate Cooling Degradation Fault
    df_cooling = pd.read_csv(os.path.join(RESULTS_DIR, 'fault_cooling.csv'))
    df_cooling_scored = score_anomalies(df_cooling, model, scaler)
    df_cooling_scored.to_csv(os.path.join(RESULTS_DIR, 'scored_fault_cooling.csv'), index=False)
    
    print("\n--- Fault 1: Cooling Degradation Detection ---")
    print(classification_report(df_cooling_scored['label'], df_cooling_scored['ml_prediction'], 
                                target_names=['Normal', 'Anomaly']))
    
    # 3. Evaluate Compute Spike Fault
    df_compute = pd.read_csv(os.path.join(RESULTS_DIR, 'fault_compute.csv'))
    df_compute_scored = score_anomalies(df_compute, model, scaler)
    df_compute_scored.to_csv(os.path.join(RESULTS_DIR, 'scored_fault_compute.csv'), index=False)
    
    print("\n--- Fault 2: Compute Spike Detection ---")
    print(classification_report(df_compute_scored['label'], df_compute_scored['ml_prediction'], 
                                target_names=['Normal', 'Anomaly']))

if __name__ == "__main__":
    main()