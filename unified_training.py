import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import joblib
import os
import csv
import configparser
import argparse
import warnings

# Function to check if a row looks like a header
def is_header_row(row):
    """Check if a row likely contains headers rather than data"""
    header_keywords = ['frequency', 'amplitude', 'mean', 'std', 'magnitude', 'fft', 
                       'skew', 'kurt', 'phase', 'cyclo', 'entropy', 'papr',
                       'ratio', 'energy']

    if not row:
        return False

    # Convert all values to lowercase for case-insensitive comparison
    row_lower = [str(val).lower() for val in row]

    # Check if any cell contains known header keywords
    for cell in row_lower:
        for keyword in header_keywords:
            if keyword in cell:
                return True

    # Check if most cells can't be converted to float (typical for headers)
    non_numeric = 0
    for val in row:
        try:
            float(val)
        except (ValueError, TypeError):
            non_numeric += 1

    # If more than half of the cells are non-numeric, likely a header
    return non_numeric > len(row) / 2

# Improved function to handle loading CSV data into features
def load_data_from_csv(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    features = []
    headers = None
    header_rows_skipped = 0
    
    print(f"Reading CSV file: {filename}")
    with open(filename, 'r') as f:
        # Read the first few lines to debug
        first_lines = [next(f) for _ in range(min(5, sum(1 for _ in open(filename))))]
        f.seek(0)  # Reset file pointer
        
        print("First few lines of the file:")
        for i, line in enumerate(first_lines):
            print(f"Line {i+1}: {line.strip()}")
        
        # Reset file pointer again
        f.seek(0)
        
        # Use CSV reader
        reader = csv.reader(f)
        
        # Get header
        try:
            headers = next(reader)
            print(f"Headers detected: {headers}")
        except StopIteration:
            raise ValueError("CSV file is empty or cannot be read.")
        
        # Process data rows
        line_number = 1
        for row in reader:
            line_number += 1
            if not row:  # Skip empty rows
                continue
                
            # Check if this row looks like another header
            if is_header_row(row):
                header_rows_skipped += 1
                warnings.warn(f"Skipping line {line_number} that appears to be another header: {row}")
                continue
                
            try:
                # Skip frequency column (first column) and convert the rest to float
                feature_values = [float(value) for value in row[1:]]
                features.append(feature_values)
            except ValueError as e:
                print(f"Error parsing line {line_number}: {row}")
                print(f"Exception: {e}")
                # Skip problematic rows but continue processing
                continue

    if not features:
        raise ValueError("No valid feature data could be extracted from the file.")
    
    print(f"Successfully loaded {len(features)} feature vectors")
    if header_rows_skipped > 0:
        print(f"Warning: Skipped {header_rows_skipped} rows that appeared to be headers")
    print(f"Feature columns: {headers[1:]}") 
    
    return np.array(features)

# Function to load configuration
def load_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Check if lite_mode is enabled
    lite_mode = config.getint('GENERAL', 'lite_mode', fallback=0)
    return config, lite_mode == 1

# Unified function to train the RF fingerprinting model based on mode
def train_rf_fingerprinting_model(features, lite_mode=False):
    # Ensure sufficient data for training
    if len(features) < 2:
        print("Not enough data to train the model.")
        return None, None

    # Different device simulation based on mode
    num_devices = 5 if lite_mode else 10
    
    # Generate labels for the entire dataset
    if lite_mode:
        # For lite mode: Simulating 5 devices
        labels = [f"Device_{i % num_devices}" for i in range(len(features))]
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    else:
        # For full mode: Split first, then simulate 10 devices
        X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)
        y_train = [f"Device_{i % num_devices}" for i in range(len(X_train))]
        y_test = [f"Device_{i % num_devices}" for i in range(len(X_test))]

    # Count samples per class for cross-validation
    class_counts = Counter(y_train)
    min_samples_per_class = min(class_counts.values())
    
    # Determine max CV splits based on mode and class sizes
    if lite_mode:
        max_cv_splits = min(3, min_samples_per_class)
        print(f"Using {max_cv_splits}-fold cross-validation (lite version).")
    else:
        max_cv_splits = min(5, min_samples_per_class)
        max_cv_splits = max(2, max_cv_splits)  # Ensure at least 2 splits
        print(f"Using {max_cv_splits}-fold cross-validation (based on smallest class size).")

    # Create the appropriate model based on mode
    if lite_mode:
        # Simple RandomForestClassifier for lite mode
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced number of trees for faster training
            max_depth=10,     # Reduced depth for lower complexity
            random_state=42,
            class_weight='balanced'  # Add class weight to handle imbalance
        )
        print("Training the RF fingerprinting model (lite version)...")
        model.fit(X_train, y_train)
    else:
        # Full version with hyperparameter tuning
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        skf = StratifiedKFold(n_splits=max_cv_splits)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, n_jobs=-1, verbose=2)
        
        print("Training the RF fingerprinting model with hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

    # Evaluate the model on test data
    y_pred = model.predict(X_test)
    print(f"Classification accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Classification Report:")
    # Fixed zero_division parameter to suppress warnings
    print(classification_report(y_test, y_pred, zero_division=0))

    # Cross-validation for performance evaluation
    skf = StratifiedKFold(n_splits=max_cv_splits)
    cv_labels = [f"Device_{i % num_devices}" for i in range(len(features))]  # Consistent labels for cross-validation
    cv_scores = cross_val_score(model, features, cv_labels, cv=skf)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores) * 100:.2f}%")

    # Train anomaly detection model
    contamination = 0.05  # Same for both modes
    print(f"Training the IsolationForest model for anomaly detection {'(lite version)' if lite_mode else ''}...")
    anomaly_detector = IsolationForest(contamination=contamination, random_state=42)
    anomaly_detector.fit(features)
    print("Anomaly detection model trained successfully.")

    return model, anomaly_detector

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train RF fingerprinting and anomaly detection models')
    parser.add_argument('-c', '--config', type=str, default='config.ini',
                        help='Path to configuration file (default: config.ini)')
    parser.add_argument('-i', '--input', type=str, 
                        help='Input CSV file containing training data (if not specified, determined by lite_mode)')
    parser.add_argument('-f', '--fingerprint', type=str,
                        help='Output file for fingerprinting model (if not specified, determined by lite_mode)')
    parser.add_argument('-a', '--anomaly', type=str,
                        help='Output file for anomaly detection model (if not specified, determined by lite_mode)')
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration to determine mode
    config, lite_mode = load_config(args.config)
    print(f"Using configuration file: {args.config}")
    print(f"Running in {'lite' if lite_mode else 'full'} mode")
    
    # Determine input and output files based on arguments or lite_mode
    data_file = args.input
    if data_file is None:
        data_file = 'collected_data_lite.csv' if lite_mode else 'collected_iq_data.csv'
    print(f"Loading data from {data_file}...")

    fingerprint_model_file = args.fingerprint
    if fingerprint_model_file is None:
        fingerprint_model_file = 'rf_fingerprinting_model_lite.pkl' if lite_mode else 'rf_fingerprinting_model.pkl'
    
    anomaly_model_file = args.anomaly
    if anomaly_model_file is None:
        anomaly_model_file = 'anomaly_detection_model_lite.pkl' if lite_mode else 'anomaly_detection_model.pkl'

    try:
        features = load_data_from_csv(data_file)
        print(f"Loaded {len(features)} samples with {len(features[0])} features each")
        if len(features) > 0:
            print(f"Feature range: min={np.min(features):.4f}, max={np.max(features):.4f}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # Train the RF fingerprinting and anomaly detection models
    model, anomaly_model = train_rf_fingerprinting_model(features, lite_mode)

    # Save the trained models to files for future use
    if model is not None:
        joblib.dump(model, fingerprint_model_file)
        print(f"Model saved to {fingerprint_model_file}")
    
    if anomaly_model is not None:
        joblib.dump(anomaly_model, anomaly_model_file)
        print(f"Anomaly detection model saved to {anomaly_model_file}")
