import numpy as np
import os
import csv
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

class ModelManager:
    """
    Class for handling RF fingerprinting and anomaly detection model training and inference.
    """
    
    def __init__(self, lite_mode=False):
        """
        Initialize the model manager.
        
        Parameters:
        lite_mode (bool): Whether to use lite mode for low-resource devices
        """
        self.lite_mode = lite_mode
        self.rf_model = None
        self.anomaly_model = None
        
        # Set appropriate filenames based on mode
        if lite_mode:
            self.rf_model_file = 'rf_fingerprinting_model_lite.pkl'
            self.anomaly_model_file = 'anomaly_detection_model_lite.pkl'
            self.data_file = 'collected_data_lite.csv'
        else:
            self.rf_model_file = 'rf_fingerprinting_model.pkl'
            self.anomaly_model_file = 'anomaly_detection_model.pkl'
            self.data_file = 'collected_iq_data.csv'
    
    def load_data(self, filename=None):
        """
        Load data from CSV file.
        
        Parameters:
        filename (str, optional): Path to the CSV file, uses default if None
        
        Returns:
        numpy.ndarray: Feature data
        
        Raises:
        FileNotFoundError: If the file doesn't exist
        """
        if filename is None:
            filename = self.data_file
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        
        features = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row
            for row in reader:
                try:
                    # Extract feature data (all columns except the first which is frequency)
                    features.append([float(value) for value in row[1:]])
                except (ValueError, IndexError) as e:
                    print(f"Error parsing row: {row}. Error: {e}")
        
        return np.array(features)
    
    def train_models(self, features=None, rf_params=None, anomaly_params=None):
        """
        Train RF fingerprinting and anomaly detection models.
        
        Parameters:
        features (numpy.ndarray, optional): Feature data, loaded from file if None
        rf_params (dict, optional): Parameters for the RF model
        anomaly_params (dict, optional): Parameters for the anomaly detection model
        
        Returns:
        tuple: (rf_model, anomaly_model) - Trained models
        """
        # Load data if not provided
        if features is None:
            try:
                features = self.load_data()
                print(f"Loaded {len(features)} samples from {self.data_file}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return None, None
        
        # Check if we have enough data
        if len(features) < 2:
            print("Not enough data to train the models.")
            return None, None
        
        # Train RF fingerprinting model
        self.rf_model = self._train_rf_model(features, rf_params)
        
        # Train anomaly detection model
        self.anomaly_model = self._train_anomaly_model(features, anomaly_params)
        
        return self.rf_model, self.anomaly_model
    
    def _train_rf_model(self, features, params=None):
        """
        Train a Random Forest model for RF fingerprinting.
        
        Parameters:
        features (numpy.ndarray): Feature data
        params (dict, optional): Parameters for the RF model
        
        Returns:
        RandomForestClassifier: Trained model
        """
        # Generate simulated device labels (for demonstration)
        # In a real scenario, you would have actual device labels
        num_devices = 5 if self.lite_mode else 10
        labels = [f"Device_{i % num_devices}" for i in range(len(features))]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Count samples per class for cross-validation
        class_counts = Counter(y_train)
        min_samples_per_class = min(class_counts.values())
        max_cv_splits = min(5 if not self.lite_mode else 3, min_samples_per_class)
        max_cv_splits = max(2, max_cv_splits)  # Ensure at least 2 splits
        
        print(f"Using {max_cv_splits}-fold cross-validation")
        
        # Define model parameters
        if self.lite_mode:
            # Simple model for lite mode
            model = RandomForestClassifier(
                n_estimators=50,  # Reduced number of trees for faster training
                max_depth=10,     # Reduced depth for lower complexity
                random_state=42
            )
        else:
            # Full model with hyperparameter tuning
            base_model = RandomForestClassifier(random_state=42)
            
            # Default parameters if none provided
            if params is None:
                params = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            
            # Set up hyperparameter tuning
            skf = StratifiedKFold(n_splits=max_cv_splits)
            model = GridSearchCV(
                estimator=base_model,
                param_grid=params,
                cv=skf,
                n_jobs=-1,
                verbose=2
            )
        
        # Train the model
        print("Training the RF fingerprinting model...")
        model.fit(X_train, y_train)
        
        # If we did GridSearchCV, get the best model
        if not self.lite_mode and hasattr(model, 'best_estimator_'):
            best_model = model.best_estimator_
            print(f"Best parameters: {model.best_params_}")
        else:
            best_model = model
        
        # Evaluate the model
        y_pred = best_model.predict(X_test)
        print(f"Classification accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation for more reliable performance evaluation
        cv_labels = [f"Device_{i % num_devices}" for i in range(len(features))]
        skf = StratifiedKFold(n_splits=max_cv_splits)
        cv_scores = cross_val_score(best_model, features, cv_labels, cv=skf)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean cross-validation score: {np.mean(cv_scores) * 100:.2f}%")
        
        return best_model
    
    def _train_anomaly_model(self, features, params=None):
        """
        Train an Isolation Forest model for anomaly detection.
        
        Parameters:
        features (numpy.ndarray): Feature data
        params (dict, optional): Parameters for the anomaly detection model
        
        Returns:
        IsolationForest: Trained model
        """
        # Set default parameters if none provided
        if params is None:
            params = {'contamination': 0.05, 'random_state': 42}
        
        # Create and train the anomaly detection model
        print("Training the anomaly detection model...")
        model = IsolationForest(**params)
        model.fit(features)
        
        return model
    
    def save_models(self, rf_model_file=None, anomaly_model_file=None):
        """
        Save the trained models to files.
        
        Parameters:
        rf_model_file (str, optional): Path for the RF model file
        anomaly_model_file (str, optional): Path for the anomaly model file
        
        Returns:
        bool: True if successful, False otherwise
        """
        if rf_model_file is None:
            rf_model_file = self.rf_model_file
        
        if anomaly_model_file is None:
            anomaly_model_file = self.anomaly_model_file
        
        success = True
        
        # Save RF fingerprinting model
        if self.rf_model is not None:
            try:
                joblib.dump(self.rf_model, rf_model_file)
                print(f"RF fingerprinting model saved to {rf_model_file}")
            except Exception as e:
                print(f"Error saving RF model: {e}")
                success = False
        else:
            print("RF model not trained, cannot save.")
            success = False
        
        # Save anomaly detection model
        if self.anomaly_model is not None:
            try:
                joblib.dump(self.anomaly_model, anomaly_model_file)
                print(f"Anomaly detection model saved to {anomaly_model_file}")
            except Exception as e:
                print(f"Error saving anomaly model: {e}")
                success = False
        else:
            print("Anomaly model not trained, cannot save.")
            success = False
        
        return success
    
    def load_models(self, rf_model_file=None, anomaly_model_file=None):
        """
        Load trained models from files.
        
        Parameters:
        rf_model_file (str, optional): Path for the RF model file
        anomaly_model_file (str, optional): Path for the anomaly model file
        
        Returns:
        bool: True if successful, False otherwise
        """
        if rf_model_file is None:
            rf_model_file = self.rf_model_file
        
        if anomaly_model_file is None:
            anomaly_model_file = self.anomaly_model_file
        
        success = True
        
        # Load RF fingerprinting model
        if os.path.exists(rf_model_file):
            try:
                self.rf_model = joblib.load(rf_model_file)
                print(f"RF fingerprinting model loaded from {rf_model_file}")
            except Exception as e:
                print(f"Error loading RF model: {e}")
                success = False
        else:
            print(f"RF model file {rf_model_file} not found.")
            success = False
        
        # Load anomaly detection model
        if os.path.exists(anomaly_model_file):
            try:
                self.anomaly_model = joblib.load(anomaly_model_file)
                print(f"Anomaly detection model loaded from {anomaly_model_file}")
            except Exception as e:
                print(f"Error loading anomaly model: {e}")
                success = False
        else:
            print(f"Anomaly model file {anomaly_model_file} not found.")
            success = False
        
        return success
    
    def detect_anomaly(self, features):
        """
        Detect if a signal is anomalous.
        
        Parameters:
        features (list or numpy.ndarray): Feature vector
        
        Returns:
        tuple: (is_anomaly, score) where is_anomaly is a boolean and score is the anomaly score
        """
        if self.anomaly_model is None:
            print("Anomaly model not loaded. Call load_models() first.")
            return False, 0
        
        # Convert to numpy array if needed
        if not isinstance(features, np.ndarray):
            features = np.array(features).reshape(1, -1)
        elif len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Predict anomaly
        prediction = self.anomaly_model.predict(features)
        score = self.anomaly_model.score_samples(features)
        
        # In Isolation Forest, -1 indicates an anomaly, 1 indicates normal
        is_anomaly = prediction[0] == -1
        
        return is_anomaly, score[0]
    
    def identify_device(self, features):
        """
        Identify the device based on RF fingerprinting.
        
        Parameters:
        features (list or numpy.ndarray): Feature vector
        
        Returns:
        tuple: (device_id, confidence) where device_id is a string and confidence is a float
        """
        if self.rf_model is None:
            print("RF model not loaded. Call load_models() first.")
            return None, 0
        
        # Convert to numpy array if needed
        if not isinstance(features, np.ndarray):
            features = np.array(features).reshape(1, -1)
        elif len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Predict device
        prediction = self.rf_model.predict(features)
        
        # Get prediction probabilities if available
        confidence = 0
        if hasattr(self.rf_model, 'predict_proba'):
            proba = self.rf_model.predict_proba(features)
            confidence = np.max(proba)
        
        return prediction[0], confidence
