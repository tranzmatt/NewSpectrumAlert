import os
import time
import sys
from config_manager import ConfigManager
from sdr_manager import SDRManager
from scanner import Scanner
from model_manager import ModelManager
from mqtt_client import MQTTClient
from gps_manager import GPSManager
from feature_extraction import extract_features, calculate_signal_strength

class SpectrumAlert:
    """
    Main class that coordinates all components of the SpectrumAlert system.
    """
    
    def __init__(self, config_file='config.ini', lite_mode=False):
        """
        Initialize the SpectrumAlert system.
        
        Parameters:
        config_file (str): Path to the configuration file
        lite_mode (bool): Whether to use lite mode for low-resource devices
        """
        self.lite_mode = lite_mode
        
        # Initialize components
        print("Initializing SpectrumAlert...")
        
        # Configuration
        print("Loading configuration...")
        self.config_manager = ConfigManager(config_file)
        if lite_mode:
            self.config_manager.set_lite_mode(True)
        
        # SDR device
        print("Initializing SDR device...")
        self.sdr_manager = SDRManager(self.config_manager.config)
        try:
            self.sdr_manager.initialize_device()
        except Exception as e:
            print(f"Error initializing SDR device: {e}")
            self.sdr_manager = None
        
        # Scanner
        if self.sdr_manager:
            print("Initializing scanner...")
            self.scanner = Scanner(self.sdr_manager, self.config_manager)
        else:
            self.scanner = None
        
        # Model manager
        print("Initializing model manager...")
        self.model_manager = ModelManager(lite_mode)
        
        # GPS manager
        print("Initializing GPS manager...")
        receiver_lat, receiver_lon = self.config_manager.get_receiver_coordinates()
        self.gps_manager = GPSManager(
            default_lat=receiver_lat,
            default_lon=receiver_lon
        )
        self.gps_manager.start()
        
        # MQTT client
        print("Initializing MQTT client...")
        mqtt_broker, mqtt_port, mqtt_topics = self.config_manager.get_mqtt_settings()
        self.mqtt_client = MQTTClient(
            broker=mqtt_broker,
            port=mqtt_port,
            topics=mqtt_topics
        )
        
        print("SpectrumAlert initialization complete.")
    
    def gather_data(self, duration_minutes, filename=None):
        """
        Gather data for a specified duration.
        
        Parameters:
        duration_minutes (float): Duration to gather data in minutes
        filename (str, optional): Path to save the data, uses default if None
        
        Returns:
        bool: True if successful, False otherwise
        """
        if not self.scanner:
            print("Scanner not initialized. Cannot gather data.")
            return False
        
        if filename is None:
            if self.lite_mode:
                filename = 'collected_data_lite.csv'
            else:
                filename = 'collected_iq_data.csv'
        
        print(f"Gathering data for {duration_minutes} minutes...")
        try:
            self.scanner.gather_data(filename, duration_minutes, use_threading=True)
            print(f"Data gathering complete. Data saved to {filename}.")
            return True
        except Exception as e:
            print(f"Error gathering data: {e}")
            return False
    
    def train_models(self, data_file=None):
        """
        Train the RF fingerprinting and anomaly detection models.
        
        Parameters:
        data_file (str, optional): Path to the data file, uses default if None
        
        Returns:
        bool: True if successful, False otherwise
        """
        print("Training models...")
        
        try:
            # Load data
            features = self.model_manager.load_data(data_file)
            
            # Train models
            rf_model, anomaly_model = self.model_manager.train_models(features)
            
            if rf_model is None or anomaly_model is None:
                print("Model training failed.")
                return False
            
            # Save models
            success = self.model_manager.save_models()
            
            if success:
                print("Model training complete.")
                return True
            else:
                print("Error saving models.")
                return False
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def monitor(self, callback=None):
        """
        Start monitoring the spectrum for anomalies.
        
        Parameters:
        callback (callable, optional): Function to call when an anomaly is detected
        
        Returns:
        bool: True if monitoring started successfully, False otherwise
        """
        if not self.scanner:
            print("Scanner not initialized. Cannot monitor.")
            return False
        
        if not self.mqtt_client.connect():
            print("Warning: Could not connect to MQTT broker.")
        
        # Load models
        if not self.model_manager.load_models():
            print("Warning: Could not load models. Anomaly detection will not be available.")
        
        print("Starting spectrum monitoring...")
        
        # Get configuration
        ham_bands = self.config_manager.get_ham_bands()
        freq_step = self.config_manager.get_freq_step()
        runs_per_freq = self.config_manager.get_runs_per_freq()
        
        # Monitor loop
        try:
            while True:
                for band_start, band_end in ham_bands:
                    current_freq = band_start
                    while current_freq <= band_end:
                        for _ in range(runs_per_freq):
                            # Set frequency
                            self.sdr_manager.set_center_freq(current_freq)
                            
                            # Read samples
                            sample_size = 128 * 1024 if self.lite_mode else 256 * 1024
                            iq_samples = self.sdr_manager.read_samples(sample_size)
                            
                            # Extract features
                            features = extract_features(iq_samples, self.lite_mode)
                            
                            # Calculate signal strength
                            signal_strength = calculate_signal_strength(iq_samples)
                            
                            # Detect anomalies if model is available
                            is_anomaly = False
                            anomaly_score = 0
                            if self.model_manager.anomaly_model is not None:
                                is_anomaly, anomaly_score = self.model_manager.detect_anomaly(features)
                            
                            # Identify device if model is available
                            device_id = None
                            confidence = 0
                            if self.model_manager.rf_model is not None:
                                device_id, confidence = self.model_manager.identify_device(features)
                            
                            # Format frequency in MHz for display
                            freq_mhz = current_freq / 1e6
                            
                            # Print monitoring info
                            print(f"Monitoring {freq_mhz:.2f} MHz, Signal: {signal_strength:.2f} dB", end="")
                            if is_anomaly:
                                print(f", ANOMALY DETECTED (score: {anomaly_score:.4f})", end="")
                            if device_id:
                                print(f", Device: {device_id} (conf: {confidence:.2f})", end="")
                            print()
                            
                            # Publish to MQTT if connected
                            if self.mqtt_client.connected:
                                # Publish signal strength
                                self.mqtt_client.publish_signal_strength(current_freq, signal_strength)
                                
                                # Publish anomaly if detected
                                if is_anomaly:
                                    self.mqtt_client.publish_anomaly(
                                        current_freq, 
                                        features=features,
                                        confidence=anomaly_score
                                    )
                                
                                # Publish coordinates periodically
                                lat, lon, alt = self.gps_manager.get_coordinates()
                                self.mqtt_client.publish_coordinates(lat, lon, alt)
                            
                            # Call callback if provided and anomaly detected
                            if callback and is_anomaly:
                                callback(current_freq, features, anomaly_score, device_id, confidence)
                            
                            # Sleep briefly to avoid hammering the CPU
                            time.sleep(0.01)
                        
                        # Move to next frequency
                        current_freq += freq_step
                
                # Sleep briefly between full band scans
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"Error during monitoring: {e}")
            return False
        finally:
            self.cleanup()
        
        return True
    
    def analyze_single_frequency(self, frequency):
        """
        Analyze a single frequency.
        
        Parameters:
        frequency (float): Frequency to analyze in Hz
        
        Returns:
        dict: Analysis results
        """
        if not self.sdr_manager:
            print("SDR manager not initialized. Cannot analyze frequency.")
            return None
        
        # Set frequency
        self.sdr_manager.set_center_freq(frequency)
        
        # Read samples
        sample_size = 128 * 1024 if self.lite_mode else 256 * 1024
        iq_samples = self.sdr_manager.read_samples(sample_size)
        
        # Extract features
        features = extract_features(iq_samples, self.lite_mode)
        
        # Calculate signal strength
        signal_strength = calculate_signal_strength(iq_samples)
        
        # Detect anomalies if model is available
        is_anomaly = False
        anomaly_score = 0
        if self.model_manager.anomaly_model is not None:
            is_anomaly, anomaly_score = self.model_manager.detect_anomaly(features)
        
        # Identify device if model is available
        device_id = None
        confidence = 0
        if self.model_manager.rf_model is not None:
            device_id, confidence = self.model_manager.identify_device(features)
        
        # Return results
        return {
            'frequency': frequency,
            'frequency_mhz': frequency / 1e6,
            'signal_strength_db': signal_strength,
            'features': features,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'device_id': device_id,
            'device_confidence': confidence
        }
    
    def cleanup(self):
        """
        Clean up resources.
        """
        print("Cleaning up resources...")
        
        # Close SDR device
        if self.sdr_manager:
            self.sdr_manager.close()
        
        # Disconnect MQTT client
        if hasattr(self, 'mqtt_client') and self.mqtt_client:
            self.mqtt_client.disconnect()
        
        # Stop GPS manager
        if hasattr(self, 'gps_manager') and self.gps_manager:
            self.gps_manager.stop()
        
        print("Cleanup complete.")

def create_spectrum_alert(config_file='config.ini', lite_mode=False):
    """
    Factory function to create a SpectrumAlert instance.
    
    Parameters:
    config_file (str): Path to the configuration file
    lite_mode (bool): Whether to use lite mode for low-resource devices
    
    Returns:
    SpectrumAlert: Initialized SpectrumAlert instance
    """
    return SpectrumAlert(config_file, lite_mode)
