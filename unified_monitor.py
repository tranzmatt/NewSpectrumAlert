#!/usr/bin/env python3
"""
Refactored unified_monitor.py script that uses the modular classes from other files.
This script monitors RF spectrum for anomalies using SDR and machine learning models.
"""

import argparse
import time

from config_manager import load_config
from feature_extraction import FeatureExtractor
from gps_manager import GPSManager
from model_manager import ModelManager
from mqtt_client import MQTTClient
from scanner import Scanner
from sdr_manager import SDRManager


def monitor_spectrum(config, anomaly_model_file=None, rf_model_file=None):
    """
    Main function to monitor the RF spectrum for anomalies.
    
    Parameters:
    config (ConfigManager): Configuration manager instance
    anomaly_model_file (str): Path to anomaly detection model file
    rf_model_file (str): Path to RF fingerprinting model file
    """
    try:
        # Use the provided config
        print(f"Using lite_mode setting from config: {config.is_lite_mode()}")

        # Initialize model manager
        print("Initializing model manager...")
        model_manager = ModelManager(config.is_lite_mode())

        # Override model filenames if specified
        if anomaly_model_file:
            model_manager.anomaly_model_file = anomaly_model_file
        if rf_model_file:
            model_manager.rf_model_file = rf_model_file

        # Load models
        if not model_manager.load_models():
            print("Warning: Failed to load one or more models. Monitoring may be limited.")

        # Initialize SDR manager
        print("Initializing SDR device...")
        sdr_manager = SDRManager(config.config)
        sdr_manager.initialize_device()

        # Initialize scanner
        print("Initializing scanner...")
        scanner = Scanner(sdr_manager, config)

        # Initialize feature extractor
        print("Initializing feature extractor...")
        feature_extractor = FeatureExtractor(config)

        # Initialize GPS manager
        print("Initializing GPS manager...")
        receiver_lat, receiver_lon = config.get_receiver_coordinates()
        gps_manager = GPSManager(default_lat=receiver_lat, default_lon=receiver_lon)
        gps_manager.start()

        # Initialize MQTT client
        print("Initializing MQTT client...")
        mqtt_broker, mqtt_port, mqtt_topics = config.get_mqtt_settings()
        mqtt_client = MQTTClient(
            broker=mqtt_broker,
            port=mqtt_port,
            topics=mqtt_topics
        )

        # Connect to MQTT broker
        if not mqtt_client.connect():
            print("Warning: Failed to connect to MQTT broker. Will continue without publishing.")

        # Get configuration for monitoring
        ham_bands = config.get_ham_bands()
        freq_step = config.get_freq_step()
        runs_per_freq = config.get_runs_per_freq()
        min_db = float(config.config['GENERAL'].get('min_db', '-40.0'))

        # Determine sample size based on lite mode
        sample_size = 128 * 1024 if config.is_lite_mode() else 256 * 1024

        print(f"Starting spectrum monitoring (lite mode: {config.is_lite_mode()})...")
        print(f"Monitoring {len(ham_bands)} frequency bands with {freq_step / 1e6:.3f} MHz steps")

        # Main monitoring loop
        try:
            # Publish initial GPS coordinates
            if mqtt_client.connected:
                lat, lon, alt = gps_manager.get_coordinates()
                mqtt_client.publish_coordinates(lat, lon, alt)

            while True:
                for band_start, band_end in ham_bands:
                    current_freq = band_start
                    while current_freq <= band_end:
                        for _ in range(runs_per_freq):
                            # Set frequency
                            sdr_manager.set_center_freq(current_freq)

                            # Read samples
                            iq_samples = sdr_manager.read_samples(sample_size)

                            # Calculate signal strength
                            signal_strength = feature_extractor.calculate_signal_strength(iq_samples)

                            # Skip processing if signal is too weak
                            if signal_strength < min_db:
                                print(
                                    f"Skipping {current_freq / 1e6:.3f} MHz - Signal too weak: {signal_strength:.1f} dB")
                                break

                            # Extract features
                            features = feature_extractor.extract_features(iq_samples)

                            # Detect anomalies
                            is_anomaly, anomaly_score = model_manager.detect_anomaly(features)

                            # Identify device if RF model is available
                            device_id, confidence = model_manager.identify_device(features)

                            # Format frequency for display
                            freq_mhz = current_freq / 1e6

                            # Print monitoring information
                            status = f"{freq_mhz:.3f} MHz, Signal: {signal_strength:.1f} dB"
                            if is_anomaly:
                                print(f"!!! ANOMALY at {status}, Score: {anomaly_score:.4f} !!!")
                            else:
                                print(f"Monitoring {status}")

                            if device_id:
                                print(f"  Identified as {device_id} (confidence: {confidence:.2f})")

                            # Publish to MQTT if connected
                            if mqtt_client.connected:
                                # Publish signal strength
                                mqtt_client.publish_signal_strength(current_freq, signal_strength)

                                # Publish coordinates periodically
                                lat, lon, alt = gps_manager.get_coordinates()
                                mqtt_client.publish_coordinates(lat, lon, alt)

                                # Publish anomaly if detected
                                if is_anomaly:
                                    mqtt_client.publish_anomaly(
                                        current_freq,
                                        features=features,
                                        confidence=anomaly_score
                                    )

                            # Small delay to prevent hammering the CPU
                            time.sleep(0.01)

                        # Move to next frequency
                        current_freq += freq_step

                # Small delay between full band scans
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        finally:
            # Clean up resources
            print("Cleaning up resources...")
            sdr_manager.close()
            if mqtt_client.connected:
                mqtt_client.disconnect()
            gps_manager.stop()

    except Exception as e:
        print(f"Error during monitoring: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Monitor RF spectrum for anomalies')
    parser.add_argument('-c', '--config', type=str, default='config.ini',
                        help='Path to configuration file (default: config.ini)')
    parser.add_argument('-a', '--anomaly', type=str,
                        help='Path to anomaly detection model file (if not specified, determined by lite_mode)')
    parser.add_argument('-f', '--fingerprint', type=str,
                        help='Path to RF fingerprinting model file (if not specified, determined by lite_mode)')
    # Removed lite mode argument as it's defined in config.ini
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # First load config to determine lite_mode for proper model defaults
    config = load_config(args.config)
    lite_mode = config.is_lite_mode()
    print(f"Using configuration file: {args.config}")
    print(f"Running in {'lite' if lite_mode else 'full'} mode")

    # Set default model files based on lite_mode if not provided as arguments
    anomaly_model_file = args.anomaly
    if anomaly_model_file is None:
        anomaly_model_file = 'anomaly_detection_model_lite.pkl' if lite_mode else 'anomaly_detection_model.pkl'
        print(f"Using default anomaly model file: {anomaly_model_file}")

    rf_model_file = args.fingerprint
    if rf_model_file is None:
        rf_model_file = None if lite_mode else 'rf_fingerprinting_model.pkl'
        if rf_model_file:
            print(f"Using default RF fingerprinting model file: {rf_model_file}")
        else:
            print("RF fingerprinting not used in lite mode")

    # Start monitoring with the already loaded config
    monitor_spectrum(
        config=config,
        anomaly_model_file=anomaly_model_file,
        rf_model_file=rf_model_file
    )
