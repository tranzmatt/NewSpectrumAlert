#!/usr/bin/env python3
"""
Test script for the SpectrumAlert refactored module.
This demonstrates how to use the new modules independently and together.
"""

import os
import time
import sys

# Import refactored modules
from config_manager import ConfigManager, load_config
from sdr_manager import SDRManager, get_sdr_device
from scanner import Scanner
from model_manager import ModelManager
from mqtt_client import MQTTClient
from gps_manager import GPSManager
from feature_extraction import extract_features, calculate_signal_strength
from spectrum_alert import create_spectrum_alert

# Import numpy for test_sdr_manager
import numpy as np

def test_config_manager(config):
    """Test the ConfigManager module."""
    print("\n=== Testing ConfigManager ===")

    # Print configuration details
    print(f"HAM Bands: {config.get_ham_bands()}")
    print(f"Frequency Step: {config.get_freq_step()} Hz")
    print(f"Sample Rate: {config.get_sample_rate()} Hz")
    print(f"Runs per Frequency: {config.get_runs_per_freq()}")
    print(f"Receiver Coordinates: {config.get_receiver_coordinates()}")
    print(f"MQTT Settings: {config.get_mqtt_settings()}")
    print(f"SDR Type: {config.get_sdr_type()}")
    print(f"Lite Mode: {config.is_lite_mode()}")
    
    # Test setting lite mode
    config.set_lite_mode(True)
    print(f"After setting lite mode:")
    print(f"Sample Rate: {config.get_sample_rate()} Hz")
    print(f"Runs per Frequency: {config.get_runs_per_freq()}")
    print(f"Lite Mode: {config.is_lite_mode()}")

def test_sdr_manager(config: ConfigManager):
    """Test the SDRManager module."""
    print("\n=== Testing SDRManager ===")
    
    try:
        # Initialize SDR device using config
        sdr = SDRManager(config.config)
        sdr.initialize_device()
        
        # Print device info
        print(f"Using SDR type: {sdr.sdr_type}")
        print(f"Sample rate: {sdr.sample_rate} Hz")
        print(f"Gain setting: {sdr.gain}")
        if sdr.device_serial:
            print(f"Device serial: {sdr.device_serial}")
        if sdr.device_index is not None:
            print(f"Device index: {sdr.device_index}")
        
        # Read some samples
        print("Reading samples...")
        sdr.set_center_freq(145e6)  # 145 MHz
        samples = sdr.read_samples(1024 * 64)
        
        # Calculate some basic statistics
        import numpy as np
        signal_strength = 10 * np.log10(np.mean(np.abs(samples)**2))
        print(f"Read {len(samples)} samples at 145 MHz")
        print(f"Signal strength: {signal_strength:.2f} dB")
        
        # Close the device
        sdr.close()
        print("SDR device closed")
    except Exception as e:
        print(f"Error testing SDR manager: {e}")

def test_feature_extraction(config: ConfigManager):
    """Test the feature extraction module."""
    print("\n=== Testing Feature Extraction ===")
    
    try:

        # Initialize SDR device using config
        sdr = SDRManager(config.config)
        sdr.initialize_device()
        
        # Read some samples
        print("Reading samples for feature extraction...")
        sdr.set_center_freq(145e6)  # 145 MHz
        samples = sdr.read_samples(1024 * 64)
        
        # Extract basic features
        basic_features = extract_features(samples, lite_mode=True)
        print(f"Basic features: {basic_features}")
        
        # Extract enhanced features
        enhanced_features = extract_features(samples, lite_mode=False)
        print(f"Enhanced features: {enhanced_features}")
        
        # Calculate signal strength
        signal_strength = calculate_signal_strength(samples)
        print(f"Signal strength: {signal_strength:.2f} dB")
        
        # Close the device
        sdr.close()
    except Exception as e:
        print(f"Error testing feature extraction: {e}")

def test_scanner(config: ConfigManager):
    """Test the Scanner module."""
    print("\n=== Testing Scanner ===")
    
    try:

        # Initialize SDR device with config
        sdr = SDRManager(config.config)
        sdr.initialize_device()
        
        # Create scanner
        scanner = Scanner(sdr, config)
        
        # Scan a specific band for a short duration
        print("Scanning 144-146 MHz band for 10 seconds...")
        
        # We'll simulate this by only processing the first few frequencies
        band_start, band_end = 144e6, 146e6
        current_freq = band_start
        freq_step = config.get_freq_step()
        
        start_time = time.time()
        end_time = start_time + 10  # 10 seconds
        
        while time.time() < end_time and current_freq <= band_end:
            print(f"Scanning {current_freq/1e6:.2f} MHz...")
            sdr.set_center_freq(current_freq)
            iq_samples = sdr.read_samples(1024 * 64)
            features = extract_features(iq_samples, config.is_lite_mode())
            signal_strength = calculate_signal_strength(iq_samples)
            print(f"Signal strength: {signal_strength:.2f} dB")
            current_freq += freq_step
            time.sleep(0.1)  # Brief pause to avoid hammering the CPU
        
        print("Scanning complete")
        
        # Close the device
        sdr.close()
    except Exception as e:
        print(f"Error testing scanner: {e}")

def test_spectrum_alert(config_file: str):
    """Test the SpectrumAlert main class."""
    print("\n=== Testing SpectrumAlert ===")
    
    try:
        # Create a SpectrumAlert instance using real config
        try:
            spectrum_alert = create_spectrum_alert(config_file)
            print(f"Using config from {config_file}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise
        
        # Analyze a single frequency
        print("Analyzing 145 MHz...")
        results = spectrum_alert.analyze_single_frequency(145e6)
        
        if results:
            print(f"Frequency: {results['frequency_mhz']:.2f} MHz")
            print(f"Signal strength: {results['signal_strength_db']:.2f} dB")
            print(f"Anomaly detected: {results['is_anomaly']}")
            if results['is_anomaly']:
                print(f"Anomaly score: {results['anomaly_score']:.4f}")
            if results['device_id']:
                print(f"Device: {results['device_id']} (confidence: {results['device_confidence']:.2f})")
        
        # Clean up
        spectrum_alert.cleanup()
        print("SpectrumAlert cleanup complete")
    except Exception as e:
        print(f"Error testing SpectrumAlert: {e}")

if __name__ == "__main__":

    # Try to load the config from file
    try:
        config = load_config('config.ini')
        print("Using config from config.ini")
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise

    # Run tests based on command line arguments
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        if test_name == 'config':
            test_config_manager(config)
        elif test_name == 'sdr':
            test_sdr_manager(config)
        elif test_name == 'features':
            test_feature_extraction(config)
        elif test_name == 'scanner':
            test_scanner(config)
        elif test_name == 'spectrum':
            test_spectrum_alert('config.ini')
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: config, sdr, features, scanner, spectrum")
    else:
        # Run only the config test by default as it's the safest
        test_config_manager(config)
        
        print("\nTo run other tests, specify the test name:")
        print("  python test_spectrum_alert.py config")
        print("  python test_spectrum_alert.py sdr")
        print("  python test_spectrum_alert.py features")
        print("  python test_spectrum_alert.py scanner")
        print("  python test_spectrum_alert.py spectrum")
