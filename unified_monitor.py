import numpy as np
import configparser
from rtlsdr import RtlSdr
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import joblib
import paho.mqtt.client as mqtt
from scipy.signal import welch
import os
import subprocess
import socket
import argparse
import time
import re
import json

try:
    import gpsd
except ImportError:
    gpsd = None
    print("GPSD module not available. GPS features will be limited.")

# Function to read and parse the config file
def read_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)

    # Parse HAM bands
    ham_bands_str = config['HAM_BANDS']['bands']
    ham_bands = []
    for band in ham_bands_str.split(','):
        start, end = band.split('-')
        ham_bands.append((float(start), float(end)))

    # Parse general settings
    freq_step = float(config['GENERAL']['freq_step'])
    sample_rate = float(config['GENERAL']['sample_rate'])
    runs_per_freq = int(config['GENERAL']['runs_per_freq'])
    lite_mode = int(config['GENERAL'].get('lite_mode', '0'))
    sdr_type = config['GENERAL'].get('sdr_type', 'rtlsdr')
    min_db = float(config['GENERAL'].get('min_db', '-40.0'))
    gain_value = config['GENERAL'].get('gain_value', 'auto')

    # Parse receiver settings
    receiver_lat = float(config['RECEIVER']['latitude'])
    receiver_lon = float(config['RECEIVER']['longitude'])

    # Parse MQTT settings
    mqtt_broker = config['MQTT'].get('broker', '127.0.0.1')
    mqtt_port = int(config['MQTT'].get('port', '1883'))
    mqtt_topics = {
        'anomalies': config['MQTT'].get('topic_anomalies', 'hamradio/anomalies'),
        'modulation': config['MQTT'].get('topic_modulation', 'hamradio/modulation'),
        'signal_strength': config['MQTT'].get('topic_signal_strength', 'hamradio/signal_strength'),
        'coordinates': config['MQTT'].get('topic_coordinates', 'hamradio/coordinates')
    }

    return (ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon,
            mqtt_broker, mqtt_port, mqtt_topics, lite_mode, sdr_type, min_db, gain_value)

# Function to load the pre-trained anomaly detection model
def load_anomaly_detection_model(model_file):
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print(f"Anomaly detection model loaded from {model_file}")
    else:
        model = IsolationForest(contamination=0.05, random_state=42)
        print(f"No pre-trained anomaly model found at {model_file}. A new model will be created.")
    return model

# Function to load the pre-trained RF fingerprinting model
def load_rf_fingerprinting_model(model_file):
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print(f"RF fingerprinting model loaded from {model_file}")
    else:
        model = RandomForestClassifier()
        print(f"No pre-trained RF fingerprinting model found at {model_file}. A new model will be created.")
    return model

def get_primary_mac():
    """Retrieves the primary MAC address (uppercase, no colons)."""
    try:
        # Get MAC address using `ip link` (Linux)
        mac_output = subprocess.check_output("ip link show | grep -m 1 'link/ether' | awk '{print $2}'",
                                             shell=True, text=True).strip()

        # Remove colons and convert to uppercase
        mac_clean = re.sub(r'[:]', '', mac_output).upper()

        return mac_clean
    except Exception as e:
        print(f"Error getting MAC address: {e}")
        return "UNKNOWNMAC"

def get_device_name():
    """
    Retrieves the device name from environment variable.
    If unavailable, falls back to 'hostname' + MAC.
    """
    # If on Balena
    device_name = os.getenv("BALENA_DEVICE_NAME_AT_INIT")

    if not device_name:
        try:
            host = subprocess.check_output("hostname", shell=True, text=True).strip()
            mac = get_primary_mac()
            device_name = f"{host}-{mac}"
        except Exception as e:
            print(f"Error getting fallback device name: {e}")
            device_name = "unknown-device"

    return device_name

def get_gps_coordinates(receiver_lat, receiver_lon):
    """
    Retrieves GPS coordinates from gpsd if GPS_SOURCE is set to 'gpsd'.
    Returns (latitude, longitude, altitude) or (None, None, None) if unavailable.
    """
    GPS_SOURCE = os.getenv("GPS_SOURCE", "none").lower()

    if GPS_SOURCE == "fixed":
        GPS_FIX_ALT = float(os.getenv("GPS_FIX_ALT", "1"))
        GPS_FIX_LAT = float(os.getenv("GPS_FIX_LAT", str(receiver_lat)))
        GPS_FIX_LON = float(os.getenv("GPS_FIX_LON", str(receiver_lon)))
        return GPS_FIX_LAT, GPS_FIX_LON, GPS_FIX_ALT

    if GPS_SOURCE == "gpsd" and gpsd is not None:
        try:
            # Connect to gpsd
            gpsd.connect(host="localhost", port=2947)

            # Get GPS data
            gps_data = gpsd.get_current()

            if gps_data is None:
                print("No GPS data available. GPS may not be active.")
                return None, None, None

            if gps_data.mode >= 2:  # 2D or 3D fix
                latitude = gps_data.lat
                longitude = gps_data.lon
                altitude = gps_data.alt if gps_data.mode == 3 else None  # Altitude available in 3D mode
                print(f"GPSD Coordinates: {latitude}, {longitude}, Alt: {altitude}m")
                return latitude, longitude, altitude
            else:
                print("No GPS fix yet.")
        except Exception as e:
            print(f"GPSD Error: {e}")
    else:
        print("No available GPS source or GPSD module not installed")

    return receiver_lat, receiver_lon, None  # Return configured coordinates if GPS is unavailable

def setup_mqtt_client(mqtt_broker, mqtt_port):
    """
    Initializes and configures the MQTT client using environment variables.
    Returns a connected MQTT client instance.
    If an error occurs, returns None.
    """
    try:
        # Load environment variables
        MQTT_BROKER = os.getenv("MQTT_BROKER", mqtt_broker)
        MQTT_PORT = int(os.getenv("MQTT_PORT", mqtt_port))
        MQTT_USER = os.getenv("MQTT_USER", None)
        MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", None)

        # TLS & CA Certificate Options
        MQTT_TLS = int(os.getenv("MQTT_TLS", 0))  # 1 = Enable TLS, 0 = Disable
        MQTT_USE_CA_CERT = int(os.getenv("MQTT_USE_CA_CERT", 0))  # 1 = Use CA Cert, 0 = Disable
        MQTT_CA_CERT = os.getenv("MQTT_CA_CERT", "/path/to/ca.crt")  # Path to CA Cert

        print(f"Configuring MQTT: {MQTT_BROKER}:{MQTT_PORT} (TLS: {MQTT_TLS}, CA Cert: {MQTT_USE_CA_CERT})")

        # Create MQTT client
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        # Enable automatic reconnect
        mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

        # Use TLS if enabled
        if MQTT_TLS:
            print("Enabling TLS for MQTT...")
            mqtt_client.tls_set(ca_certs=MQTT_CA_CERT if MQTT_USE_CA_CERT else None)

        # Define callback functions for connection management
        def on_connect(client, userdata, flags, rc, properties):
            if rc == 0:
                print("MQTT Connected Successfully!")
            else:
                print(f"MQTT Connection Failed with Code {rc}")

        def on_disconnect(client, userdata, rc, *args):
            print("MQTT Disconnected! Trying to reconnect...")
            try:
                client.reconnect()
            except Exception as e:
                print(f"MQTT Reconnect Failed: {e}")

        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect

        # Set username/password if provided
        if MQTT_USER:
            mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

        # Connect to MQTT broker
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("Connected to MQTT broker successfully!")

        return mqtt_client

    except Exception as e:
        print(f"MQTT Setup Error: {e}")
        return None

# Feature extraction for lite mode
def extract_lite_features(iq_data):
    I = np.real(iq_data)
    Q = np.imag(iq_data)
    amplitude = np.sqrt(I**2 + Q**2)

    # Basic amplitude statistics
    mean_amplitude = np.mean(amplitude)
    std_amplitude = np.std(amplitude)

    # Return only 2 features (matching the trained model)
    return [mean_amplitude, std_amplitude]

# Feature extraction for full mode
def extract_full_features(iq_data, target_num_features=None):
    I = np.real(iq_data)
    Q = np.imag(iq_data)
    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.unwrap(np.angle(iq_data))

    # Basic features
    fft_values = np.fft.fft(iq_data)
    fft_magnitude = np.abs(fft_values)
    mean_amplitude = np.mean(amplitude)
    std_amplitude = np.std(amplitude)
    mean_fft_magnitude = np.mean(fft_magnitude)
    std_fft_magnitude = np.std(fft_magnitude)

    # Higher-order statistics for RF fingerprinting
    skew_amplitude = np.mean((amplitude - mean_amplitude) ** 3) / (std_amplitude ** 3)
    kurt_amplitude = np.mean((amplitude - mean_amplitude) ** 4) / (std_amplitude ** 4)
    skew_phase = np.mean((phase - np.mean(phase)) ** 3) / (np.std(phase) ** 3)
    kurt_phase = np.mean((phase - np.mean(phase)) ** 4) / (np.std(phase) ** 4)

    # Cyclostationary features (simplified)
    cyclo_autocorr = np.abs(np.correlate(amplitude, amplitude, mode='full')[len(amplitude)//2:]).mean()

    features = [
        mean_amplitude, std_amplitude, mean_fft_magnitude, std_fft_magnitude,
        skew_amplitude, kurt_amplitude, skew_phase, kurt_phase, cyclo_autocorr
    ]

    # If target_num_features is provided, adjust the features list accordingly
    if target_num_features is not None:
        if len(features) < target_num_features:
            # Pad with zeros if fewer features than expected
            features += [0] * (target_num_features - len(features))
        elif len(features) > target_num_features:
            # Trim features if there are more than expected
            features = features[:target_num_features]

    return features

# Function to calculate signal strength
def calculate_signal_strength(iq_data):
    amplitude = np.abs(iq_data)
    signal_strength_db = 10 * np.log10(np.mean(amplitude**2))
    return signal_strength_db

# Main monitoring function
def monitor_spectrum(sdr, rf_model, anomaly_model, mqtt_client, mqtt_topics, 
                     ham_bands, freq_step, sample_rate, runs_per_freq, 
                     receiver_lat, receiver_lon, lite_mode, min_db):
    # Get the number of features the anomaly_model expects
    try:
        expected_num_features = anomaly_model.estimators_[0].n_features_in_
    except (AttributeError, IndexError):
        # Default to 2 for lite mode, 9 for full mode if unknown
        expected_num_features = 2 if lite_mode else 9
    
    # Initialize feature extraction function based on mode
    extract_features = extract_lite_features if lite_mode else extract_full_features
    
    # Initialize storage for known features if in full mode
    known_features = []
    
    # Set sample size based on mode
    sample_size = 64 * 1024 if lite_mode else 128 * 1024
    
    # Get device name for reporting
    device_name = get_device_name()
    print(f"Monitoring as device: {device_name}")
    
    try:
        while True:
            for band_start, band_end in ham_bands:
                current_freq = band_start
                while current_freq <= band_end:
                    for _ in range(runs_per_freq):
                        sdr.center_freq = current_freq
                        iq_samples = sdr.read_samples(sample_size)
                        
                        # Skip processing if signal is too weak
                        signal_strength_db = calculate_signal_strength(iq_samples)
                        if signal_strength_db < min_db:
                            print(f"Skipping {current_freq/1e6:.3f} MHz - Signal too weak: {signal_strength_db:.1f} dB")
                            continue
                        
                        # Extract features according to mode
                        if lite_mode:
                            features = extract_lite_features(iq_samples)
                        else:
                            features = extract_full_features(iq_samples, target_num_features=expected_num_features)
                        
                        # Check for anomalies
                        is_anomaly = anomaly_model.predict([features])[0] == -1
                        
                        # Get GPS coordinates
                        lat, lon, alt = get_gps_coordinates(receiver_lat, receiver_lon)
                        
                        # Format basic info
                        freq_mhz = current_freq / 1e6
                        timestamp = int(time.time())
                        freq_str = f"{freq_mhz:.3f} MHz"
                        
                        # Publish data via MQTT if client is available
                        if mqtt_client:
                            # Basic data payload
                            data_payload = {
                                "device": device_name,
                                "timestamp": timestamp,
                                "frequency": freq_mhz,
                                "signal_strength_db": signal_strength_db,
                                "latitude": lat,
                                "longitude": lon,
                                "altitude": alt if alt is not None else 0
                            }
                            
                            # Publish signal strength
                            mqtt_client.publish(
                                mqtt_topics['signal_strength'], 
                                json.dumps(data_payload)
                            )
                            
                            # Publish coordinates if available
                            if lat is not None and lon is not None:
                                mqtt_client.publish(
                                    mqtt_topics['coordinates'],
                                    json.dumps({
                                        "device": device_name,
                                        "timestamp": timestamp,
                                        "latitude": lat,
                                        "longitude": lon,
                                        "altitude": alt if alt is not None else 0
                                    })
                                )
                            
                            # Publish anomaly information if detected
                            if is_anomaly:
                                mqtt_client.publish(
                                    mqtt_topics['anomalies'],
                                    json.dumps({
                                        "device": device_name,
                                        "timestamp": timestamp,
                                        "frequency": freq_mhz,
                                        "signal_strength_db": signal_strength_db,
                                        "features": features,
                                        "latitude": lat,
                                        "longitude": lon,
                                        "altitude": alt if alt is not None else 0
                                    })
                                )
                        
                        # Print information to console
                        status = f"{'ANOMALY at ' if is_anomaly else ''}{freq_str}, Signal: {signal_strength_db:.1f} dB"
                        if is_anomaly:
                            print(f"!!! {status} !!!")
                        else:
                            print(f"Monitoring {status}")
                        
                        # In full mode, update known features and retrain if needed
                        if not lite_mode and len(known_features) > 1:
                            known_features.append(features)
                            if len(known_features) % 10 == 0:  # Retrain periodically
                                labels = [f"Device_{i % 5}" for i in range(len(known_features))]
                                rf_model.fit(known_features, labels)
                    
                    # Move to next frequency
                    current_freq += freq_step
    
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
    except Exception as e:
        print(f"Error during monitoring: {e}")

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Monitor RF spectrum for anomalies')
    parser.add_argument('-c', '--config', type=str, default='config.ini',
                        help='Path to configuration file (default: config.ini)')
    parser.add_argument('-a', '--anomaly', type=str,
                        help='Path to anomaly detection model file (if not specified, determined by lite_mode)')
    parser.add_argument('-f', '--fingerprint', type=str,
                        help='Path to RF fingerprinting model file (if not specified, determined by lite_mode)')
    parser.add_argument('-l', '--lite', action='store_true',
                        help='Force lite mode operation regardless of config setting')
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load the configuration
        (ham_bands, freq_step, sample_rate, runs_per_freq, receiver_lat, receiver_lon,
         mqtt_broker, mqtt_port, mqtt_topics, lite_mode, sdr_type, min_db, gain_value) = read_config(args.config)
        
        # Allow command line to override lite_mode
        if args.lite:
            lite_mode = True
            
        print(f"Using configuration file: {args.config}")
        print(f"Running in {'lite' if lite_mode else 'full'} mode")
        
        # Determine model files based on arguments or lite_mode
        anomaly_model_file = args.anomaly
        if anomaly_model_file is None:
            anomaly_model_file = 'anomaly_detection_model_lite.pkl' if lite_mode else 'anomaly_detection_model.pkl'
            
        fingerprint_model_file = args.fingerprint
        if fingerprint_model_file is None:
            fingerprint_model_file = 'rf_fingerprinting_model_lite.pkl' if lite_mode else 'rf_fingerprinting_model.pkl'
        
        # Load the pre-trained models
        anomaly_model = load_anomaly_detection_model(anomaly_model_file)
        rf_model = load_rf_fingerprinting_model(fingerprint_model_file)
        
        # Initialize SDR device
        print(f"Initializing {sdr_type} device...")
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        
        # Set gain based on configuration
        if gain_value == 'auto':
            sdr.gain = 'auto'
            print("Using automatic gain control")
        else:
            try:
                sdr.gain = float(gain_value)
                print(f"Setting gain to {gain_value} dB")
            except:
                sdr.gain = 'auto'
                print("Invalid gain value, using automatic gain control")
        
        # Setup MQTT client
        mqtt_client = setup_mqtt_client(mqtt_broker, mqtt_port)
        if mqtt_client is None:
            print("MQTT setup failed. Continuing without MQTT publishing.")
        
        # Start monitoring
        print(f"Starting spectrum monitoring on {len(ham_bands)} frequency bands...")
        monitor_spectrum(
            sdr, rf_model, anomaly_model, mqtt_client, mqtt_topics,
            ham_bands, freq_step, sample_rate, runs_per_freq,
            receiver_lat, receiver_lon, lite_mode, min_db
        )
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        try:
            sdr.close()
            print("SDR device closed.")
        except:
            pass
            
        try:
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
                print("MQTT client disconnected.")
        except:
            pass
