import numpy as np
import os
import csv
import time
import sys
import threading
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from config_manager import ConfigManager
from SoapySDR import Device as SoapyDevice
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

# Thread lock for safe file access
file_lock = threading.Lock()

# Shared object for header written status
header_lock = threading.Lock()
header_written = False

# Lite mode parameters
LITE_SAMPLE_SIZE = 128 * 1024  # Reduced sample size for Raspberry Pi
LITE_SAMPLE_RATE = 1.024e6  # Reduced sample rate for efficiency
LITE_RUNS_PER_FREQ = 3  # Fewer runs per frequency to save resources

# Import SoapySDR for all SDR types
from SoapySDR import Device as SoapyDevice


# Function to initialize appropriate SoapySDR device
def initialize_sdr_device(sdr_type, device_serial='-1', device_index='-1'):
    # Create device arguments dictionary based on sdr_type and identifiers
    args = {'driver': sdr_type}

    # Add serial number if provided
    if device_serial != '-1':
        args['serial'] = device_serial

    # Use device index as fallback if no serial is provided
    elif device_index != '-1':
        # Different SDR types might use different index parameters
        if sdr_type == 'rtlsdr':
            args['rtl'] = device_index
        else:
            # Generic approach for other devices
            args['device_id'] = device_index

    # Return SoapySDR Device with appropriate arguments
    return SoapyDevice(args)


# Function to extract enhanced features from IQ data
def extract_features(iq_data, lite_mode=False):
    I = np.real(iq_data)
    Q = np.imag(iq_data)
    amplitude = np.sqrt(I ** 2 + Q ** 2)  # Magnitude of the complex signal

    # Basic features (available in both modes)
    mean_amplitude = np.mean(amplitude)
    std_amplitude = np.std(amplitude)

    # If in lite mode, return only basic features
    if lite_mode:
        return [mean_amplitude, std_amplitude]

    # Full feature extraction for normal mode
    phase = np.unwrap(np.angle(iq_data))  # Unwrap the phase

    # FFT of the signal
    fft_values = np.fft.fft(iq_data)
    fft_magnitude = np.abs(fft_values)

    # Mean and standard deviation of the FFT magnitude
    mean_fft_magnitude = np.mean(fft_magnitude)
    std_fft_magnitude = np.std(fft_magnitude)

    # Skewness and kurtosis of amplitude
    if std_amplitude != 0:
        skew_amplitude = np.mean((amplitude - mean_amplitude) ** 3) / (std_amplitude ** 3)
        kurt_amplitude = np.mean((amplitude - mean_amplitude) ** 4) / (std_amplitude ** 4)
    else:
        skew_amplitude = 0
        kurt_amplitude = 0

    # Skewness and kurtosis of phase
    std_phase = np.std(phase)
    mean_phase = np.mean(phase)
    if std_phase != 0:
        skew_phase = np.mean((phase - mean_phase) ** 3) / (std_phase ** 3)
        kurt_phase = np.mean((phase - mean_phase) ** 4) / (std_phase ** 4)
    else:
        skew_phase = 0
        kurt_phase = 0

    # Cyclostationary autocorrelation (average of autocorrelation)
    if len(amplitude) > 1:
        cyclo_autocorr = np.abs(np.correlate(amplitude, amplitude, mode='full')[len(amplitude) // 2:]).mean()
    else:
        cyclo_autocorr = 0

    # Spectral entropy (FFT magnitude normalized)
    fft_magnitude_sum = np.sum(fft_magnitude)
    if fft_magnitude_sum > 0:
        normalized_fft = fft_magnitude / fft_magnitude_sum
        spectral_entropy = -np.sum(normalized_fft * np.log2(normalized_fft + 1e-12))  # Add small value to avoid log(0)
    else:
        spectral_entropy = 0

    # Peak-to-Average Power Ratio (PAPR)
    if mean_amplitude > 0:
        papr = np.max(amplitude) ** 2 / np.mean(amplitude ** 2)
    else:
        papr = 0

    # Band Energy Ratio (lower half of FFT vs total)
    fft_magnitude_half = fft_magnitude[:len(fft_magnitude) // 2]
    if fft_magnitude_sum > 0:
        band_energy_ratio = np.sum(fft_magnitude_half) / fft_magnitude_sum
    else:
        band_energy_ratio = 0

    return [
        mean_amplitude, std_amplitude, mean_fft_magnitude, std_fft_magnitude,
        skew_amplitude, kurt_amplitude, skew_phase, kurt_phase, cyclo_autocorr,
        spectral_entropy, papr, band_energy_ratio
    ]


# Function to save the collected data as a CSV
def save_data_to_csv(data, filename, lite_mode=False):
    global header_written
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with file_lock:  # Ensure thread-safe file writing
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)

            with header_lock:
                if not header_written:
                    if lite_mode:
                        writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude'])
                    else:
                        writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude', 'Mean_FFT_Magnitude',
                                         'Std_FFT_Magnitude', 'Skew_Amplitude', 'Kurt_Amplitude', 'Skew_Phase',
                                         'Kurt_Phase', 'Cyclo_Autocorr', 'Spectral_Entropy', 'PAPR',
                                         'Band_Energy_Ratio'])
                    header_written = True  # Update the flag after writing the header

            # Debug: Print the data being written to the CSV
            print(f"Writing to CSV: {data}")
            writer.writerow(data)

    print(f"Data saved to {filename}")
    if not lite_mode:
        sleep(10)  # Original delay in full mode


# Function to scan a single band (used in parallel mode)
def scan_band(band_start, band_end, config, filename, pca):
    current_freq = band_start

    # Initialize SDR device with appropriate parameters
    sdr = initialize_sdr_device(config.sdr_type, config.device_serial, config.device_index)

    # Set up the RX stream (channel 0)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, config.sample_rate)

    # Set gain if available
    if hasattr(config, 'gain_value'):
        try:
            # Some SDRs have named gain elements
            gain_elements = sdr.listGains(SOAPY_SDR_RX, 0)
            if gain_elements:
                # Set each gain element
                for element in gain_elements:
                    sdr.setGain(SOAPY_SDR_RX, 0, element, config.gain_value)
            else:
                # Or use overall gain
                sdr.setGain(SOAPY_SDR_RX, 0, config.gain_value)
        except Exception as e:
            print(f"Warning: Could not set gain: {e}")

    # Setup the RX stream
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx_stream)

    sample_size = LITE_SAMPLE_SIZE if config.lite_mode else 256 * 1024

    # Create buffer for samples
    buffer = np.zeros(sample_size, dtype=np.complex64)

    while current_freq <= band_end:
        run_features = []
        for _ in range(config.runs_per_freq):
            # Set frequency for this run
            sdr.setFrequency(SOAPY_SDR_RX, 0, current_freq)

            # Read samples
            buffer_ptr = buffer.ctypes.data
            total_samples = 0
            while total_samples < sample_size:
                samples_to_read = sample_size - total_samples
                samples_read = sdr.readStream(
                    rx_stream,
                    [buffer_ptr + total_samples * 8],  # 8 bytes per complex sample (4 for I, 4 for Q)
                    samples_to_read
                )[1]

                if samples_read <= 0:
                    break

                total_samples += samples_read

            # Process samples if we got enough data
            if total_samples > 0:
                iq_samples = buffer[:total_samples]
                features = extract_features(iq_samples, config.lite_mode)
                run_features.append(features)

        # Skip frequency if no data was collected
        if not run_features:
            print(f"Warning: No data collected at frequency {current_freq / 1e6:.3f} MHz")
            current_freq += config.freq_step
            continue

        avg_features = np.mean(run_features, axis=0)
        if pca:
            with threading.Lock():
                reduced_features = pca.transform([avg_features])  # Thread-safe PCA transformation

            # Use PCA-reduced features for saving in lite mode
            if config.lite_mode:
                data = [current_freq] + reduced_features[0].tolist()
            else:
                # Use all original features in full mode
                data = [current_freq] + avg_features.tolist()
        else:
            # No PCA, use raw features
            data = [current_freq] + avg_features.tolist()

        # Save to CSV
        save_data_to_csv(data, filename, config.lite_mode)

        # Move to the next frequency
        current_freq += config.freq_step

    # Clean up
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)


# Main function for data gathering with parallel or sequential processing
def gather_data(config, filename, duration_minutes):
    global header_written
    header_written = False
    start_time = time.time()
    duration_seconds = duration_minutes * 60

    sample_size = LITE_SAMPLE_SIZE if config.lite_mode else 256 * 1024

    # Collect initial data to fit PCA
    pca_training_data = []

    # Initialize SDR device for initial data collection
    sdr = initialize_sdr_device(config.sdr_type, config.device_serial, config.device_index)

    # Set up the RX stream (channel 0)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, config.sample_rate)

    # Set gain if available
    if hasattr(config, 'gain_value'):
        try:
            gain_elements = sdr.listGains(SOAPY_SDR_RX, 0)
            if gain_elements:
                for element in gain_elements:
                    sdr.setGain(SOAPY_SDR_RX, 0, element, config.gain_value)
            else:
                sdr.setGain(SOAPY_SDR_RX, 0, config.gain_value)
        except Exception as e:
            print(f"Warning: Could not set gain: {e}")

    # Setup the RX stream
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx_stream)

    # Create buffer for samples
    buffer = np.zeros(sample_size, dtype=np.complex64)
    buffer_ptr = buffer.ctypes.data

    # Collect initial samples for PCA training
    for band_start, band_end in config.ham_bands:
        # Set frequency
        sdr.setFrequency(SOAPY_SDR_RX, 0, band_start)

        # Read samples
        total_samples = 0
        while total_samples < sample_size:
            samples_to_read = sample_size - total_samples
            samples_read = sdr.readStream(
                rx_stream,
                [buffer_ptr + total_samples * 8],
                samples_to_read
            )[1]

            if samples_read <= 0:
                break

            total_samples += samples_read

        if total_samples > 0:
            iq_samples = buffer[:total_samples]
            features = extract_features(iq_samples, config.lite_mode)
            pca_training_data.append(features)

    # Clean up initial SDR connection
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)

    # Initialize PCA for dimensionality reduction
    num_features = len(pca_training_data[0])
    n_components = min(2 if config.lite_mode else 8, len(pca_training_data), num_features)

    pca = None
    if config.lite_mode or n_components < num_features:
        pca = PCA(n_components=n_components)
        pca.fit(pca_training_data)

    # Initialize anomaly detector for lite mode
    anomaly_detector = None
    if config.lite_mode:
        anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        anomaly_detector.fit(pca_training_data)

    # Ensure we have enough training data for PCA
    if not pca_training_data:
        print("Warning: Could not collect initial data for PCA training. Using default features.")
        # Create some synthetic data with appropriate dimensions
        feature_count = 2 if config.lite_mode else 12
        pca_training_data = [np.zeros(feature_count)]

    if config.lite_mode:
        # Sequential scanning for lite mode (resource-optimized)
        while time.time() - start_time < duration_seconds:
            for band_start, band_end in config.ham_bands:
                scan_band(band_start, band_end, config, filename, pca)
    else:
        # Parallel scanning of bands for full mode
        with ThreadPoolExecutor() as executor:
            futures = []
            for band_start, band_end in config.ham_bands:
                futures.append(executor.submit(
                    scan_band, band_start, band_end, config, filename, pca
                ))

            # Wait for all threads to finish
            for future in futures:
                future.result()


# Main execution
if __name__ == "__main__":
    try:
        # Check for lite_mode flag in command line (overrides config setting)
        cli_lite_mode = '--lite' in sys.argv
        if cli_lite_mode:
            sys.argv.remove('--lite')
            print("Command line flag: Running in lite mode for resource-constrained environments")

        # Load configuration using ConfigManager
        config_file = 'config.ini'
        if len(sys.argv) > 2:
            config_file = sys.argv[2]

        config = ConfigManager(config_file)

        # Override config lite_mode with command line if specified
        if cli_lite_mode:
            config.lite_mode = True

        # Get duration
        if len(sys.argv) > 1:
            try:
                duration = float(sys.argv[1])
            except ValueError:
                print("Invalid duration value. Using interactive input.")
                duration = float(input("Enter the duration for data gathering (in minutes): "))
        else:
            # Interactive input if no duration provided
            duration = float(input("Enter the duration for data gathering (in minutes): "))

        # Output filename based on mode
        filename = 'collected_data_lite.csv' if config.lite_mode else 'collected_iq_data.csv'

        # Print configuration summary
        print(f"Starting {'lite ' if config.lite_mode else ''}IQ data collection for {duration} minutes...")
        print(f"SDR Type: {config.sdr_type}")
        print(f"Sample rate: {config.sample_rate / 1e6:.3f} MHz, Runs per frequency: {config.runs_per_freq}")
        print(f"Scanning bands: {config.ham_bands}")
        if config.device_serial != '-1':
            print(f"Device serial: {config.device_serial}")
        if config.device_index != '-1':
            print(f"Device index: {config.device_index}")

        # List available SoapySDR devices
        try:
            print("\nAvailable SoapySDR devices:")
            available_devices = SoapyDevice.enumerate()
            for i, device in enumerate(available_devices):
                print(f"  Device {i}: {device}")
            print()
        except Exception as e:
            print(f"Warning: Could not enumerate SoapySDR devices: {e}\n")

        gather_data(config, filename, duration)
        print(f"Data collection completed. Data saved to {filename}")

    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()