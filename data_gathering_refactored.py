#!/usr/bin/env python3
import os
import sys
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA

# Import our refactored modules
from config_manager import load_config
from sdr_manager import SDRManager
from feature_extraction import extract_enhanced_features
from scanner import Scanner

# Thread lock for safe file access
file_lock = threading.Lock()
header_lock = threading.Lock()
header_written = False

def save_data_to_csv(data, filename):
    """
    Save collected data to a CSV file with thread-safe file access.
    
    Parameters:
    data (list): Data row to save (frequency and features)
    filename (str): Path to the CSV file
    """
    global header_written
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with file_lock:  # Ensure thread-safe file writing
        with open(filename, 'a', newline='') as f:
            import csv
            writer = csv.writer(f)

            with header_lock:
                if not header_written:
                    writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude', 'Mean_FFT_Magnitude', 
                                    'Std_FFT_Magnitude', 'Skew_Amplitude', 'Kurt_Amplitude', 'Skew_Phase', 
                                    'Kurt_Phase', 'Cyclo_Autocorr', 'Spectral_Entropy', 'PAPR', 'Band_Energy_Ratio'])
                    header_written = True

            # Log the data being written to CSV
            print(f"Writing to CSV: {data}")
            writer.writerow(data)

    print(f"Data saved to {filename}")
    time.sleep(1)  # Short delay to prevent overwhelming the system

def scan_band(config, sdr_manager, band_start, band_end, filename, pca=None):
    """
    Scan a frequency band and collect signal data.
    
    Parameters:
    config: Configuration object
    sdr_manager: SDR device manager
    band_start (float): Start frequency
    band_end (float): End frequency
    filename (str): CSV file to save data
    pca: Optional PCA object for dimensionality reduction
    """
    current_freq = band_start
    freq_step = config.get_freq_step()
    runs_per_freq = config.get_runs_per_freq()
    
    print(f"Starting scan of band {band_start/1e6:.2f}-{band_end/1e6:.2f} MHz")
    
    while current_freq <= band_end:
        print(f"Scanning frequency {current_freq/1e6:.3f} MHz")
        run_features = []
        for _ in range(runs_per_freq):
            sdr_manager.set_center_freq(current_freq)
            sample_size = 256 * 1024
            iq_samples = sdr_manager.read_samples(sample_size)
            
            # Extract features
            features = extract_enhanced_features(iq_samples)
            run_features.append(features)

        # Average features over runs
        avg_features = np.mean(run_features, axis=0)
        
        # Apply PCA transformation if provided
        if pca is not None:
            with threading.Lock():
                reduced_features = pca.transform([avg_features])
                
        # Use all original features for saving
        data = [current_freq] + avg_features.tolist()
        
        # Save to CSV
        save_data_to_csv(data, filename)
        
        # Move to the next frequency
        current_freq += freq_step
    
    print(f"Completed scan of band {band_start/1e6:.2f}-{band_end/1e6:.2f} MHz")

def gather_data_parallel(config, sdr_manager, filename, duration_minutes):
    """
    Gather data from all configured bands in parallel.
    
    Parameters:
    config: Configuration object
    sdr_manager: SDR device manager
    filename (str): CSV file to save data
    duration_minutes (float): Duration to gather data
    """
    start_time = time.time()
    duration_seconds = duration_minutes * 60
    ham_bands = config.get_ham_bands()

    # Collect initial data to fit PCA using the configured device
    # We're already using the correctly configured SDR manager
    pca_training_data = []
    for band_start, band_end in ham_bands:
        sdr_manager.set_center_freq(band_start)
        iq_samples = sdr_manager.read_samples(256 * 1024)
        features = extract_enhanced_features(iq_samples)
        pca_training_data.append(features)

    # Configure PCA
    num_features = len(pca_training_data[0])
    n_components = min(8, len(pca_training_data), num_features)
    pca = PCA(n_components=n_components)
    pca.fit(pca_training_data)

    # Keep gathering data until duration is reached
    while time.time() - start_time < duration_seconds:
        # Parallel scanning of bands
        with ThreadPoolExecutor() as executor:
            futures = []
            for band_start, band_end in ham_bands:
                futures.append(executor.submit(
                    scan_band, config, sdr_manager, band_start, band_end, filename, pca
                ))

            # Wait for all threads to finish
            for future in futures:
                future.result()

        # If we still have time left, do another scan
        if time.time() - start_time < duration_seconds:
            print(f"Completed scan cycle. Time remaining: {duration_seconds - (time.time() - start_time):.1f} seconds")
        else:
            print("Data gathering duration completed")

def main():
    """
    Main function to run the data gathering process.
    """
    try:
        # Parse command line arguments
        if len(sys.argv) > 1:
            duration = float(sys.argv[1])
        else:
            duration = float(input("Enter the duration for data gathering (in minutes): "))

        # Load configuration
        print("Loading configuration...")
        config = load_config('Trainer/config.ini')
        
        # Initialize SDR device
        print("Initializing SDR device...")
        sdr_manager = SDRManager(config.config)
        sdr_manager.initialize_device()
        
        # Start data gathering
        print(f"Starting data collection for {duration} minutes...")
        filename = 'collected_iq_data.csv'
        gather_data_parallel(config, sdr_manager, filename, duration)
        
        # Clean up
        sdr_manager.close()
        print("Data collection completed successfully.")
        
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
        if 'sdr_manager' in locals():
            sdr_manager.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'sdr_manager' in locals():
            sdr_manager.close()

if __name__ == "__main__":
    main()
