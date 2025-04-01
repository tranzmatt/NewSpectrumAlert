import numpy as np
import time
import os
import csv
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA

from feature_extraction import extract_features, calculate_signal_strength

# Thread lock for safe file access
file_lock = threading.Lock()

class Scanner:
    """
    A class for scanning frequency bands and collecting signal data.
    """
    
    def __init__(self, sdr_manager, config_manager):
        """
        Initialize the scanner.
        
        Parameters:
        sdr_manager: SDR device manager
        config_manager: Configuration manager
        """
        self.sdr_manager = sdr_manager
        self.config_manager = config_manager
        self.ham_bands = config_manager.get_ham_bands()
        self.freq_step = config_manager.get_freq_step()
        self.sample_rate = config_manager.get_sample_rate()
        self.runs_per_freq = config_manager.get_runs_per_freq()
        self.lite_mode = config_manager.is_lite_mode()
        
        # Get min_db from config if available
        if hasattr(config_manager.config, 'get'):
            # It's a ConfigParser object
            self.min_db = float(config_manager.config.get('GENERAL', 'min_db', fallback='-40.0'))
        else:
            # It's a dictionary
            self.min_db = float(config_manager.config.get('min_db', -40.0))
        self.header_written = False
        self.header_lock = threading.Lock()
        self.pca = None
    
    def initialize_pca(self, n_components=8):
        """
        Initialize PCA for dimensionality reduction.
        
        Parameters:
        n_components (int): Number of components for PCA
        """
        # Collect initial data to fit PCA
        pca_training_data = []
        for band_start, band_end in self.ham_bands:
            self.sdr_manager.set_center_freq(band_start)
            sample_size = 128 * 1024 if self.lite_mode else 256 * 1024
            iq_samples = self.sdr_manager.read_samples(sample_size)
            features = extract_features(iq_samples, self.lite_mode)
            pca_training_data.append(features)

        # Determine appropriate number of components
        num_features = len(pca_training_data[0])
        n_components = min(n_components, len(pca_training_data), num_features)

        # Fit PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(pca_training_data)
    
    def save_data_to_csv(self, data, filename):
        """
        Save collected data to a CSV file.
        
        Parameters:
        data (list): Data to save
        filename (str): CSV file path
        """
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with file_lock:  # Ensure thread-safe file writing
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                
                with self.header_lock:
                    if not self.header_written:
                        if self.lite_mode:
                            writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude'])
                        else:
                            writer.writerow([
                                'Frequency', 'Mean_Amplitude', 'Std_Amplitude', 
                                'Mean_FFT_Magnitude', 'Std_FFT_Magnitude',
                                'Skew_Amplitude', 'Kurt_Amplitude', 'Skew_Phase', 'Kurt_Phase', 
                                'Cyclo_Autocorr', 'Spectral_Entropy', 'PAPR', 'Band_Energy_Ratio'
                            ])
                        self.header_written = True
                
                writer.writerow(data)
    
    def scan_band(self, band_start, band_end, filename):
        """
        Scan a frequency band and collect signal data.
        
        Parameters:
        band_start (float): Start frequency of the band
        band_end (float): End frequency of the band
        filename (str): CSV file to save the data
        """
        current_freq = band_start
        
        while current_freq <= band_end:
            run_features = []
            for _ in range(self.runs_per_freq):
                self.sdr_manager.set_center_freq(current_freq)
                sample_size = 128 * 1024 if self.lite_mode else 256 * 1024
                iq_samples = self.sdr_manager.read_samples(sample_size)
                features = extract_features(iq_samples, self.lite_mode)
                run_features.append(features)

            # Average features over runs
            avg_features = np.mean(run_features, axis=0)
            
            # Apply PCA if initialized
            if self.pca is not None:
                with threading.Lock():
                    reduced_features = self.pca.transform([avg_features])[0]
                data = [current_freq] + reduced_features.tolist()
            else:
                data = [current_freq] + avg_features.tolist()
            
            # Save to CSV
            self.save_data_to_csv(data, filename)
            
            # Move to the next frequency
            current_freq += self.freq_step
    
    def scan_all_bands(self, filename, use_threading=True):
        """
        Scan all configured frequency bands.
        
        Parameters:
        filename (str): CSV file to save the data
        use_threading (bool): Whether to use multi-threading for scanning
        """
        if use_threading:
            with ThreadPoolExecutor() as executor:
                futures = []
                for band_start, band_end in self.ham_bands:
                    futures.append(
                        executor.submit(self.scan_band, band_start, band_end, filename)
                    )
                
                # Wait for all threads to finish
                for future in futures:
                    future.result()
        else:
            for band_start, band_end in self.ham_bands:
                self.scan_band(band_start, band_end, filename)
    
    def gather_data(self, filename, duration_minutes, use_threading=True):
        """
        Gather data for a specified duration.
        
        Parameters:
        filename (str): CSV file to save the data
        duration_minutes (float): Duration to gather data in minutes
        use_threading (bool): Whether to use multi-threading for scanning
        """
        start_time = time.time()
        duration_seconds = duration_minutes * 60
        
        # Initialize PCA if not already initialized
        if self.pca is None:
            self.initialize_pca()
        
        # Keep scanning until the duration is reached
        while time.time() - start_time < duration_seconds:
            self.scan_all_bands(filename, use_threading)
            
            # Sleep briefly to avoid hammering the CPU
            time.sleep(0.1)
    
    def detect_signal(self, frequency, threshold_db=None):
        """
        Detect if a signal is present at a given frequency.
        
        Parameters:
        frequency (float): Frequency to check
        threshold_db (float, optional): Signal strength threshold in dB, uses min_db from config if None
        
        Returns:
        bool: True if signal detected, False otherwise
        """
        if threshold_db is None:
            threshold_db = self.min_db
            
        self.sdr_manager.set_center_freq(frequency)
        sample_size = 128 * 1024 if self.lite_mode else 256 * 1024
        iq_samples = self.sdr_manager.read_samples(sample_size)
        signal_strength = calculate_signal_strength(iq_samples)
        
        return signal_strength > threshold_db
