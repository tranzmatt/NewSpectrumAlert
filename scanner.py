import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA

from feature_extraction import FeatureExtractor

from csv_writer import CSVWriter
from config_manager import ConfigManager
from sdr_manager import SDRManager

class Scanner:
    """
    A class for scanning frequency bands and collecting signal data.
    """
    
    def __init__(self, sdr_manager: SDRManager, config: ConfigManager):
        """
        Initialize the scanner.
        
        Parameters:
        sdr_manager: SDR device manager
        config: Configuration manager
        """
        self.sdr_manager = sdr_manager
        print(f"Scanner SDR manager {sdr_manager}")
        self.config = config
        print(f"Scanner Config manager {config}")
        self.ham_bands = self.config.get_ham_bands()
        print(f"Scanner ham_bands {self.ham_bands}")
        self.freq_step = config.get_freq_step()
        print(f"Scanner freq_step {self.freq_step}")
        self.sample_rate = config.get_sample_rate()
        print(f"Scanner sample_rate {self.sample_rate}")
        self.runs_per_freq = config.get_runs_per_freq()
        print(f"Scanner runs_per_freq {self.runs_per_freq}")
        self.lite_mode = config.is_lite_mode()
        print(f"Scanner lite_mode {self.lite_mode}")
        self.csv_writer = CSVWriter(self.lite_mode)
        print(f"Scanner csv_writer {self.csv_writer}")
        self.feature_extractor = FeatureExtractor(config)
        print(f"Scanner feature_extractor {self.feature_extractor}")

        self.sample_size = 128 * 1024 if self.lite_mode else 256 * 1024
        
        # Get min_db from config if available
        if hasattr(config.config, 'get'):
            # It's a ConfigParser object
            self.min_db = float(config.config.get('GENERAL', 'min_db', fallback='-40.0'))
        else:
            # It's a dictionary
            self.min_db = float(config.config.get('min_db', -40.0))
        self.header_written = False
        self.header_lock = threading.Lock()
        self.pca = None

    def initialize_pca(self, n_components=8, min_samples=8, max_attempts=100):
        """
        Initialize PCA for dimensionality reduction using valid feature vectors
        from multiple frequencies across the bands.

        Parameters:
        n_components (int): Max number of PCA components (default: 8)
        min_samples (int): Minimum number of valid samples required (default: 8)
        max_attempts (int): Max total sampling attempts to avoid infinite loops
        """
        print(f"🔍 Collecting at least {min_samples} valid samples to initialize PCA...")
        pca_training_data = []
        attempts = 0
        skipped = 0

        while len(pca_training_data) < min_samples and attempts < max_attempts:
            for band_start, band_end in self.ham_bands:
                if len(pca_training_data) >= min_samples:
                    break

                # Try multiple frequencies within each band instead of just band_start
                # Sample at beginning, middle, and end of band
                frequencies = [
                    band_start,
                    band_start + (band_end - band_start) / 2,
                    band_end - self.freq_step
                ]

                for freq in frequencies:
                    if len(pca_training_data) >= min_samples or attempts >= max_attempts:
                        break

                    self.sdr_manager.set_center_freq(freq)
                    print(f"Reading frequency {freq / 1e6:.3f} MHz...")

                    # Add a small delay to let the SDR settle
                    time.sleep(0.1)

                    iq_samples = self.sdr_manager.read_samples(self.sample_size)
                    if iq_samples is None or len(iq_samples) == 0:
                        print(f"⚠️ No IQ samples received for {freq / 1e6:.3f} MHz – skipping.")
                        skipped += 1
                        attempts += 1
                        continue

                    features = self.feature_extractor.extract_features(iq_samples)
                    if np.isnan(features).any() or np.isinf(features).any():
                        print(f"⚠️ Skipping invalid feature set at {freq / 1e6:.3f} MHz:", features)
                        skipped += 1
                        attempts += 1
                        continue

                    pca_training_data.append(features)
                    print(f"✅ Valid sample collected at {freq / 1e6:.3f} MHz ({len(pca_training_data)}/{min_samples})")
                    attempts += 1

        if len(pca_training_data) < min_samples:
            # Instead of raising an exception, which would stop the program,
            # use a warning and proceed with whatever samples we have
            print(f"⚠️ Warning: Only collected {len(pca_training_data)} valid samples (needed {min_samples})")
            if len(pca_training_data) == 0:
                print("❌ PCA initialization failed: No valid samples collected.")
                return None

        # Set appropriate number of components
        num_features = len(pca_training_data[0])
        adjusted_components = min(n_components, len(pca_training_data), num_features)

        self.pca = PCA(n_components=adjusted_components)
        self.pca.fit(pca_training_data)

        print(f"✅ PCA initialized with {len(pca_training_data)} samples "
              f"({skipped} skipped), using {adjusted_components} components.")

        return self.pca

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
                iq_samples = self.sdr_manager.read_samples(self.sample_size)
                features = self.feature_extractor.extract_features(iq_samples)
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
            #print(f"Deactivating stream...")
            self.sdr_manager.deactivate_stream()
            self.csv_writer.save_data_to_csv(data, filename)
            #print(f"Activating stream...")
            self.sdr_manager.activate_stream()

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
    
    def timed_scan(self, filename, duration_minutes, use_threading=True):
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
        iq_samples = self.sdr_manager.read_samples(self.sample_size)
        signal_strength = self.feature_extractor.calculate_signal_strength(iq_samples)
        
        return signal_strength > threshold_db
