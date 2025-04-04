# data_collector.py
from config_manager import ConfigManager
from gps_manager import GPSManager
from scanner import Scanner


class DataCollector:
    """
    Handles SDR data collection over time, optionally with GPS metadata.
    Designed for reuse in training, testing, and data labeling.
    """

    def __init__(self, config: ConfigManager, scanner: Scanner, gps_manager: GPSManager):
        """
        :param scanner: Instance of Scanner for SDR access
        :param gps_manager: (Optional) Instance of GPSManager for location tagging
        :param lite_mode: If True, defaults to lighter data format
        """
        self.config = config
        self.scanner = scanner
        self.gps_manager = gps_manager
        self.lite_mode = self.config.lite_mode

    def gather_data_once(self, include_location=False):
        """
        Gather a single IQ sample with optional GPS location.
        :return: IQ data, or (IQ data, location) if include_location is True
        """
        iq_data = self.scanner.scan()
        if include_location and self.gps_manager:
            location = self.gps_manager.get_location()
            return iq_data, location
        return iq_data

    def gather_data(self, duration_minutes, filename=None):
        """
        Gather data for a specified duration and save to a file.

        :param duration_minutes: How long to scan for (in minutes)
        :param filename: Optional output filename. Defaults based on lite_mode.
        :return: True on success, False on failure
        """
        if not self.scanner:
            print("Scanner not initialized. Cannot gather data.")
            return False

        if filename is None:
            filename = 'collected_data_lite.csv' if self.lite_mode else 'collected_iq_data.csv'

        print(f"Gathering data for {duration_minutes} minutes...")
        try:
            self.scanner.timed_scan(filename, duration_minutes, use_threading=True)
            print(f"Data gathering complete. Data saved to {filename}.")
            return True
        except Exception as e:
            print(f"Error gathering data: {e}")
            return False

    # Main function for data gathering with parallel or sequential processing
    def new_gather_data(sdr_class, config, filename, duration_minutes):
        global header_written
        header_written = False
        start_time = time.time()
        duration_seconds = duration_minutes * 60

        sample_size = LITE_SAMPLE_SIZE if config.lite_mode else 256 * 1024

        # Collect initial data to fit PCA
        pca_training_data = []
        sdr = sdr_class()
        sdr.sample_rate = config.sample_rate

        # Set device-specific parameters if available
        if hasattr(sdr, 'gain') and hasattr(config, 'gain_value'):
            sdr.gain = config.gain_value

        # Set device serial if provided and supported
        if config.device_serial != '-1' and hasattr(sdr, 'serial_number'):
            sdr.serial_number = config.device_serial

        # Set device index if provided and no serial is set
        if config.device_serial == '-1' and config.device_index != '-1' and hasattr(sdr, 'device_index'):
            sdr.device_index = int(config.device_index)

        for band_start, band_end in config.ham_bands:
            sdr.center_freq = band_start
            iq_samples = sdr.read_samples(sample_size)
            features = extract_features(iq_samples, config.lite_mode)
            pca_training_data.append(features)
        sdr.close()

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

        if config.lite_mode:
            # Sequential scanning for lite mode (resource-optimized)
            while time.time() - start_time < duration_seconds:
                for band_start, band_end in config.ham_bands:
                    scan_band(sdr_class, band_start, band_end, config, filename, pca)
        else:
            # Parallel scanning of bands for full mode
            with ThreadPoolExecutor() as executor:
                futures = []
                for band_start, band_end in config.ham_bands:
                    futures.append(executor.submit(
                        scan_band, sdr_class, band_start, band_end, config, filename, pca
                    ))

                # Wait for all threads to finish
                for future in futures:
                    future.result()
