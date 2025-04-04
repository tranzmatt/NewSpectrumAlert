import os
from configparser import ConfigParser


class ConfigManager:
    """
    Centralized configuration management for SpectrumAlert.
    Handles reading, validating, and providing access to configuration parameters.
    """

    def __init__(self, config_file='config.ini'):
        """
        Initialize the configuration manager.
        
        Parameters:
        config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.config = ConfigParser()
        self.ham_bands = []
        self.freq_step = 0
        self.sample_rate = 0
        self.runs_per_freq = 0
        self.receiver_lat = 0
        self.receiver_lon = 0
        self.mqtt_broker = ""
        self.mqtt_port = 0
        self.mqtt_topics = {}
        self.sdr_type = "rtlsdr"
        self.lite_mode = False

        # Load the configuration
        self.load_config()

    def load_config(self):
        """
        Load and parse the configuration file.
        
        Raises:
        FileNotFoundError: If the configuration file is not found
        ValueError: If the configuration file has invalid or missing values
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")

        self.config.read(self.config_file)

        # Validate required sections
        required_sections = ['HAM_BANDS', 'GENERAL', 'RECEIVER', 'MQTT']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required section '{section}' missing in the configuration file.")

        # Parse HAM bands
        self.parse_ham_bands()

        # SDR Settings
        self.device_serial = self.config['GENERAL'].get('device_serial', '-1')
        self.device_index = self.config['GENERAL'].get('device_index', '-1')

        # Parse general settings
        self.freq_step = float(self.config['GENERAL'].get('freq_step', '500e3'))
        self.sample_rate = float(self.config['GENERAL'].get('sample_rate', '2.048e6'))
        self.runs_per_freq = int(self.config['GENERAL'].get('runs_per_freq', '5'))
        self.sdr_type = self.config['GENERAL'].get('sdr_type', 'rtlsdr')
        self.lite_mode = self.config['GENERAL'].getboolean('lite_mode', True)

        # Parse receiver settings
        self.receiver_lat = float(os.getenv("GPS_FIX_LAT", self.config['RECEIVER'].get('latitude', '0')))
        self.receiver_lon = float(os.getenv("GPS_FIX_LON", self.config['RECEIVER'].get('longitude', '0')))

        # Parse MQTT settings
        self.mqtt_broker = self.config['MQTT'].get('broker', 'localhost')
        self.mqtt_port = int(self.config['MQTT'].get('port', '1883'))
        self.mqtt_topics = {
            'anomalies': self.config['MQTT'].get('topic_anomalies', 'hamradio/anomalies'),
            'modulation': self.config['MQTT'].get('topic_modulation', 'hamradio/modulation'),
            'signal_strength': self.config['MQTT'].get('topic_signal_strength', 'hamradio/signal_strength'),
            'coordinates': self.config['MQTT'].get('topic_coordinates', 'hamradio/coordinates')
        }

    def parse_ham_bands(self):
        """
        Parse the HAM bands from the configuration.
        
        Raises:
        ValueError: If the HAM bands are not properly formatted
        """
        ham_bands_str = self.config['HAM_BANDS'].get('bands', None)
        if ham_bands_str is None:
            raise ValueError("Missing 'bands' entry in 'HAM_BANDS' section.")

        self.ham_bands = []
        for band in ham_bands_str.split(','):
            try:
                start, end = band.split('-')
                self.ham_bands.append((float(start), float(end)))
            except ValueError:
                raise ValueError(f"Invalid frequency range format: {band}. Expected 'start-end'.")

    def set_lite_mode(self, enabled=True):
        """
        Enable or disable lite mode for low-resource devices.
        
        Parameters:
        enabled (bool): True to enable lite mode, False to disable
        """
        self.lite_mode = enabled
        if enabled:
            # Override some settings for lite mode
            self.sample_rate = min(self.sample_rate, 1.024e6)  # Reduced sample rate
            self.runs_per_freq = min(self.runs_per_freq, 3)  # Fewer runs per frequency

    def get_ham_bands(self):
        """Get the parsed HAM bands."""
        return self.ham_bands

    def get_freq_step(self):
        """Get the frequency step."""
        return self.freq_step

    def get_sample_rate(self):
        """Get the sample rate."""
        return self.sample_rate

    def get_runs_per_freq(self):
        """Get the number of runs per frequency."""
        return self.runs_per_freq

    def get_receiver_coordinates(self):
        """Get the receiver coordinates (latitude, longitude)."""
        return (self.receiver_lat, self.receiver_lon)

    def get_mqtt_settings(self):
        """Get the MQTT settings (broker, port, topics)."""
        return (self.mqtt_broker, self.mqtt_port, self.mqtt_topics)

    def get_sdr_type(self):
        """Get the SDR type."""
        return self.sdr_type

    def get_device_serial(self):
        """Get the SDR device_serial."""
        return self.device_serial

    def get_device_index(self):
        """Get the SDR device_index."""
        return self.device_index

    def is_lite_mode(self):
        """Check if lite mode is enabled."""
        return self.lite_mode

    def get_all_settings(self):
        """
        Get all configuration settings as a dictionary.
        
        Returns:
        dict: All configuration settings
        """
        return {
            'ham_bands': self.ham_bands,
            'freq_step': self.freq_step,
            'sample_rate': self.sample_rate,
            'runs_per_freq': self.runs_per_freq,
            'receiver_lat': self.receiver_lat,
            'receiver_lon': self.receiver_lon,
            'mqtt_broker': self.mqtt_broker,
            'mqtt_port': self.mqtt_port,
            'mqtt_topics': self.mqtt_topics,
            'sdr_type': self.sdr_type,
            'lite_mode': self.lite_mode
        }


def load_config(config_file='config.ini', lite_mode=False):
    """
    Factory function to load configuration and create a ConfigManager object.
    
    Parameters:
    config_file (str): Path to the configuration file
    lite_mode (bool): Whether to enable lite mode for low-resource devices
    
    Returns:
    ConfigManager: Initialized configuration manager
    """
    config_manager = ConfigManager(config_file)
    if lite_mode:
        config_manager.set_lite_mode(True)
    return config_manager
