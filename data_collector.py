# data_collector.py
from scanner import Scanner
from gps_manager import GPSManager
from config_manager import ConfigManager

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
            self.scanner.gather_data(filename, duration_minutes, use_threading=True)
            print(f"Data gathering complete. Data saved to {filename}.")
            return True
        except Exception as e:
            print(f"Error gathering data: {e}")
            return False

