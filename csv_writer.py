import csv
import os
import threading
import time

# Thread lock for safe file access
file_lock = threading.Lock()

class CSVWriter:

    def __init__(self, lite_mode=True):
        self.header_written = {}  # Track header status per filename
        self.lite_mode = lite_mode
        self.header_lock = threading.Lock()

    def _check_existing_header(self, filename):
        """
        Check if the file already exists and has a header.

        Parameters:
        filename (str): CSV file path

        Returns:
        bool: True if file exists and has a header, False otherwise
        """
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            return False

        with open(filename, 'r') as f:
            try:
                first_line = f.readline().strip()
                # Check if the first line looks like our header
                if self.lite_mode:
                    return "Frequency,Mean_Amplitude,Std_Amplitude" in first_line
                else:
                    return "Frequency,Mean_Amplitude,Std_Amplitude,Mean_FFT_Magnitude" in first_line
            except Exception:
                # If we can't read the file, assume no header
                return False

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

        # Check if we've already recorded header status for this file
        with self.header_lock:
            if filename not in self.header_written:
                # First time seeing this file in this session - check it
                self.header_written[filename] = self._check_existing_header(filename)

        with file_lock:  # Ensure thread-safe file writing
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)

                with self.header_lock:
                    if not self.header_written[filename]:
                        if self.lite_mode:
                            writer.writerow(['Frequency', 'Mean_Amplitude', 'Std_Amplitude'])
                        else:
                            writer.writerow([
                                'Frequency', 'Mean_Amplitude', 'Std_Amplitude',
                                'Mean_FFT_Magnitude', 'Std_FFT_Magnitude',
                                'Skew_Amplitude', 'Kurt_Amplitude', 'Skew_Phase', 'Kurt_Phase',
                                'Cyclo_Autocorr', 'Spectral_Entropy', 'PAPR', 'Band_Energy_Ratio'
                            ])
                        self.header_written[filename] = True

                writer.writerow(data)

        print(f"Data saved to {filename}: {data[0]/1e6:.3f} MHz")
        time.sleep(1)  # Short delay to prevent overwhelming the system
