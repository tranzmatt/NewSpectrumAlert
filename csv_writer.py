import csv
import os
import threading
import time

# Thread lock for safe file access
file_lock = threading.Lock()

class CSVWriter:

    def __init__(self, lite_mode = True ):
        self.header_written = False
        self.lite_mode = lite_mode
        self.header_lock = threading.Lock()

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

        print(f"Data saved to {filename}: {data[0]/1e6:.3f} MHz")
        time.sleep(1)  # Short delay to prevent overwhelming the system


