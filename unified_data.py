#!/usr/bin/env python3
"""
Refactored unified_data.py script that uses the modular classes from other files.
This script collects data from SDR and saves it to a CSV file.
"""

import argparse

from config_manager import load_config
from scanner import Scanner
from sdr_manager import SDRManager


def scan_band(scanner, band_start, band_end, freq_step, filename):
    """
    Scan a frequency band and collect signal data.

    Parameters:
    scanner (Scanner): Scanner instance
    band_start (float): Start frequency of the band
    band_end (float): End frequency of the band
    freq_step (float): Step size for frequency increments
    filename (str): CSV file to save the data
    """
    scanner.scan_band(band_start, band_end, filename)


def gather_data(config, filename, duration_minutes, use_threading=True):
    """
    Gather RF data for a specified duration and save to a CSV file.

    Parameters:
    config (ConfigManager): Configuration manager instance
    filename (str): Output CSV filename
    duration_minutes (float): Duration to gather data in minutes
    use_threading (bool): Whether to use multi-threading for scanning
    """
    try:
        # Initialize SDR device
        print(f"Initializing SDR device...")
        sdr_manager = SDRManager(config.config)
        sdr_manager.initialize_device()

        # Create scanner with feature extractor
        print(f"Creating scanner...")
        scanner = Scanner(sdr_manager, config)

        # Initialize PCA if needed for dimensionality reduction
        print(f"Initializing PCA...")
        scanner.initialize_pca()

        # Start the timed scan
        print(f"Starting data collection for {duration_minutes} minutes...")
        scanner.timed_scan(filename, duration_minutes, use_threading=use_threading)

        # Clean up
        sdr_manager.close()
        print(f"Data collection completed. Data saved to {filename}")

    except Exception as e:
        print(f"Error during data collection: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to parse arguments and start data collection."""
    parser = argparse.ArgumentParser(description="RF Data Collector using SDR")
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file")
    parser.add_argument("-d", "--duration", type=float, default=10,
                        help="Duration to collect data in minutes")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV filename")
    parser.add_argument("--lite", action="store_true",
                        help="Run in lite mode for resource-constrained environments")

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)

        # Determine whether to use threading - automatically disable in lite mode
        use_threading = not config.is_lite_mode()
        if config.is_lite_mode():
            print("Config setting: Running in lite mode for resource-constrained environments")
            print("Threading disabled in lite mode to conserve resources")
        else:
            print("Using multi-threading for band scanning")

        # Determine output filename
        if args.output:
            filename = args.output
        else:
            filename = 'collected_data_lite.csv' if config.is_lite_mode() else 'collected_iq_data.csv'

        # Get duration interactively if not provided
        duration = args.duration

        # Print configuration summary
        print(f"Starting {'lite ' if config.is_lite_mode() else ''}IQ data collection for {duration} minutes...")
        print(f"SDR Type: {config.get_sdr_type()}")
        print(f"Sample rate: {config.get_sample_rate() / 1e6:.3f} MHz")
        print(f"Runs per frequency: {config.get_runs_per_freq()}")
        print(f"HAM bands: {config.get_ham_bands()}")
        print(f"Output file: {filename}")

        # Start data collection
        gather_data(config, filename, duration, use_threading=use_threading)

    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
