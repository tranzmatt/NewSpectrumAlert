#!/usr/bin/env python3
"""
Refactored unified_data.py script that uses the modular classes from other files.
This script collects data from SDR and saves it to a CSV file.
With fixes to ensure it respects the duration parameter.
"""

import argparse
import signal
import sys
import time

from config_manager import load_config
from scanner import Scanner
from sdr_manager import SDRManager


# Signal handler for graceful termination
def signal_handler(sig, frame):
    print("\nReceived interrupt signal. Shutting down gracefully...")
    sys.exit(0)


def gather_data(config, filename, duration_minutes, use_threading=True):
    """
    Gather RF data for a specified duration and save to a CSV file.

    Parameters:
    config (ConfigManager): Configuration manager instance
    filename (str): Output CSV filename
    duration_minutes (float): Duration to gather data in minutes
    use_threading (bool): Whether to use multi-threading for scanning
    """
    sdr_manager = None

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

        # Log start time for tracking
        start_time = time.time()

        # Start the timed scan
        print(f"Starting data collection for {duration_minutes} minutes...")
        scanner.timed_scan(filename, duration_minutes, use_threading=use_threading)

        # Calculate actual elapsed time
        elapsed_time = (time.time() - start_time) / 60  # in minutes
        print(f"Data collection completed after {elapsed_time:.2f} minutes. Data saved to {filename}")

        return True

    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
        return False
    except Exception as e:
        print(f"Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure SDR resources are properly released
        if sdr_manager:
            try:
                sdr_manager.close()
                print("SDR resources released.")
            except Exception as e:
                print(f"Error closing SDR resources: {e}")


def main():
    """Main function to parse arguments and start data collection."""
    # Register signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="RF Data Collector using SDR")
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file")
    parser.add_argument("-d", "--duration", type=float, default=10,
                        help="Duration to collect data in minutes")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV filename")
    parser.add_argument("--lite", action="store_true",
                        help="Run in lite mode for resource-constrained environments")
    parser.add_argument("--no-threads", action="store_true",
                        help="Disable multi-threading even in full mode")

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)

        # Override lite mode if specified in command line
        if args.lite:
            config.set_lite_mode(True)
            print("Command line override: Running in lite mode for resource-constrained environments")

        # Determine whether to use threading
        use_threading = not config.is_lite_mode() and not args.no_threads
        if config.is_lite_mode() or args.no_threads:
            threading_status = "disabled"
            if config.is_lite_mode():
                threading_status += " (lite mode)"
            if args.no_threads:
                threading_status += " (command line override)"
            print(f"Threading {threading_status}")
        else:
            print("Using multi-threading for band scanning")

        # Determine output filename
        if args.output:
            filename = args.output
        else:
            filename = 'collected_data_lite.csv' if config.is_lite_mode() else 'collected_iq_data.csv'

        # Get duration from command line
        duration = args.duration
        if duration <= 0:
            print("Error: Duration must be greater than 0.")
            return 1

        # Print configuration summary
        print(f"Starting {'lite ' if config.is_lite_mode() else ''}IQ data collection for {duration} minutes...")
        print(f"SDR Type: {config.get_sdr_type()}")
        print(f"Sample rate: {config.get_sample_rate() / 1e6:.3f} MHz")
        print(f"Runs per frequency: {config.get_runs_per_freq()}")

        # Format and print HAM bands for better readability
        ham_bands = config.get_ham_bands()
        for i, (start, end) in enumerate(ham_bands):
            print(f"Band {i + 1}: {start / 1e6:.3f} MHz - {end / 1e6:.3f} MHz")

        print(f"Output file: {filename}")

        # Start data collection
        success = gather_data(config, filename, duration, use_threading=use_threading)
        return 0 if success else 1

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
