import argparse
import sys

from config_manager import load_config
from data_collector import DataCollector
from gps_manager import GPSManager
from scanner import Scanner
from sdr_manager import SDRManager


def main():
    parser = argparse.ArgumentParser(description="Lite SDR Data Collector")
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file")
    parser.add_argument("-d", "--duration", type=float, default=10,
                        help="Duration to collect data in minutes")
    parser.add_argument("-o", "--output", type=str, default="collected_data_lite.csv",
                        help="Output CSV filename")

    args = parser.parse_args()

    try:
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)

        print(f"Getting SDRManager from config file...")
        sdr = SDRManager(config.config)
        sdr.initialize_device()
        print("Initializing scanner and collector...")
        scanner = Scanner(sdr, config)
        print(f"Getting GPSManager from config file...")
        gps = GPSManager(scanner, config.config)
        print(f"Getting DataCollector from {config.config}...")
        collector = DataCollector(config=config, scanner=scanner, gps_manager=gps)

        print(f"Starting data collection for {args.duration} minutes...")
        collector.gather_data(duration_minutes=args.duration, filename=args.output)

        print(f"Data successfully saved to {args.output}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
