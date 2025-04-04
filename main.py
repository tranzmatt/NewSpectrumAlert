#!/usr/bin/env python3
import os

from spectrum_alert import create_spectrum_alert

ASCII_LOGO = """
▗▄▄▖▗▄▄▖  ▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖▗▄▄▖ ▗▖ ▗▖▗▖  ▗▖     ▗▄▖ ▗▖   ▗▄▄▄▖▗▄▄▖▗▄▄▄▖
▐▌   ▐▌ ▐▌▐▌   ▐▌     █  ▐▌ ▐▌▐▌ ▐▌▐▛▚▞▜▌    ▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌ █  
 ▝▀▚▖▐▛▀▘ ▐▛▀▘▐▌     █  ▐▛▀▚▖▐▌ ▐▌▐▌  ▐▌    ▐▛▀▜▌▐▌   ▐▛▀▘▐▛▀▚▖ █  
▗▄▄▞▘▐▌   ▐▙▄▄▖▝▚▄▄▖  █  ▐▌ ▐▌▝▚▄▞▘▐▌  ▐▌    ▐▌ ▐▌▐▙▄▄▖▐▙▄▄▖▐▌ ▐▌ █                                                            
                                                                  
"""

# File paths for lite and normal versions
DATA_FILE = "collected_data_lite.csv"
NORMAL_DATA_FILE = "collected_iq_data.csv"
MODEL_FILE = "rf_fingerprinting_model_lite.pkl"
NORMAL_MODEL_FILE = "rf_fingerprinting_model.pkl"
ANOMALY_MODEL_FILE = "anomaly_detection_model_lite.pkl"
NORMAL_ANOMALY_MODEL_FILE = "anomaly_detection_model.pkl"


def start_from_scratch():
    """Delete existing datasets and models."""
    # Delete Lite files
    for file in [DATA_FILE, MODEL_FILE, ANOMALY_MODEL_FILE]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")

    # Delete Normal files
    for file in [NORMAL_DATA_FILE, NORMAL_MODEL_FILE, NORMAL_ANOMALY_MODEL_FILE]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")

    print("Starting from scratch. All datasets and models have been deleted.")


def automate_process(duration, lite_mode=False):
    """Automate data gathering, model training, and monitoring."""
    # Initialize SpectrumAlert
    spectrum_alert = create_spectrum_alert(lite_mode=lite_mode)

    # Check for lite dataset and models
    if os.path.exists(DATA_FILE if lite_mode else NORMAL_DATA_FILE):
        print(f"Dataset found: {DATA_FILE if lite_mode else NORMAL_DATA_FILE}. Training the model...")
        spectrum_alert.train_models(DATA_FILE if lite_mode else NORMAL_DATA_FILE)
    elif os.path.exists(MODEL_FILE if lite_mode else NORMAL_MODEL_FILE) and \
            os.path.exists(ANOMALY_MODEL_FILE if lite_mode else NORMAL_ANOMALY_MODEL_FILE):
        print(f"Models found. Starting monitor...")
        spectrum_alert.monitor()
    else:
        print("No dataset or models found. Starting data gathering...")
        # Use the duration already provided
        if duration is None:
            duration = float(input("Enter the duration for data gathering (in minutes): "))

        spectrum_alert.gather_data(duration)

        print("Data gathering completed. Training the model...")
        spectrum_alert.train_models()

        print("Model training completed. Starting monitor...")
        spectrum_alert.monitor()


def main():
    while True:
        print(ASCII_LOGO)
        print("Welcome to Spectrum Alert")
        print("Please choose an option:")
        print("1. Gather Data")
        print("2. Train Model")
        print("3. Monitor Spectrum")
        print("4. Automate: Gather Data -> Train Model -> Monitor Spectrum")
        print("5. Automate: Train model or run monitor depending on existing data/models")
        print("6. Start from scratch (delete datasets and models)")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")

        if choice == "1":
            version_choice = input("Do you want to run the lite version for Raspberry Pi? (y/n): ").lower()
            lite_mode = version_choice == 'y'

            duration = float(input("Enter the duration for data gathering (in minutes): "))
            spectrum_alert = create_spectrum_alert(lite_mode=lite_mode)
            spectrum_alert.gather_data(duration)

        elif choice == "2":
            version_choice = input("Do you want to run the lite version for Raspberry Pi? (y/n): ").lower()
            lite_mode = version_choice == 'y'

            spectrum_alert = create_spectrum_alert(lite_mode=lite_mode)
            spectrum_alert.train_models()

        elif choice == "3":
            version_choice = input("Do you want to run the lite version for Raspberry Pi? (y/n): ").lower()
            lite_mode = version_choice == 'y'

            spectrum_alert = create_spectrum_alert(lite_mode=lite_mode)
            spectrum_alert.monitor()

        elif choice == "4":
            version_choice = input("Do you want to run the lite version for Raspberry Pi? (y/n): ").lower()
            lite_mode = version_choice == 'y'

            # Get the duration before automating the process
            duration = float(input("Enter the duration for data gathering (in minutes): "))
            print("Automating process: Gather Data -> Train Model -> Monitor Spectrum")

            spectrum_alert = create_spectrum_alert(lite_mode=lite_mode)
            spectrum_alert.gather_data(duration)
            spectrum_alert.train_models()
            spectrum_alert.monitor()

        elif choice == "5":
            version_choice = input("Do you want to run the lite version for Raspberry Pi? (y/n): ").lower()
            lite_mode = version_choice == 'y'

            print("Checking for existing dataset or model...")
            automate_process(None, lite_mode)

        elif choice == "6":
            print("Starting from scratch...")
            start_from_scratch()

        elif choice == "7":
            print("Exiting... Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.\n")


if __name__ == "__main__":
    main()
