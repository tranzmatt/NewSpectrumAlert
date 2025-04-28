"""
FRS Signal Detector with MQTT Reporting
--------------------------------------
A Python application that uses SoapySDR to detect FRS radio signals and report them via MQTT.
Supports both HackRF and RTL-SDR devices.

FRS (Family Radio Service) operates in the 462 MHz and 467 MHz bands.
"""

import numpy as np
import time
import json
import argparse
import signal
import sys
import logging
import os
from datetime import datetime
from threading import Thread, Event

# SoapySDR for radio interface
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

# MQTT for reporting
import paho.mqtt.client as mqtt
import ssl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('frs_detector')

# Global variables
exit_event = Event()

class FRSDetector:
    """
    Detector for FRS signals using SoapySDR.
    """
    # FRS operates in the 462 and 467 MHz bands
    DEFAULT_FREQ = 462.6e6  # Center frequency in Hz
    DEFAULT_SAMPLE_RATE = 8e6  # Sample rate in Hz (8 MHz to cover more spectrum at once)
    DEFAULT_BANDWIDTH = 6e6  # Bandwidth in Hz (6 MHz to cover many channels at once)
    DEFAULT_GAIN = 40  # Gain in dB
    
    # Detection parameters
    DETECTION_THRESHOLD = 15.0  # Power threshold for signal detection
    DB_THRESHOLD = 15.0  # dB above noise floor for peak detection
    MIN_SNR = 10.0  # Minimum SNR to report
    
    # Signal classification parameters
    # Based on spectral shape:
    # - Narrow spikes (likely interference): < 5 kHz bandwidth
    # - FRS transmissions: typically 8-20 kHz bandwidth
    MIN_VALID_BW_KHZ = 5.0  # Minimum bandwidth for a valid signal (not a spike)
    NARROW_BW_THRESHOLD_KHZ = 5.0  # Signals below this are classified as "narrow_spike"
    MEDIUM_BW_THRESHOLD_KHZ = 20.0  # Signals below this are "frs_signal", above are "wide_signal"
    
    # Ignore persistent signals (they reappear in the same place constantly)
    PERSISTENT_SIGNAL_MEMORY = 100  # Remember this many signals
    PERSISTENT_SIGNAL_THRESHOLD = 80  # If a signal appears in >80% of scans, consider it persistent
    PERSISTENT_SIGNAL_MARGIN_KHZ = 2.0  # Frequency margin for identifying the same signal
    
    FFT_SIZE = 4096  # FFT size for spectrum analysis
    SAMPLES_PER_READ = 32768  # Number of samples per read
    
    # Frequency ranges for FRS channels
    # FRS channels span 462.5625 - 462.7250 MHz and 467.5625 - 467.7250 MHz
    MIN_FREQ = 462.5e6  # 462.5 MHz (slightly below FRS Channel 1)
    MAX_FREQ = 467.8e6  # 467.8 MHz (slightly above FRS Channel 22)
    
    # Define the frequency scan range for FRS
    # FRS channels span 462.5625 - 462.7250 MHz and 467.5625 - 467.7250 MHz
    MIN_FREQ = 462.5e6  # 462.5 MHz (slightly below FRS Channel 1)
    MAX_FREQ = 467.8e6  # 467.8 MHz (slightly above FRS Channel 22)
    
    # FRS standard channel frequencies in MHz
    FRS_CHANNELS = [
        462.5625,  # Channel 1
        462.5875,  # Channel 2
        462.6125,  # Channel 3
        462.6375,  # Channel 4
        462.6625,  # Channel 5
        462.6875,  # Channel 6
        462.7125,  # Channel 7
        467.5625,  # Channel 8
        467.5875,  # Channel 9
        467.6125,  # Channel 10
        467.6375,  # Channel 11
        467.6625,  # Channel 12
        467.6875,  # Channel 13
        467.7125,  # Channel 14
        462.5500,  # Channel 15
        462.5750,  # Channel 16
        462.6000,  # Channel 17
        462.6250,  # Channel 18
        462.6500,  # Channel 19
        462.6750,  # Channel 20
        462.7000,  # Channel 21
        462.7250   # Channel 22
    ]
    
    # Optimized frequency centers for broader coverage with fewer hops
    # Just 2 frequency points to cover the entire FRS spectrum
    OPTIMIZED_SCAN_FREQS = [
        462.625e6,  # Center frequency to cover 462 MHz band (Channels 1-7, 15-22)
        467.625e6   # Center frequency to cover 467 MHz band (Channels 8-14)
    ]
    
    # For spectrum scanning when not using optimized points
    FREQ_STEP = 5e6  # 5 MHz steps with 6 MHz bandwidth gives good overlap
    
    def __init__(self, device_args="", 
                 frequency=DEFAULT_FREQ, 
                 sample_rate=DEFAULT_SAMPLE_RATE,
                 bandwidth=DEFAULT_BANDWIDTH,
                 gain=DEFAULT_GAIN,
                 mqtt_client=None):
        """
        Initialize the FRS detector.
        
        Args:
            device_args: SoapySDR device arguments (e.g., "driver=hackrf" or "driver=rtlsdr")
            frequency: Center frequency in Hz
            sample_rate: Sample rate in Hz
            bandwidth: Bandwidth in Hz
            gain: Gain in dB
            mqtt_client: MQTT client for reporting
        """
        self.device_args = device_args
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain
        self.mqtt_client = mqtt_client
        
        self.sdr = None
        self.rx_stream = None
        self.running = False
        self.detection_thread = None
        
        # Frequency scanning parameters
        self.scan_enabled = True  # Enable frequency scanning by default
        self.freq_dwell_time = 1.0  # Time to spend at each frequency in seconds
        self.scan_by_channels = True  # Scan using predefined FRS channels by default
        
        # Persistent signal tracking
        self.persistent_signals = {}  # {freq_mhz: count}
        self.scan_count = 0  # Total number of scans performed
        
        logger.info(f"Initializing FRS detector with device: {device_args}")
        logger.info(f"Frequency: {frequency/1e6} MHz, Sample rate: {sample_rate/1e6} MHz, Bandwidth: {bandwidth/1e6} MHz")
    
    def setup_device(self):
        """Set up the SDR device."""
        try:
            # Check if device is already in use before attempting to open it
            if 'hackrf' in self.device_args.lower():
                # For HackRF, try to detect if the device is busy first
                try:
                    import subprocess
                    result = subprocess.run(['hackrf_info'], capture_output=True, text=True)
                    if result.returncode != 0 and ("Resource busy" in result.stderr or "failed to open" in result.stderr):
                        logger.error("HackRF device is busy or in use by another application")
                        logger.error("Close any applications using the device (like CubeSDR) before starting this detector")
                        return False
                except FileNotFoundError:
                    logger.warning("hackrf_info command not found, skipping busy check")
                except Exception as e:
                    logger.warning(f"Error checking if HackRF is busy: {e}, proceeding anyway")
            
            # Attempt to create the SDR device
            try:
                self.sdr = SoapySDR.Device(self.device_args)
            except Exception as e:
                error_msg = str(e).lower()
                if "resource busy" in error_msg or "failed to open" in error_msg or "busy" in error_msg:
                    logger.error(f"Device {self.device_args} is already in use by another application")
                    logger.error("Close the other application (like CubeSDR) and try again")
                    return False
                else:
                    raise
            
            # Set sample rate
            self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            
            # Set center frequency
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.frequency)
            
            # Set bandwidth
            self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
            
            # Set gain - handling differently for different SDR types
            if 'hackrf' in self.device_args.lower():
                # HackRF uses 'LNA' and 'VGA' gains
                try:
                    self.sdr.setGain(SOAPY_SDR_RX, 0, 'LNA', min(40, self.gain))
                    self.sdr.setGain(SOAPY_SDR_RX, 0, 'VGA', min(40, max(0, self.gain - 40)))
                except Exception as e:
                    logger.warning(f"Error setting individual gains, trying overall gain: {e}")
                    self.sdr.setGain(SOAPY_SDR_RX, 0, min(40, self.gain))
            elif 'rtlsdr' in self.device_args.lower():
                # RTL-SDR uses a single gain value
                self.sdr.setGain(SOAPY_SDR_RX, 0, min(47, self.gain))  # Max RTL-SDR gain is 47 dB
            else:
                # Generic approach for other SDRs
                self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)
            
            # Get actual values that were set
            try:
                actual_freq = self.sdr.getFrequency(SOAPY_SDR_RX, 0)
                actual_rate = self.sdr.getSampleRate(SOAPY_SDR_RX, 0)
                
                logger.info(f"Device setup successful")
                logger.info(f"Actual frequency: {actual_freq/1e6} MHz")
                logger.info(f"Actual sample rate: {actual_rate/1e6} MHz")
            except Exception as e:
                logger.warning(f"Could not get actual device parameters: {e}")
            
            # Setup the RX stream
            self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            
            return True
        
        except Exception as e:
            logger.error(f"Error setting up SDR device: {e}")
            return False
    
    def start(self):
        """Start the detection process."""
        if self.running:
            logger.warning("Detector is already running")
            return
        
        if not self.setup_device():
            logger.error("Failed to set up the SDR device")
            return
        
        self.running = True
        self.sdr.activateStream(self.rx_stream)
        
        # Start the detection thread
        self.detection_thread = Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        logger.info("FRS detector started")
    
    def stop(self):
        """Stop the detection process."""
        if not self.running:
            return
        
        self.running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        if self.rx_stream:
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)
        
        if self.sdr:
            # No explicit close method in SoapySDR.Device, it's handled by the destructor
            self.sdr = None
        
        logger.info("FRS detector stopped")
    
    def _detection_loop(self):
        """Main detection loop running in a separate thread."""
        buffer = np.array([0] * self.SAMPLES_PER_READ, np.complex64)
        
        # Set up frequency scanning if enabled
        if self.scan_enabled:
            # Determine scan mode
            if hasattr(self, 'use_optimized_scan') and self.use_optimized_scan:
                # Use optimized scan points for maximum coverage with minimum hops
                scan_freqs = np.array(self.OPTIMIZED_SCAN_FREQS)
                logger.info(f"Starting optimized scan with {len(scan_freqs)} frequency points to cover entire FRS spectrum")
            elif self.scan_by_channels:
                # Use predefined FRS channels
                scan_freqs = np.array([ch * 1e6 for ch in self.FRS_CHANNELS])
                logger.info(f"Starting scan across {len(scan_freqs)} FRS channels")
            else:
                # Sweep the frequency range
                scan_freqs = np.arange(self.MIN_FREQ, self.MAX_FREQ + self.FREQ_STEP, self.FREQ_STEP)
                logger.info(f"Starting frequency sweep between {self.MIN_FREQ/1e6} MHz and {self.MAX_FREQ/1e6} MHz")
            
            current_freq_idx = 0
            last_freq_change = time.time()
            
            # Set initial frequency
            if scan_freqs.size > 0:
                self.sdr.setFrequency(SOAPY_SDR_RX, 0, scan_freqs[current_freq_idx])
                if self.scan_by_channels:
                    ch_num = current_freq_idx + 1
                    logger.info(f"Scanning FRS Channel {ch_num}: {scan_freqs[current_freq_idx]/1e6} MHz")
                else:
                    logger.info(f"Scanning at frequency: {scan_freqs[current_freq_idx]/1e6} MHz")
        else:
            # Use the fixed frequency
            scan_freqs = np.array([self.frequency])
            current_freq_idx = 0
            last_freq_change = time.time()
            
            # Find which channel this corresponds to, if any
            ch_match = None
            for i, ch_freq in enumerate([ch * 1e6 for ch in self.FRS_CHANNELS]):
                if abs(self.frequency - ch_freq) < 1000:  # Within 1 kHz
                    ch_match = i + 1
                    break
            
            if ch_match:
                logger.info(f"Fixed frequency mode: FRS Channel {ch_match} ({self.frequency/1e6} MHz)")
            else:
                logger.info(f"Fixed frequency mode: {self.frequency/1e6} MHz (not a standard FRS channel)")
                
        # Store FFT analysis results for visualization if needed
        self.spectrum_data = []
        last_spectrum_update = time.time()
        
        while self.running and not exit_event.is_set():
            try:
                # Check if it's time to change frequency and scanning is enabled
                current_time = time.time()
                if self.scan_enabled and current_time - last_freq_change > self.freq_dwell_time and scan_freqs.size > 1:
                    # Move to next frequency
                    current_freq_idx = (current_freq_idx + 1) % scan_freqs.size
                    new_freq = scan_freqs[current_freq_idx]
                    
                    # Set new frequency
                    self.sdr.setFrequency(SOAPY_SDR_RX, 0, new_freq)
                    
                    # Update timestamp
                    last_freq_change = current_time
                    
                    # Log which frequency we're scanning
                    if hasattr(self, 'use_optimized_scan') and self.use_optimized_scan:
                        # For optimized scan, show which band we're covering
                        if new_freq < 465e6:
                            logger.info(f"Scanning 462 MHz band (Channels 1-7, 15-22) at {new_freq/1e6} MHz")
                        else:
                            logger.info(f"Scanning 467 MHz band (Channels 8-14) at {new_freq/1e6} MHz")
                    elif self.scan_by_channels:
                        ch_num = current_freq_idx + 1
                        logger.info(f"Scanning FRS Channel {ch_num}: {new_freq/1e6} MHz")
                    else:
                        logger.info(f"Scanning at frequency: {new_freq/1e6} MHz")
                
                # Read samples
                sr = self.sdr.readStream(self.rx_stream, [buffer], len(buffer), timeoutUs=1000000)
                if sr.ret > 0:
                    # Process the samples - get current frequency
                    try:
                        current_freq = self.sdr.getFrequency(SOAPY_SDR_RX, 0)
                    except Exception:
                        # If we can't get the frequency, use the one we think we're at
                        current_freq = scan_freqs[current_freq_idx]
                    
                    # Calculate FFT for spectral analysis
                    spectrum = np.fft.fftshift(np.fft.fft(buffer[:min(sr.ret, self.FFT_SIZE)]))
                    spectrum_db = 10 * np.log10(np.abs(spectrum)**2 + 1e-10)
                    
                    # Store spectrum data periodically for visualization
                    if current_time - last_spectrum_update > 1.0:  # Once per second
                        # Calculate frequency axis
                        freq_range = np.linspace(
                            current_freq - self.sample_rate/2,
                            current_freq + self.sample_rate/2,
                            len(spectrum_db)
                        ) / 1e6  # Convert to MHz
                        
                        # Store data for visualization or logging
                        self.spectrum_data.append({
                            'timestamp': datetime.now().isoformat(),
                            'center_freq': current_freq / 1e6,
                            'freq_range': freq_range.tolist(),
                            'spectrum': spectrum_db.tolist()
                        })
                        
                        # Limit stored data to prevent memory issues
                        if len(self.spectrum_data) > 10:
                            self.spectrum_data.pop(0)
                            
                        last_spectrum_update = current_time
                        
                    # Process the samples with the current frequency
                    self._process_samples(buffer[:sr.ret], current_freq, spectrum_db)
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1)  # Sleep a bit longer on error
    
    def _process_samples(self, samples, current_freq, spectrum_db=None):
        """
        Process the captured samples to detect FRS signals.
        
        This version distinguishes between different signal types based on bandwidth:
        - Narrow spikes (likely interference)
        - Medium bandwidth signals (likely FRS transmissions)
        - Wide bandwidth signals
        
        Also tracks persistent signals to identify likely interference sources.
        
        Args:
            samples: Complex samples from the SDR
            current_freq: Current center frequency being monitored
            spectrum_db: Pre-calculated spectrum in dB (optional)
        """
        # Calculate power
        power = np.mean(np.abs(samples)**2)
        
        # Calculate spectrum using FFT if not provided
        if spectrum_db is None:
            spectrum = np.fft.fftshift(np.fft.fft(samples[:self.FFT_SIZE]))
            spectrum_db = 10 * np.log10(np.abs(spectrum)**2 + 1e-10)
        
        # Calculate frequency axis for the spectrum
        freq_axis = np.linspace(
            current_freq - self.sample_rate/2,
            current_freq + self.sample_rate/2,
            len(spectrum_db)
        )
        
        # Update scan counter for persistent signal tracking
        self.scan_count += 1
        if self.scan_count > self.PERSISTENT_SIGNAL_MEMORY:
            # Clean up old persistent signals
            for freq, count in list(self.persistent_signals.items()):
                self.persistent_signals[freq] = count * (self.PERSISTENT_SIGNAL_MEMORY - 1) / self.PERSISTENT_SIGNAL_MEMORY
        
        # Detect active signals in the spectrum using spectral peaks
        peaks = []
        detection_count = 0
        noise_floor = np.percentile(spectrum_db, 25)  # Estimate noise floor
        
        # Find peaks in the spectrum that are above threshold
        for i in range(1, len(spectrum_db)-1):
            if (spectrum_db[i] > spectrum_db[i-1] and 
                spectrum_db[i] > spectrum_db[i+1] and 
                spectrum_db[i] > noise_floor + self.DB_THRESHOLD):  # Using configured dB threshold
                
                # Calculate SNR
                snr = spectrum_db[i] - noise_floor
                
                # Only count peaks with sufficient SNR
                if snr >= self.MIN_SNR:
                    # This is a peak - calculate its frequency
                    peak_freq = freq_axis[i]
                    peak_freq_mhz = peak_freq / 1e6
                    peak_power = spectrum_db[i]
                    
                    # Measure signal bandwidth (rough estimate)
                    # Find points 3 dB down from peak
                    left_idx = i
                    while left_idx > 0 and spectrum_db[left_idx] > peak_power - 3:
                        left_idx -= 1
                    
                    # Look right for 3dB down point
                    right_idx = i
                    while right_idx < len(spectrum_db)-1 and spectrum_db[right_idx] > peak_power - 3:
                        right_idx += 1
                    
                    # Calculate bandwidth in kHz
                    bandwidth_hz = (freq_axis[right_idx] - freq_axis[left_idx])
                    bandwidth_khz = bandwidth_hz / 1000
                    
                    # Track this frequency for persistent signal detection
                    freq_key = f"{peak_freq_mhz:.4f}"
                    for stored_freq in list(self.persistent_signals.keys()):
                        # Check if this is the same signal (within margin)
                        if abs(float(stored_freq) - peak_freq_mhz) < (self.PERSISTENT_SIGNAL_MARGIN_KHZ / 1000):
                            self.persistent_signals[stored_freq] += 1
                            freq_key = stored_freq  # Use the existing key
                            break
                    
                    # If it's a new frequency, add it to tracking
                    if freq_key not in self.persistent_signals:
                        self.persistent_signals[freq_key] = 1
                    
                    # Check if this is a persistent signal
                    is_persistent = False
                    persistence_pct = 0
                    if self.scan_count > 10:  # Need some history to determine persistence
                        persistence_pct = (self.persistent_signals[freq_key] / self.scan_count) * 100
                        is_persistent = persistence_pct > self.PERSISTENT_SIGNAL_THRESHOLD
                    
                    # Determine signal type based on bandwidth
                    if bandwidth_khz < self.NARROW_BW_THRESHOLD_KHZ:
                        signal_type = "narrow_spike"
                    elif bandwidth_khz < self.MEDIUM_BW_THRESHOLD_KHZ:
                        signal_type = "frs_signal"
                    else:
                        signal_type = "wide_signal"
                    
                    # Only accept signals with sufficient bandwidth or for debugging
                    if bandwidth_khz >= self.MIN_VALID_BW_KHZ or signal_type == "narrow_spike":
                        # Add persistence info to the signal data
                        peaks.append((peak_freq, peak_power, snr, bandwidth_khz, 
                                      signal_type, is_persistent, persistence_pct))
                        
                        # Count valid FRS signals for detection summary
                        if signal_type == "frs_signal" and not is_persistent:
                            detection_count += 1
                        
                        logger.debug(f"Peak at {peak_freq_mhz:.4f} MHz, BW: {bandwidth_khz:.1f} kHz, " +
                                    f"SNR: {snr:.1f} dB, Type: {signal_type}, " +
                                    f"Persistent: {is_persistent} ({persistence_pct:.1f}%)")
        
        # Process each detected peak
        for peak_data in peaks:
            peak_freq, peak_power, snr, bandwidth_khz, signal_type, is_persistent, persistence_pct = peak_data
            
            # Skip reporting persistent narrow spikes (likely interference)
            if is_persistent and signal_type == "narrow_spike":
                logger.debug(f"Skipping persistent narrow spike at {peak_freq/1e6:.4f} MHz")
                continue
            
            # Find which FRS channel this frequency is closest to
            peak_freq_mhz = peak_freq / 1e6
            closest_ch = None
            closest_dist = float('inf')
            
            for i, ch_freq in enumerate(self.FRS_CHANNELS):
                dist = abs(peak_freq_mhz - ch_freq)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ch = i + 1  # Channel numbers start at 1
            
            # Report the detection
            if closest_dist <= 0.1:  # Within 100 kHz of a standard channel
                self._report_detection(
                    peak_power, snr, peak_freq, closest_ch, 
                    detection_type=signal_type,
                    bandwidth_khz=bandwidth_khz, 
                    is_persistent=is_persistent,
                    persistence_pct=persistence_pct
                )
            else:
                # If not on a standard channel, still report
                self._report_detection(
                    peak_power, snr, peak_freq, None, 
                    detection_type=signal_type,
                    bandwidth_khz=bandwidth_khz,
                    is_persistent=is_persistent,
                    persistence_pct=persistence_pct
                )
        
        # Log a summary if FRS signals were detected
        if detection_count > 0:
            logger.info(f"Detected {detection_count} potential FRS signals at {current_freq/1e6:.3f} MHz")
        
        # Fallback to basic power detection if no peaks were found but power is high
        if detection_count == 0 and power > self.DETECTION_THRESHOLD:
            # Calculate an approximate SNR
            approx_snr = 10 * np.log10(power / 0.1)  # Assuming 0.1 is a typical noise level
            
            # Only report if SNR is sufficient
            if approx_snr >= self.MIN_SNR:
                # Broad spectrum activity detected
                self._report_detection(power, approx_snr, current_freq, None, detection_type="broad_spectrum")
    
    def _report_detection(self, power, snr, freq_hz, channel=None, detection_type="signal", 
                         bandwidth_khz=None, is_persistent=False, persistence_pct=0):
        """
        Report signal detection via MQTT.
        
        Args:
            power: Signal power
            snr: Estimated signal-to-noise ratio
            freq_hz: Frequency in Hz
            channel: FRS channel number (1-22) if identifiable
            detection_type: Type of detection ('frs_signal', 'narrow_spike', 'wide_signal', 'broad_spectrum')
            bandwidth_khz: Estimated signal bandwidth in kHz (if available)
            is_persistent: Whether this is a persistent signal (likely interference)
            persistence_pct: Percentage of scans this signal has been present in
        """
        # Formatted frequency in MHz
        freq_mhz = freq_hz / 1e6
        
        # Format channel information
        if channel:
            channel_info = f"Channel {channel}"
        else:
            channel_info = "Unknown"
        
        # Format bandwidth information
        bw_info = f", BW: {bandwidth_khz:.1f} kHz" if bandwidth_khz else ""
        
        # Format persistence information
        persist_info = f", Persistent: {persistence_pct:.1f}%" if is_persistent else ""
        
        # Create a human-readable signal type description
        if detection_type == "narrow_spike":
            type_desc = "Narrow spike (likely interference)"
        elif detection_type == "frs_signal":
            type_desc = "FRS signal"
        elif detection_type == "wide_signal":
            type_desc = "Wide bandwidth signal"
        else:
            type_desc = detection_type
            
        # Skip logging for persistent narrow spikes after a while
        if is_persistent and detection_type == "narrow_spike" and self.scan_count > 20:
            return
        
        # Log detection based on type
        if not self.mqtt_client:
            if detection_type == "broad_spectrum":
                logger.info(f"FRS activity detected! Broad spectrum at {freq_mhz:.4f} MHz, Power: {power:.2f} dB{bw_info}")
            else:
                logger.info(f"{type_desc} detected! {channel_info} at {freq_mhz:.4f} MHz, " +
                          f"Power: {power:.2f}, SNR: {snr:.2f} dB{bw_info}{persist_info}")
            return
        
        # Prepare detection report
        timestamp = datetime.now().isoformat()
        report = {
            "type": "FRS",
            "timestamp": timestamp,
            "frequency_mhz": float(freq_mhz),
            "power": float(power),
            "snr": float(snr),
            "device": self.device_args,
            "detection_type": detection_type,
            "signal_desc": type_desc
        }
        
        # Add channel information if available
        if channel:
            report["channel"] = int(channel)
        
        # Add bandwidth information if available
        if bandwidth_khz:
            report["bandwidth_khz"] = float(bandwidth_khz)
        
        # Add persistence information if relevant
        if is_persistent:
            report["is_persistent"] = is_persistent
            report["persistence_pct"] = float(persistence_pct)
        
        # Add coverage information - useful for spectrum monitoring
        # Calculate the approximate frequency coverage
        bandwidth = self.bandwidth / 1e6  # Convert to MHz
        report["min_freq_mhz"] = freq_mhz - (bandwidth / 2)
        report["max_freq_mhz"] = freq_mhz + (bandwidth / 2)
        
        # Get the MQTT topic from the handler
        topic = getattr(self.mqtt_client._userdata, 'topic', 'sensors/rf/frs/detection')
        
        # Publish to MQTT
        try:
            self.mqtt_client.publish(topic, json.dumps(report))
            
            # Only log important detections to avoid flooding
            if detection_type == "frs_signal" or (not is_persistent and detection_type != "narrow_spike"):
                logger.info(f"Detection reported to MQTT topic {topic}: {report}")
            else:
                logger.debug(f"Detection reported to MQTT topic {topic}: {report}")
        except Exception as e:
            logger.error(f"Error publishing to MQTT: {e}")


class MQTTHandler:
    """
    Handles MQTT connection and messaging.
    """
    def __init__(self, broker=None, port=None, client_id="frs_detector", 
                 username=None, password=None, use_tls=False, ca_cert=None,
                 topic=None):
        """
        Initialize the MQTT handler.
        
        Args:
            broker: MQTT broker address
            port: MQTT broker port
            client_id: Client ID for MQTT connection
            username: MQTT username for authentication
            password: MQTT password for authentication
            use_tls: Whether to use TLS for connection
            ca_cert: CA certificate file path for TLS
            topic: MQTT topic for publishing detection reports
        """
        # Get configuration from environment variables if not provided
        self.broker = broker or os.environ.get('MQTT_HOST', 'localhost')
        
        # Convert port to int, with fallbacks
        if port is not None:
            self.port = port
        else:
            port_str = os.environ.get('MQTT_PORT')
            self.port = int(port_str) if port_str else 1883
            
        self.client_id = client_id
        
        # Authentication credentials from args or environment
        self.username = username or os.environ.get('MQTT_USER')
        self.password = password or os.environ.get('MQTT_PASSWORD')
        
        # TLS settings from args or environment
        self.use_tls = use_tls or (os.environ.get('MQTT_TLS', '').lower() in ('true', 't', 'yes', 'y', '1'))
        self.ca_cert = ca_cert or os.environ.get('MQTT_CA_CERT')
        
        # Topic for publishing
        self.topic = topic or os.environ.get('MQTT_TOPIC', 'sensors/rf/frs/detection')
        
        # Create client
        self.client = mqtt.Client(client_id=client_id)
        
        # Store topic in userdata for access during publishing
        self.client.user_data_set(self)
        
        # Set up authentication if provided
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        
        logger.info(f"MQTT client initialized: {self.broker}:{self.port}")
        if self.username:
            logger.info(f"Using MQTT authentication with username: {self.username}")
        if self.use_tls:
            logger.info(f"MQTT TLS enabled" + (f" with CA cert: {self.ca_cert}" if self.ca_cert else ""))
    
    def connect(self):
        """Connect to the MQTT broker."""
        try:
            # Configure TLS if enabled
            if self.use_tls:
                self.client.tls_set(
                    ca_certs=self.ca_cert,
                    cert_reqs=ssl.CERT_REQUIRED if self.ca_cert else ssl.CERT_NONE,
                    tls_version=ssl.PROTOCOL_TLS
                )
            
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the broker."""
        if rc == 0:
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"Failed to connect to MQTT broker, return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects from the broker."""
        if rc != 0:
            logger.warning(f"Unexpected disconnection from MQTT broker: {rc}")


def list_devices():
    """List available SDR devices."""
    # Try SoapySDR device enumeration first
    results = SoapySDR.Device.enumerate()
    
    print("Checking for SDR devices...")
    
    # If no devices found through SoapySDR, try to check for HackRF directly
    if not results:
        print("No devices found through SoapySDR API.")
        
        # Try to check if HackRF is available through hackrf_info (requires subprocess)
        try:
            import subprocess
            print("\nAttempting to detect HackRF devices directly...")
            result = subprocess.run(['hackrf_info'], capture_output=True, text=True)
            
            if result.returncode == 0 and "Found HackRF" in result.stdout:
                # Extract serial from hackrf_info output
                serial = None
                for line in result.stdout.splitlines():
                    if "Serial number:" in line:
                        serial = line.split("Serial number:")[1].strip()
                
                print(f"Found HackRF device with serial number: {serial}")
                print("Use '--device \"driver=hackrf\"' to select this device.")
                if serial:
                    print(f"You can also specify '--serial \"{serial}\"' to ensure this specific device is selected.")
                return
            elif result.returncode != 0 and "Resource busy" in result.stderr:
                print("HackRF device detected but is currently in use by another application.")
                print("Close the other application and try again.")
                return
        except FileNotFoundError:
            print("hackrf_info command not found. Make sure HackRF tools are installed.")
        except Exception as e:
            print(f"Error checking for HackRF: {e}")
    
        # Try to check if RTL-SDR is available (requires subprocess)
        try:
            import subprocess
            print("\nAttempting to detect RTL-SDR devices directly...")
            result = subprocess.run(['rtl_test'], capture_output=True, text=True)
            
            if "Found" in result.stdout:
                print("RTL-SDR device detected.")
                print("Use '--device \"driver=rtlsdr\"' to select this device.")
                return
        except FileNotFoundError:
            print("rtl_test command not found. Make sure RTL-SDR tools are installed.")
        except Exception as e:
            print(f"Error checking for RTL-SDR: {e}")
            
        print("\nNo SDR devices were found. Please ensure your device is properly connected.")
        print("You may need to install the appropriate drivers and libraries for your SDR device.")
        return
    
    # Report devices found through SoapySDR
    print(f"Found {len(results)} SDR device(s) through SoapySDR:")
    for i, result in enumerate(results):
        print(f"Device {i+1}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect FRS radio signals using SoapySDR and report via MQTT."
    )
    
    parser.add_argument("--list-devices", action="store_true",
                        help="List available SDR devices and exit")
    
    parser.add_argument("--device", default="",
                        help="SoapySDR device driver (e.g., 'driver=hackrf' or 'driver=rtlsdr')")
    
    parser.add_argument("--serial", default=None,
                        help="Device serial number for specific device selection")
    
    # Frequency scanning parameters
    freq_group = parser.add_argument_group('Frequency Parameters')
    
    freq_group.add_argument("--freq", type=float, default=FRSDetector.DEFAULT_FREQ/1e6,
                        help=f"Center frequency in MHz (default: {FRSDetector.DEFAULT_FREQ/1e6})")
    
    freq_group.add_argument("--min-freq", type=float, default=FRSDetector.MIN_FREQ/1e6,
                        help=f"Minimum scan frequency in MHz (default: {FRSDetector.MIN_FREQ/1e6})")
    
    freq_group.add_argument("--max-freq", type=float, default=FRSDetector.MAX_FREQ/1e6,
                        help=f"Maximum scan frequency in MHz (default: {FRSDetector.MAX_FREQ/1e6})")
    
    freq_group.add_argument("--freq-step", type=float, default=FRSDetector.FREQ_STEP/1e6,
                        help=f"Frequency step size for scanning in MHz (default: {FRSDetector.FREQ_STEP/1e6})")
    
    freq_group.add_argument("--disable-scan", action="store_true",
                        help="Disable frequency scanning (stay on center frequency)")
    
    freq_group.add_argument("--scan-by-channels", action="store_true",
                        help="Scan only standard FRS channels instead of sweeping (default: True)")
    
    freq_group.add_argument("--sweep-mode", action="store_true",
                        help="Scan by sweeping frequency range instead of jumping between channels")
    
    freq_group.add_argument("--dwell-time", type=float, default=1.0,
                        help="Time to spend at each frequency in seconds (default: 1.0)")
    
    # SDR parameters
    sdr_group = parser.add_argument_group('SDR Parameters')
    
    sdr_group.add_argument("--sample-rate", type=float, default=FRSDetector.DEFAULT_SAMPLE_RATE/1e6,
                        help=f"Sample rate in MHz (default: {FRSDetector.DEFAULT_SAMPLE_RATE/1e6})")
    
    sdr_group.add_argument("--bandwidth", type=float, default=FRSDetector.DEFAULT_BANDWIDTH/1e6,
                        help=f"Bandwidth in MHz (default: {FRSDetector.DEFAULT_BANDWIDTH/1e6})")
    
    sdr_group.add_argument("--gain", type=float, default=FRSDetector.DEFAULT_GAIN,
                        help=f"Gain in dB (default: {FRSDetector.DEFAULT_GAIN})")
    
    # Detection parameters
    detect_group = parser.add_argument_group('Detection Parameters')
    
    detect_group.add_argument("--threshold", type=float, default=FRSDetector.DETECTION_THRESHOLD,
                        help=f"Detection threshold for power level (default: {FRSDetector.DETECTION_THRESHOLD})")
    
    detect_group.add_argument("--db-threshold", type=float, default=FRSDetector.DB_THRESHOLD,
                        help=f"dB threshold above noise floor for peak detection (default: {FRSDetector.DB_THRESHOLD})")
    
    detect_group.add_argument("--min-snr", type=float, default=FRSDetector.MIN_SNR,
                        help=f"Minimum SNR in dB to report a detection (default: {FRSDetector.MIN_SNR})")
    
    # MQTT connection settings
    mqtt_group = parser.add_argument_group('MQTT Settings')
    
    mqtt_group.add_argument("--mqtt-broker", default=None,
                        help="MQTT broker address (default: from MQTT_HOST env var or 'localhost')")
    
    mqtt_group.add_argument("--mqtt-port", type=int, default=None,
                        help="MQTT broker port (default: from MQTT_PORT env var or 1883)")
    
    mqtt_group.add_argument("--mqtt-topic", default=None,
                        help="MQTT topic for detection reports (default: from MQTT_TOPIC env var or 'sensors/rf/frs/detection')")
    
    mqtt_group.add_argument("--mqtt-user", default=None,
                        help="MQTT username (default: from MQTT_USER env var)")
    
    mqtt_group.add_argument("--mqtt-password", default=None,
                        help="MQTT password (default: from MQTT_PASSWORD env var)")
    
    mqtt_group.add_argument("--mqtt-tls", action="store_true",
                        help="Use TLS for MQTT connection (default: from MQTT_TLS env var)")
    
    mqtt_group.add_argument("--mqtt-ca-cert", default=None,
                        help="CA certificate file for TLS (default: from MQTT_CA_CERT env var)")
    
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Logging level (default: INFO)")
    
    return parser.parse_args()


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    logger.info("Interrupt received, shutting down...")
    exit_event.set()
    sys.exit(0)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    # List devices if requested
    if args.list_devices:
        list_devices()
        return
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Build device args string
    device_args = args.device
    if args.serial:
        # Add serial number to the device arguments
        if device_args:
            device_args += f",serial={args.serial}"
        else:
            device_args = f"serial={args.serial}"
        logger.info(f"Using device with serial number: {args.serial}")
    
    # The device string must include a driver
    if not device_args:
        logger.error("No device specified. Use --device 'driver=hackrf' or --device 'driver=rtlsdr'")
        return
    
    if "driver=" not in device_args:
        logger.error("Device specification must include a driver (e.g., 'driver=hackrf')")
        return
    
    # Check if the HackRF device is busy before proceeding
    if 'hackrf' in device_args.lower():
        try:
            import subprocess
            result = subprocess.run(['hackrf_info'], capture_output=True, text=True)
            if result.returncode != 0 and ("Resource busy" in result.stderr or "failed to open" in result.stderr):
                logger.error("HackRF device is already in use by another application (likely CubeSDR)")
                logger.error("Close the other application before starting this detector")
                return
        except FileNotFoundError:
            logger.warning("hackrf_info command not found, skipping busy check")
        except Exception as e:
            logger.warning(f"Error checking if HackRF is busy: {e}, proceeding anyway")
    
    # Set up MQTT
    mqtt_handler = MQTTHandler(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        client_id=f"frs_detector_{int(time.time())}",
        username=args.mqtt_user,
        password=args.mqtt_password,
        use_tls=args.mqtt_tls,
        ca_cert=args.mqtt_ca_cert,
        topic=args.mqtt_topic
    )
    
    if not mqtt_handler.connect():
        logger.error("Failed to connect to MQTT broker, exiting.")
        return
    
    # Create and start the detector with frequency scan parameters
    detector = FRSDetector(
        device_args=device_args,
        frequency=args.freq * 1e6,
        sample_rate=args.sample_rate * 1e6,
        bandwidth=args.bandwidth * 1e6,
        gain=args.gain,
        mqtt_client=mqtt_handler.client
    )
    
    # Set frequency scan parameters if provided
    if hasattr(args, 'min_freq'):
        detector.MIN_FREQ = args.min_freq * 1e6
    if hasattr(args, 'max_freq'):
        detector.MAX_FREQ = args.max_freq * 1e6
    if hasattr(args, 'freq_step'):
        detector.FREQ_STEP = args.freq_step * 1e6
    
    # Configure scan mode
    if hasattr(args, 'disable_scan') and args.disable_scan:
        detector.scan_enabled = False
        logger.info(f"Frequency scanning disabled. Fixed frequency: {args.freq} MHz")
    else:
        detector.scan_enabled = True
        
        # Determine scan mode
        if hasattr(args, 'sweep_mode') and args.sweep_mode:
            detector.scan_by_channels = False
            detector.use_optimized_scan = False
            logger.info(f"Frequency sweep mode enabled: {detector.MIN_FREQ/1e6} MHz to {detector.MAX_FREQ/1e6} MHz " +
                      f"in {detector.FREQ_STEP/1e6} MHz steps")
        elif hasattr(args, 'scan_by_channels') and args.scan_by_channels:
            detector.scan_by_channels = True
            detector.use_optimized_scan = False
            logger.info(f"Channel scanning mode enabled: scanning {len(detector.FRS_CHANNELS)} FRS channels")
        else:
            # Default to optimized scan mode
            detector.scan_by_channels = False
            detector.use_optimized_scan = True
            logger.info(f"Optimized scan mode enabled: covering entire FRS spectrum with {len(detector.OPTIMIZED_SCAN_FREQS)} frequency points")
    
    # Set dwell time if provided
    if hasattr(args, 'dwell_time'):
        detector.freq_dwell_time = args.dwell_time
        logger.info(f"Dwell time at each frequency: {args.dwell_time} seconds")
    
    # Set dwell time if provided
    if hasattr(args, 'dwell_time'):
        detector.freq_dwell_time = args.dwell_time
        logger.info(f"Dwell time at each frequency: {args.dwell_time} seconds")
    
    # Update detection parameters
    if hasattr(args, 'threshold'):
        detector.DETECTION_THRESHOLD = args.threshold
        logger.info(f"Power detection threshold set to: {args.threshold}")
    
    if hasattr(args, 'db_threshold'):
        detector.DB_THRESHOLD = args.db_threshold
        logger.info(f"dB threshold above noise floor set to: {args.db_threshold} dB")
    
    if hasattr(args, 'min_snr'):
        detector.MIN_SNR = args.min_snr
        logger.info(f"Minimum SNR threshold set to: {args.min_snr} dB")
    
    # Start the detector
    detector.start()
    
    try:
        # Keep running until interrupted
        while not exit_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Clean up
        detector.stop()
        mqtt_handler.disconnect()
        logger.info("Clean shutdown complete.")


if __name__ == "__main__":
    main()