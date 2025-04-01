import importlib
import sys
import os

class SDRManager:
    """
    A class to manage SDR device initialization and configuration.
    Uses SoapySDR for all device types with support for device selection.
    """
    
    def __init__(self, config):
        """
        Initialize the SDR manager.
        
        Parameters:
        config (ConfigParser or dict): Configuration object containing SDR settings
        """
        # Extract settings from config
        if hasattr(config, 'get'):
            # It's a ConfigParser object
            self.sdr_type = config.get('GENERAL', 'sdr_type', fallback='rtlsdr')
            self.sample_rate = float(config.get('GENERAL', 'sample_rate', fallback='2.048e6'))
            self.gain = config.get('GENERAL', 'gain_value', fallback='auto')
            self.device_serial = config.get('GENERAL', 'device_serial', fallback=None)
            device_index = config.get('GENERAL', 'device_index', fallback=None)
        else:
            # It's a dictionary
            self.sdr_type = config.get('sdr_type', 'rtlsdr')
            self.sample_rate = float(config.get('sample_rate', 2.048e6))
            self.gain = config.get('gain_value', 'auto')
            self.device_serial = config.get('device_serial', None)
            device_index = config.get('device_index', None)
        
        # Convert device_index to int if specified
        if device_index is not None:
            try:
                self.device_index = int(device_index)
            except ValueError:
                self.device_index = None
        else:
            self.device_index = None
            
        # Store the full config for future reference
        self.config = config
        self.sdr = None
        self.soapy_available = False
        self.rtlsdr_available = False
        
        # Try to import SoapySDR
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
            self.SoapySDR = SoapySDR
            self.SOAPY_SDR_RX = SOAPY_SDR_RX
            self.SOAPY_SDR_CF32 = SOAPY_SDR_CF32
            self.soapy_available = True
        except ImportError:
            self.SoapySDR = None
            self.soapy_available = False
            print("SoapySDR Python bindings not found. Limited functionality available.")
        
        # Try to import rtlsdr (as fallback for rtlsdr devices)
        try:
            from rtlsdr import RtlSdr
            self.RtlSdr = RtlSdr
            self.rtlsdr_available = True
        except ImportError:
            self.RtlSdr = None
            self.rtlsdr_available = False
            print("rtlsdr Python bindings not found. RTL-SDR devices may not be usable.")
        
    def initialize_device(self):
        """
        Initialize the SDR device based on the specified type.
        
        Returns:
        object: Initialized SDR device
        
        Raises:
        ImportError: If the required SDR library is not installed
        ValueError: If the specified SDR type or device cannot be found
        """
        try:
            # Prefer SoapySDR for all device types
            if self.soapy_available:
                return self._initialize_with_soapy()
            # Fall back to rtlsdr direct API if needed and available
            elif self.sdr_type == 'rtlsdr' and self.rtlsdr_available:
                return self._initialize_with_rtlsdr()
            else:
                raise ImportError(f"No suitable SDR API available for {self.sdr_type}. Please install SoapySDR.")
            
        except Exception as e:
            print(f"Error initializing SDR device: {e}")
            raise
    
    def _initialize_with_soapy(self):
        """
        Initialize the device using SoapySDR.
        
        Returns:
        object: Initialized SDR device
        """
        # Get list of available devices
        devices = self.SoapySDR.Device.enumerate()
        if not devices:
            raise ValueError("No SoapySDR devices found")
        
        # Create device arguments dictionary
        device_args = {'driver': self.sdr_type}
        
        # Try to find the device by serial number if provided (highest priority)
        if self.device_serial:
            device_args['serial'] = self.device_serial
            try:
                # Try to create the device with the specified serial
                self.sdr = self.SoapySDR.Device(device_args)
                print(f"Found {self.sdr_type} device with serial {self.device_serial}")
                
                # Configure the device
                self._configure_soapy_device()
                return self.sdr
            except Exception as e:
                print(f"Failed to initialize device with serial {self.device_serial}: {e}")
                # If serial number was specified but not found, raise an error
                if self.sdr_type == 'rtlsdr':  # RTL-SDR requires serial number to be correct
                    raise ValueError(f"Could not find RTL-SDR device with serial {self.device_serial}")
                # For other types, continue to try with device index
        
        # If no serial provided or serial not found for non-RTL-SDR devices, try device index
        if self.device_index is not None:
            try:
                # For non-RTL-SDR devices, we can try to use device index
                if len(devices) > self.device_index:
                    # Get the specific device info
                    device_info = devices[self.device_index]
                    
                    # Check if this device matches our requested type
                    if 'driver' in device_info and device_info['driver'] == self.sdr_type:
                        # Use the device's serial if available
                        if 'serial' in device_info:
                            device_args['serial'] = device_info['serial']
                        
                        # Create the device
                        self.sdr = self.SoapySDR.Device(device_args)
                        print(f"Found {self.sdr_type} device at index {self.device_index}")
                        
                        # Configure the device
                        self._configure_soapy_device()
                        return self.sdr
                    else:
                        print(f"Device at index {self.device_index} is not a {self.sdr_type}")
                else:
                    print(f"Device index {self.device_index} is out of range")
            except Exception as e:
                print(f"Failed to initialize device at index {self.device_index}: {e}")
        
        # If no specific device was found with serial or index, use the first device of the requested type
        try:
            for device_info in devices:
                if 'driver' in device_info and device_info['driver'].lower() == self.sdr_type.lower():
                    # Create device with just the driver specified (will use the first available)
                    self.sdr = self.SoapySDR.Device({'driver': self.sdr_type})
                    
                    # Log which device was selected
                    device_serial = device_info.get('serial', 'unknown')
                    print(f"Using the first available {self.sdr_type} device (serial: {device_serial})")
                    
                    # Configure the device
                    self._configure_soapy_device()
                    return self.sdr
            
            # If we got here, no device of the requested type was found
            raise ValueError(f"No {self.sdr_type} devices found")
            
        except Exception as e:
            print(f"Failed to initialize default device: {e}")
            raise
    
    def _initialize_with_rtlsdr(self):
        """
        Initialize the device using rtlsdr API directly (fallback).
        
        Returns:
        object: Initialized RTL-SDR device
        """
        try:
            if self.device_serial:
                # Find the device by serial
                from rtlsdr import RtlSdr, librtlsdr
                
                # Get number of devices
                device_count = librtlsdr.rtlsdr_get_device_count()
                
                # Try to find device with matching serial
                for i in range(device_count):
                    serial = librtlsdr.rtlsdr_get_device_usb_strings(i)[2].decode('utf-8')
                    if serial == self.device_serial:
                        self.sdr = self.RtlSdr(i)
                        print(f"Found RTL-SDR device with serial {self.device_serial} at index {i}")
                        break
                else:
                    # No matching device found
                    raise ValueError(f"Could not find RTL-SDR device with serial {self.device_serial}")
            elif self.device_index is not None:
                # Use the specified device index
                self.sdr = self.RtlSdr(self.device_index)
                print(f"Using RTL-SDR device at index {self.device_index}")
            else:
                # Use the first available device
                self.sdr = self.RtlSdr()
                print("Using the first available RTL-SDR device")
            
            # Configure the device
            self._configure_rtlsdr_device()
            return self.sdr
            
        except Exception as e:
            print(f"Failed to initialize RTL-SDR device: {e}")
            raise
    
    def _configure_soapy_device(self):
        """
        Configure a SoapySDR device with the specified parameters.
        """
        if self.sdr is None:
            raise ValueError("SoapySDR device not initialized")
        
        # Set sample rate
        self.sdr.setSampleRate(self.SOAPY_SDR_RX, 0, self.sample_rate)
        
        # Set frequency correction (not all devices support this)
        try:
            self.sdr.setFrequencyCorrection(self.SOAPY_SDR_RX, 0, 0.0)
        except Exception:
            pass  # Ignore if not supported
        
        # Set gain mode and gain
        if self.gain == 'auto':
            # Try to enable automatic gain control if supported
            try:
                self.sdr.setGainMode(self.SOAPY_SDR_RX, 0, True)
            except Exception:
                print("Automatic gain control not supported, using manual gain instead")
                try:
                    # Try to set a reasonable default gain
                    gains = self.sdr.getGainRange(self.SOAPY_SDR_RX, 0)
                    mid_gain = (gains.minimum() + gains.maximum()) / 2
                    self.sdr.setGain(self.SOAPY_SDR_RX, 0, mid_gain)
                except Exception as e:
                    print(f"Error setting default gain: {e}")
        else:
            # Disable automatic gain control
            try:
                self.sdr.setGainMode(self.SOAPY_SDR_RX, 0, False)
            except Exception:
                pass  # Ignore if not supported
            
            # Set manual gain
            try:
                gain_value = float(self.gain)
                self.sdr.setGain(self.SOAPY_SDR_RX, 0, gain_value)
            except Exception as e:
                print(f"Error setting gain to {self.gain}: {e}")
                # Try setting individual gain elements if available
                try:
                    gain_elements = self.sdr.listGains(self.SOAPY_SDR_RX, 0)
                    for elem in gain_elements:
                        try:
                            gain_range = self.sdr.getGainRange(self.SOAPY_SDR_RX, 0, elem)
                            normalized_gain = float(self.gain) / 100.0  # Normalize to 0-1
                            elem_gain = gain_range.minimum() + normalized_gain * (gain_range.maximum() - gain_range.minimum())
                            self.sdr.setGain(self.SOAPY_SDR_RX, 0, elem, elem_gain)
                        except Exception:
                            continue
                except Exception:
                    print("Could not set gain elements, using default gain")
        
        # Create streaming objects
        self.rx_stream = self.sdr.setupStream(self.SOAPY_SDR_RX, self.SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx_stream)
    
    def _configure_rtlsdr_device(self):
        """
        Configure an RTL-SDR device with the specified parameters.
        """
        if self.sdr is None:
            raise ValueError("RTL-SDR device not initialized")
        
        # Set sample rate
        self.sdr.sample_rate = self.sample_rate
        
        # Set gain mode and gain
        if self.gain == 'auto':
            self.sdr.gain = 'auto'
        else:
            try:
                gain_value = float(self.gain)
                self.sdr.gain = gain_value
            except Exception as e:
                print(f"Error setting gain to {self.gain}: {e}")
                self.sdr.gain = 'auto'
    
    def set_center_freq(self, freq):
        """
        Set the center frequency of the SDR device.
        
        Parameters:
        freq (float): Center frequency in Hz
        """
        if self.sdr is None:
            raise ValueError("SDR device not initialized. Call initialize_device() first.")
        
        try:
            if self.soapy_available and hasattr(self.sdr, 'setFrequency'):
                # SoapySDR device
                self.sdr.setFrequency(self.SOAPY_SDR_RX, 0, freq)
            else:
                # RTL-SDR device
                self.sdr.center_freq = freq
        except Exception as e:
            print(f"Error setting center frequency to {freq/1e6} MHz: {e}")
    
    def read_samples(self, num_samples):
        """
        Read samples from the SDR device.
        
        Parameters:
        num_samples (int): Number of samples to read
        
        Returns:
        numpy.ndarray: Complex IQ samples
        """
        if self.sdr is None:
            raise ValueError("SDR device not initialized. Call initialize_device() first.")
        
        import numpy as np
        
        try:
            if self.soapy_available and hasattr(self.sdr, 'readStream'):
                # SoapySDR device
                buffer = np.zeros(num_samples, dtype=np.complex64)
                buffer_size = len(buffer)
                
                # Read data in chunks
                total_samples = 0
                while total_samples < buffer_size:
                    samples_to_read = buffer_size - total_samples
                    buffer_chunk = np.zeros(samples_to_read, dtype=np.complex64)
                    
                    flags = 0
                    timeNs = 0
                    sr = self.sdr.readStream(self.rx_stream, [buffer_chunk], samples_to_read, flags, timeNs)
                    
                    if sr.ret > 0:
                        buffer[total_samples:total_samples + sr.ret] = buffer_chunk[:sr.ret]
                        total_samples += sr.ret
                    else:
                        # Error or timeout
                        if sr.ret == 0:  # Timeout
                            continue
                        else:
                            break  # Error
                
                if total_samples == 0:
                    raise ValueError("Failed to read any samples")
                
                return buffer[:total_samples]
            else:
                # RTL-SDR device
                return self.sdr.read_samples(num_samples)
        except Exception as e:
            print(f"Error reading samples: {e}")
            # Return zeros if read fails
            return np.zeros(num_samples, dtype=np.complex64)
    
    def close(self):
        """
        Close the SDR device and release resources.
        """
        try:
            if self.sdr is not None:
                if self.soapy_available and hasattr(self.sdr, 'deactivateStream'):
                    # SoapySDR device
                    try:
                        self.sdr.deactivateStream(self.rx_stream)
                        self.sdr.closeStream(self.rx_stream)
                    except Exception as e:
                        print(f"Error closing SoapySDR stream: {e}")
                else:
                    # RTL-SDR device
                    try:
                        self.sdr.close()
                    except Exception as e:
                        print(f"Error closing RTL-SDR device: {e}")
                
                self.sdr = None
        except Exception as e:
            print(f"Error during device cleanup: {e}")

def get_sdr_device(config):
    """
    Factory function to create and initialize an SDR device based on configuration.
    
    Parameters:
    config (dict or configparser.ConfigParser): Configuration containing SDR settings
    
    Returns:
    SDRManager: Initialized SDR manager object
    """
    # Create and initialize the SDR manager with the entire config object
    sdr_manager = SDRManager(config)
    
    try:
        sdr_manager.initialize_device()
        return sdr_manager
    except Exception as e:
        print(f"Failed to initialize SDR device: {e}")
        return None
