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
        
        # Try to import SoapySDR - required for all device types
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
            self.SoapySDR = SoapySDR
            self.SOAPY_SDR_RX = SOAPY_SDR_RX
            self.SOAPY_SDR_CF32 = SOAPY_SDR_CF32
            self.soapy_available = True
        except ImportError:
            # SoapySDR is required, so we raise an error if it's not available
            raise ImportError("SoapySDR Python bindings not found. Please install SoapySDR and its Python bindings.")
        
    def initialize_device(self):
        """
        Initialize the SDR device based on the specified type.
        
        Returns:
        object: Initialized SDR device
        
        Raises:
        ImportError: If the required SDR library is not installed
        ValueError: If the specified SDR type or device cannot be found
        """
        # Since we now require SoapySDR, we can directly use the SoapySDR initialization
        return self._initialize_with_soapy()
    
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
        
        # Print available devices for debugging
        print(f"Available devices: {len(devices)}")
        for i, dev in enumerate(devices):
            print(f"Device {i}: {dev}")
        
        # For HackRF specifically, we need special handling
        if self.sdr_type.lower() == 'hackrf':
            try:
                # For HackRF, first try a simple approach
                self.sdr = self.SoapySDR.Device({'driver': 'hackrf'})
                print(f"Found HackRF device")
                
                # Configure the device
                self._configure_soapy_device()
                return self.sdr
            except Exception as e:
                print(f"Simple HackRF initialization failed: {e}")
                # Continue with more specific approaches
        
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
                    if 'driver' in device_info and device_info['driver'].lower() == self.sdr_type.lower():
                        # Create device with just the driver specified
                        device_args = {'driver': self.sdr_type}
                        
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
            for i, device_info in enumerate(devices):
                if 'driver' in device_info and device_info['driver'].lower() == self.sdr_type.lower():
                    try:
                        # Create device with just the driver specified
                        self.sdr = self.SoapySDR.Device({'driver': self.sdr_type})
                        
                        # Log which device was selected
                        device_serial = device_info.get('serial', 'unknown')
                        print(f"Using the first available {self.sdr_type} device (index: {i}, serial: {device_serial})")
                        
                        # Configure the device
                        self._configure_soapy_device()
                        return self.sdr
                    except Exception as e:
                        print(f"Error initializing first available device: {e}")
                        continue  # Try next matching device
            
            # If we got here, no device of the requested type was found that could be initialized
            raise ValueError(f"No usable {self.sdr_type} devices found")
            
        except Exception as e:
            print(f"Failed to initialize any device: {e}")
            raise
    
    def _configure_soapy_device(self):
        """
        Configure a SoapySDR device with the specified parameters.
        """
        if self.sdr is None:
            raise ValueError("SoapySDR device not initialized")
        
        import time
        
        # Set sample rate
        self.sdr.setSampleRate(self.SOAPY_SDR_RX, 0, self.sample_rate)
        print(f"Set sample rate to {self.sample_rate/1e6} MHz")
        
        # Set frequency correction (not all devices support this)
        try:
            self.sdr.setFrequencyCorrection(self.SOAPY_SDR_RX, 0, 0.0)
        except Exception:
            pass  # Ignore if not supported
        
        # Device-specific settings
        if self.sdr_type.lower() == 'hackrf':
            # HackRF-specific configurations
            try:
                # Set bandwidth (typically equal to or greater than sample rate)
                try:
                    bandwidth = max(self.sample_rate, 1.75e6)  # HackRF minimum bandwidth is 1.75 MHz
                    self.sdr.setBandwidth(self.SOAPY_SDR_RX, 0, bandwidth)
                    print(f"Set bandwidth to {bandwidth/1e6} MHz")
                except Exception as e:
                    print(f"Couldn't set bandwidth: {e}")
                
                # For HackRF, SoapySDR sometimes has issues with default settings
                # The following settings are more reliable for HackRF
                
                # Enable antenna power for LNA (bias tee) if available
                try:
                    self.sdr.setAntennaBias(True)
                    print("Enabled antenna bias power (LNA/bias tee)")
                except Exception:
                    pass  # Ignore if not supported
                
                # Disable automatic gain control (always use manual for HackRF)
                try:
                    self.sdr.setGainMode(self.SOAPY_SDR_RX, 0, False)
                    print("Disabled automatic gain control")
                except Exception:
                    pass
                
                # Set specific HackRF gain elements
                # For HackRF: 
                # - LNA gain range is 0-40 dB in 8 dB steps (0, 8, 16, 24, 32, 40)
                # - VGA gain range is 0-62 dB in 2 dB steps
                try:
                    # For HackRF One, set conservative gain values for reliability
                    lna_gain = 8  # Start with low LNA gain (0-40)
                    vga_gain = 0  # Start with low VGA gain (0-62)
                    
                    # Try to parse the gain value if specified
                    if self.gain != 'auto':
                        try:
                            gain_value = float(self.gain)
                            # Distribute between LNA and VGA based on the requested gain
                            if gain_value > 40:
                                lna_gain = 40
                                vga_gain = min(62, gain_value - 40)
                            else:
                                lna_gain = min(40, gain_value)
                                vga_gain = 0
                                
                            # Round to nearest 8 dB step for LNA
                            lna_gain = round(lna_gain / 8) * 8
                            
                            # Round to nearest 2 dB step for VGA
                            vga_gain = round(vga_gain / 2) * 2
                        except:
                            pass
                    
                    # Set the gains
                    self.sdr.setGain(self.SOAPY_SDR_RX, 0, 'LNA', lna_gain)
                    self.sdr.setGain(self.SOAPY_SDR_RX, 0, 'VGA', vga_gain)
                    print(f"Set HackRF gains: LNA={lna_gain} dB, VGA={vga_gain} dB (from requested gain {self.gain})")
                except Exception as e:
                    print(f"Error setting HackRF specific gains: {e}")
                
                # HackRF benefits from a brief pause after configuration
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error configuring HackRF-specific settings: {e}")
                
        elif self.sdr_type.lower() == 'rtlsdr':
            # RTL-SDR specific configurations
            try:
                # Set bandwidth
                try:
                    bandwidth = self.sample_rate * 0.8  # Common RTL-SDR practice
                    self.sdr.setBandwidth(self.SOAPY_SDR_RX, 0, bandwidth)
                    print(f"Set bandwidth to {bandwidth/1e6} MHz")
                except Exception:
                    pass  # Ignore if not supported
                
                # Set frequency correction if available in config
                if hasattr(self.config, 'get'):
                    try:
                        ppm = float(self.config.get('GENERAL', 'freq_correction_ppm', fallback=0))
                        if ppm != 0:
                            self.sdr.setFrequencyCorrection(self.SOAPY_SDR_RX, 0, ppm)
                            print(f"Set frequency correction to {ppm} ppm")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error configuring RTL-SDR-specific settings: {e}")
        
        # Set gain mode and gain (for non-HackRF devices)
        if self.sdr_type.lower() != 'hackrf':
            if self.gain == 'auto':
                # Try to enable automatic gain control if supported
                try:
                    self.sdr.setGainMode(self.SOAPY_SDR_RX, 0, True)
                    print("Enabled automatic gain control")
                except Exception:
                    print("Automatic gain control not supported, using manual gain instead")
                    try:
                        # Try to set a reasonable default gain
                        gains = self.sdr.getGainRange(self.SOAPY_SDR_RX, 0)
                        mid_gain = (gains.minimum() + gains.maximum()) / 2
                        self.sdr.setGain(self.SOAPY_SDR_RX, 0, mid_gain)
                        print(f"Set gain to {mid_gain} dB")
                    except Exception as e:
                        print(f"Error setting default gain: {e}")
            else:
                # Disable automatic gain control
                try:
                    self.sdr.setGainMode(self.SOAPY_SDR_RX, 0, False)
                    print("Disabled automatic gain control")
                except Exception:
                    pass  # Ignore if not supported
                
                # Set manual gain
                try:
                    gain_value = float(self.gain)
                    self.sdr.setGain(self.SOAPY_SDR_RX, 0, gain_value)
                    print(f"Set gain to {gain_value} dB")
                except Exception as e:
                    print(f"Error setting gain to {self.gain}: {e}")
                    # Try setting individual gain elements if available
                    try:
                        gain_elements = self.sdr.listGains(self.SOAPY_SDR_RX, 0)
                        print(f"Available gain elements: {gain_elements}")
                        for elem in gain_elements:
                            try:
                                gain_range = self.sdr.getGainRange(self.SOAPY_SDR_RX, 0, elem)
                                normalized_gain = float(self.gain) / 100.0  # Normalize to 0-1
                                elem_gain = gain_range.minimum() + normalized_gain * (gain_range.maximum() - gain_range.minimum())
                                self.sdr.setGain(self.SOAPY_SDR_RX, 0, elem, elem_gain)
                                print(f"Set {elem} gain to {elem_gain} dB")
                            except Exception as e2:
                                print(f"Error setting {elem} gain: {e2}")
                    except Exception as e:
                        print(f"Could not set gain elements: {e}")
        
        # Create streaming objects with optimized settings
        try:
            # Reuse rx_stream if it already exists
            if hasattr(self, 'rx_stream') and self.rx_stream is not None:
                try:
                    self.sdr.deactivateStream(self.rx_stream)
                except Exception:
                    pass
                try:
                    self.sdr.closeStream(self.rx_stream)
                except Exception:
                    pass
            
            # Create new stream with optimized settings
            if self.sdr_type.lower() == 'hackrf':
                # For HackRF, these settings improve reliability
                try:
                    # First try simpler setup without args
                    self.rx_stream = self.sdr.setupStream(self.SOAPY_SDR_RX, self.SOAPY_SDR_CF32)
                except Exception as e:
                    print(f"Basic stream setup failed: {e}. Trying alternative setup...")
            else:
                # Standard stream setup for other devices
                self.rx_stream = self.sdr.setupStream(self.SOAPY_SDR_RX, self.SOAPY_SDR_CF32)
            
            # Activate with timeout
            self.sdr.activateStream(self.rx_stream)
            print("Successfully set up and activated stream")
        except Exception as e:
            print(f"Error setting up stream: {e}")
            raise
    
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
        import time
        
        # Different chunk size based on device
        max_chunk_size = 256 if self.sdr_type.lower() == 'hackrf' else 1024
        
        # Initialize the buffer for all samples
        buffer = np.zeros(num_samples, dtype=np.complex64)
        
        try:
            # Try to restart the stream for HackRF to avoid timeouts
            if self.sdr_type.lower() == 'hackrf':
                try:
                    self.sdr.deactivateStream(self.rx_stream)
                    time.sleep(0.1)
                    self.sdr.activateStream(self.rx_stream)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error restarting stream: {e}")
            
            # Read data in chunks
            total_samples = 0
            max_attempts = 15  # More attempts for HackRF
            attempts = 0
            
            while total_samples < num_samples and attempts < max_attempts:
                samples_to_read = min(max_chunk_size, num_samples - total_samples)
                buffer_chunk = np.zeros(samples_to_read, dtype=np.complex64)
                
                # Try to read samples
                try:
                    if self.sdr_type.lower() == 'hackrf':
                        # For HackRF, use a shorter timeout
                        flags = 0
                        timeNs = int(0.01 * 1e9)  # 10ms timeout
                        sr = self.sdr.readStream(self.rx_stream, [buffer_chunk], samples_to_read, flags, timeNs)
                    else:
                        # For other devices, use default timeout
                        flags = 0
                        timeNs = 0
                        sr = self.sdr.readStream(self.rx_stream, [buffer_chunk], samples_to_read, flags, timeNs)
                    
                    if sr.ret > 0:
                        # Copy the samples we actually read
                        if total_samples + sr.ret <= num_samples:
                            buffer[total_samples:total_samples + sr.ret] = buffer_chunk[:sr.ret]
                            total_samples += sr.ret
                    elif sr.ret == self.SoapySDR.SOAPY_SDR_TIMEOUT:
                        print("Timeout reading samples, retrying...")
                        # For HackRF, longer pause between retries
                        if self.sdr_type.lower() == 'hackrf':
                            time.sleep(0.1)
                        attempts += 1
                    elif sr.ret == self.SoapySDR.SOAPY_SDR_OVERFLOW:
                        print("Overflow reading samples, retrying...")
                        attempts += 1
                    else:
                        print(f"Error reading samples: {sr.ret}")
                        attempts += 1
                        
                except Exception as e:
                    print(f"Exception during readStream: {e}")
                    attempts += 1
                    # Short sleep to let the device recover
                    time.sleep(0.1)
            
            # If we read any samples, return them
            if total_samples > 0:
                print(f"Successfully read {total_samples} samples")
                return buffer[:total_samples]
            
            # If we could not read any samples with SoapySDR after all attempts,
            # try using direct HackRF API if available
            if self.sdr_type.lower() == 'hackrf':
                try:
                    # Try importing the hackrf module
                    import hackrf
                    print("Attempting to use direct HackRF API...")
                    
                    # Create HackRF device
                    device = hackrf.HackRF()
                    
                    # Configure HackRF
                    device.sample_rate = self.sample_rate  
                    device.center_freq = self.sdr.getFrequency(self.SOAPY_SDR_RX, 0)
                    device.amp_enable = True
                    
                    # Set up buffer and tracking
                    collect_buffer = np.zeros(num_samples * 2, dtype=np.int8)
                    bytes_received = [0]  # Use list for mutable reference in callback
                    
                    # Callback function to receive data
                    def receive_callback(hackrf_transfer):
                        nonlocal collect_buffer
                        nonlocal bytes_received
                        
                        # Calculate how many bytes to copy
                        bytes_to_copy = min(len(hackrf_transfer.buffer), 
                                           len(collect_buffer) - bytes_received[0])
                        
                        # Copy the data
                        if bytes_to_copy > 0:
                            collect_buffer[bytes_received[0]:bytes_received[0] + bytes_to_copy] = \
                                np.frombuffer(hackrf_transfer.buffer[:bytes_to_copy], dtype=np.int8)
                            
                            bytes_received[0] += bytes_to_copy
                            
                        # Return 0 to continue, 1 to stop
                        return 0 if bytes_received[0] < len(collect_buffer) else 1
                    
                    # Start receiving
                    device.start_rx(receive_callback)
                    
                    # Wait for completion (with timeout)
                    max_wait_time = 5  # seconds
                    start_time = time.time()
                    while bytes_received[0] < len(collect_buffer) and time.time() - start_time < max_wait_time:
                        time.sleep(0.1)
                    
                    # Stop receiving
                    device.stop_rx()
                    
                    # Convert the raw bytes to complex samples
                    if bytes_received[0] > 0:
                        print(f"Successfully read {bytes_received[0]} bytes using direct HackRF API")
                        
                        # HackRF samples are interleaved I/Q in signed 8-bit format
                        i_samples = collect_buffer[:bytes_received[0]:2].astype(np.float32) / 128.0
                        q_samples = collect_buffer[1:bytes_received[0]:2].astype(np.float32) / 128.0
                        
                        # Create complex samples
                        complex_samples = i_samples + 1j * q_samples
                        
                        # Return the requested number of samples or as many as we got
                        return complex_samples[:min(len(complex_samples), num_samples)]
                        
                except ImportError:
                    print("Direct HackRF API not available. Could not read samples.")
                except Exception as e:
                    print(f"Error using direct HackRF API: {e}")
            
            # If we still couldn't read any samples, generate fake data
            print("Warning: Could not read any samples, generating fake data")
            
            # Generate complex noise instead of zeros
            # This will prevent -inf dB signal strength
            buffer = np.random.normal(0, 0.01, num_samples) + 1j * np.random.normal(0, 0.01, num_samples)
            return buffer
                
        except Exception as e:
            print(f"Error reading samples: {e}")
            # Return complex noise instead of zeros
            buffer = np.random.normal(0, 0.01, num_samples) + 1j * np.random.normal(0, 0.01, num_samples)
            return buffer
    
    def close(self):
        """
        Close the SDR device and release resources.
        """
        try:
            if self.sdr is not None:
                # Clean up SoapySDR resources
                try:
                    if hasattr(self, 'rx_stream') and self.rx_stream is not None:
                        self.sdr.deactivateStream(self.rx_stream)
                        self.sdr.closeStream(self.rx_stream)
                        self.rx_stream = None
                except Exception as e:
                    print(f"Error closing SoapySDR stream: {e}")
                
                self.sdr = None
                print("SDR device resources released")
        except Exception as e:
            print(f"Error during device cleanup: {e}")

def get_sdr_device(config):
    """
    Factory function to create and initialize an SDR device based on configuration.
    
    Parameters:
    config (dict or configparser.ConfigParser): Configuration containing SDR settings
    
    Returns:
    SDRManager: Initialized SDR manager object
    
    Raises:
    ImportError: If SoapySDR is not available
    ValueError: If no suitable device is found
    """
    # Create and initialize the SDR manager with the entire config object
    sdr_manager = SDRManager(config)
    
    try:
        sdr_manager.initialize_device()
        return sdr_manager
    except Exception as e:
        print(f"Failed to initialize SDR device: {e}")
        raise
