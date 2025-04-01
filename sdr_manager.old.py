import importlib
import sys

class SDRManager:
    """
    A class to manage SDR device initialization and configuration.
    Supports various SDR types like RTL-SDR, LimeSDR, HackRF, etc.
    """
    
    def __init__(self, sdr_type='rtlsdr', sample_rate=2.048e6, gain='auto'):
        """
        Initialize the SDR manager.
        
        Parameters:
        sdr_type (str): The type of SDR to use ('rtlsdr', 'limesdr', 'hackrf', 'rsp1a', 'usrp')
        sample_rate (float): Sample rate in Hz
        gain (str or float): Gain setting, 'auto' or a specific value
        """
        self.sdr_type = sdr_type
        self.sample_rate = sample_rate
        self.gain = gain
        self.sdr = None
        
    def initialize_device(self):
        """
        Initialize the SDR device based on the specified type.
        
        Returns:
        object: Initialized SDR device
        
        Raises:
        ImportError: If the required SDR library is not installed
        ValueError: If the specified SDR type is not supported
        """
        try:
            if self.sdr_type == 'rtlsdr':
                from rtlsdr import RtlSdr
                self.sdr = RtlSdr()
            elif self.sdr_type in ['limesdr', 'hackrf', 'rsp1a', 'usrp']:
                try:
                    from SoapySDR import Device as SoapyDevice
                    self.sdr = SoapyDevice(dict(driver=self.sdr_type))
                except ImportError:
                    raise ImportError(f"SoapySDR Python bindings not found. Please install them to use {self.sdr_type}.")
            else:
                raise ValueError(f"Unsupported SDR type: {self.sdr_type}")
            
            # Configure the device
            self.configure_device()
            return self.sdr
            
        except ImportError as e:
            print(f"Error importing SDR library: {e}")
            print(f"Please ensure the appropriate SDR libraries are installed.")
            raise
        
    def configure_device(self):
        """
        Configure the SDR device with the specified parameters.
        """
        if self.sdr is None:
            raise ValueError("SDR device not initialized. Call initialize_device() first.")
        
        # Different SDR types have different configuration methods
        if self.sdr_type == 'rtlsdr':
            self.sdr.sample_rate = self.sample_rate
            if self.gain == 'auto':
                self.sdr.gain = 'auto'
            else:
                self.sdr.gain = float(self.gain)
        else:
            # For SoapySDR-based devices
            # This is a simplified example, actual parameters may vary by device
            self.sdr.setSampleRate(0, 0, self.sample_rate)
            if self.gain != 'auto':
                self.sdr.setGain(0, 0, float(self.gain))
    
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
        
        if self.sdr_type == 'rtlsdr':
            return self.sdr.read_samples(num_samples)
        else:
            # For SoapySDR-based devices
            # This is a simplified example, actual implementation may vary
            return self.sdr.readStream(0, [num_samples], 0)
    
    def set_center_freq(self, freq):
        """
        Set the center frequency of the SDR device.
        
        Parameters:
        freq (float): Center frequency in Hz
        """
        if self.sdr is None:
            raise ValueError("SDR device not initialized. Call initialize_device() first.")
        
        if self.sdr_type == 'rtlsdr':
            self.sdr.center_freq = freq
        else:
            # For SoapySDR-based devices
            self.sdr.setFrequency(0, 0, freq)
    
    def close(self):
        """
        Close the SDR device and release resources.
        """
        if self.sdr is not None:
            if self.sdr_type == 'rtlsdr':
                self.sdr.close()
            else:
                # Some SoapySDR devices might need specific cleanup
                pass
            self.sdr = None

def get_sdr_device(config):
    """
    Factory function to create and initialize an SDR device based on configuration.
    
    Parameters:
    config (dict or configparser.ConfigParser): Configuration containing SDR settings
    
    Returns:
    SDRManager: Initialized SDR manager object
    """
    # Handle different config input types
    if hasattr(config, 'get'):
        # It's a ConfigParser object
        sdr_type = config.get('GENERAL', 'sdr_type', fallback='rtlsdr')
        sample_rate = float(config.get('GENERAL', 'sample_rate', fallback='2.048e6'))
        gain = config.get('GENERAL', 'gain', fallback='auto')
    else:
        # It's a dictionary
        sdr_type = config.get('sdr_type', 'rtlsdr')
        sample_rate = float(config.get('sample_rate', 2.048e6))
        gain = config.get('gain', 'auto')
    
    # Create and initialize the SDR manager
    sdr_manager = SDRManager(sdr_type, sample_rate, gain)
    sdr_manager.initialize_device()
    
    return sdr_manager
