import os
import time
import threading

class GPSManager:
    """
    A class to manage GPS functionality for direction finding.
    """
    
    def __init__(self, default_lat=0, default_lon=0, default_alt=0):
        """
        Initialize the GPS manager.
        
        Parameters:
        default_lat (float): Default latitude if GPS is not available
        default_lon (float): Default longitude if GPS is not available
        default_alt (float): Default altitude if GPS is not available
        """
        self.default_lat = default_lat
        self.default_lon = default_lon
        self.default_alt = default_alt
        self.gps_source = os.getenv("GPS_SOURCE", "none").lower()
        self.current_lat = default_lat
        self.current_lon = default_lon
        self.current_alt = default_alt
        self.gpsd_available = False
        self.last_update_time = 0
        self.update_interval = 5  # seconds
        self.running = False
        self.thread = None
        
        # Try to import gpsd library
        try:
            import gpsd
            self.gpsd = gpsd
            self.gpsd_available = True
        except ImportError:
            self.gpsd = None
            self.gpsd_available = False
            print("GPSD Python library not available. GPS functionality will be limited.")
    
    def start(self):
        """
        Start the GPS manager and begin periodic updates.
        
        Returns:
        bool: True if successful, False otherwise
        """
        if self.running:
            return True
        
        self.running = True
        
        # If using fixed coordinates, just use those
        if self.gps_source == "fixed":
            self.current_lat = float(os.getenv("GPS_FIX_LAT", self.default_lat))
            self.current_lon = float(os.getenv("GPS_FIX_LON", self.default_lon))
            self.current_alt = float(os.getenv("GPS_FIX_ALT", self.default_alt))
            print(f"Using fixed GPS coordinates: {self.current_lat}, {self.current_lon}, {self.current_alt}")
            return True
        
        # If using GPSD, start the update thread
        if self.gps_source == "gpsd" and self.gpsd_available:
            try:
                # Connect to GPSD
                self.gpsd.connect(host=os.getenv("GPSD_HOST", "localhost"), 
                                port=int(os.getenv("GPSD_PORT", "2947")))
                
                # Start the update thread
                self.thread = threading.Thread(target=self._update_loop, daemon=True)
                self.thread.start()
                return True
            except Exception as e:
                print(f"Error connecting to GPSD: {e}")
                self.running = False
                return False
        
        # No valid GPS source
        print(f"No valid GPS source configured. Using default coordinates.")
        return False
    
    def _update_loop(self):
        """
        Background thread to periodically update GPS coordinates.
        """
        while self.running:
            try:
                self.update()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in GPS update loop: {e}")
                time.sleep(self.update_interval)
    
    def update(self):
        """
        Update the current GPS coordinates.
        
        Returns:
        bool: True if successful, False otherwise
        """
        if not self.running or self.gps_source != "gpsd" or not self.gpsd_available:
            return False
        
        try:
            # Get current GPS data
            packet = self.gpsd.get_current()
            
            if packet.mode >= 2:  # 2D or 3D fix
                self.current_lat = packet.lat
                self.current_lon = packet.lon
                if packet.mode == 3:  # 3D fix (with altitude)
                    self.current_alt = packet.alt
                
                self.last_update_time = time.time()
                return True
            else:
                print("No GPS fix yet.")
                return False
        except Exception as e:
            print(f"Error updating GPS coordinates: {e}")
            return False
    
    def get_coordinates(self):
        """
        Get the current GPS coordinates.
        
        Returns:
        tuple: (latitude, longitude, altitude)
        """
        # Update if using GPSD and it's been too long since the last update
        if (self.gps_source == "gpsd" and self.gpsd_available and 
            time.time() - self.last_update_time > self.update_interval):
            self.update()
        
        return (self.current_lat, self.current_lon, self.current_alt)
    
    def has_fix(self):
        """
        Check if we have a GPS fix.
        
        Returns:
        bool: True if we have a fix, False otherwise
        """
        if self.gps_source == "fixed":
            return True
        
        if self.gps_source == "gpsd" and self.gpsd_available:
            try:
                packet = self.gpsd.get_current()
                return packet.mode >= 2
            except:
                return False
        
        return False
    
    def stop(self):
        """
        Stop the GPS manager and release resources.
        """
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
