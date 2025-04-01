import paho.mqtt.client as mqtt
import os
import json
import time
import socket
import subprocess
import re
import threading

class MQTTClient:
    """
    MQTT client for publishing monitoring data and receiving commands.
    """
    
    def __init__(self, broker='localhost', port=1883, topics=None, client_id=None):
        """
        Initialize the MQTT client.
        
        Parameters:
        broker (str): MQTT broker address
        port (int): MQTT broker port
        topics (dict): Dictionary of topics for different message types
        client_id (str): Client ID for MQTT connection (auto-generated if None)
        """
        self.broker = broker
        self.port = port
        self.topics = topics or {
            'anomalies': 'hamradio/anomalies',
            'modulation': 'hamradio/modulation',
            'signal_strength': 'hamradio/signal_strength',
            'coordinates': 'hamradio/coordinates'
        }
        self.client_id = client_id or self.get_device_name()
        self.client = None
        self.connected = False
        self.reconnect_delay = 1  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 60  # Maximum reconnect delay in seconds
        self.stopping = False
        self.command_handlers = {}
        
        # Environmental variables
        self.mqtt_user = os.getenv("MQTT_USER", None)
        self.mqtt_password = os.getenv("MQTT_PASSWORD", None)
        self.mqtt_tls = int(os.getenv("MQTT_TLS", 0))  # 1 = Enable TLS, 0 = Disable
        self.mqtt_use_ca_cert = int(os.getenv("MQTT_USE_CA_CERT", 0))  # 1 = Use CA Cert, 0 = Disable
        self.mqtt_ca_cert = os.getenv("MQTT_CA_CERT", "/path/to/ca.crt")  # Path to CA Cert
    
    def get_device_name(self):
        """
        Get a unique device name for MQTT client ID.
        Uses environment variables if available, or hostname + MAC.
        
        Returns:
        str: Unique device name for client identification
        """
        # Check for Balena device name
        device_name = os.getenv("BALENA_DEVICE_NAME_AT_INIT")
        
        if not device_name:
            try:
                # Get hostname
                host = subprocess.check_output("hostname", shell=True, text=True).strip()
                
                # Get MAC address
                mac = self.get_primary_mac()
                
                device_name = f"{host}-{mac}"
            except Exception as e:
                print(f"Error getting device name: {e}")
                # Fallback to a random name
                device_name = f"spectrumAlert-{int(time.time())}"
        
        return device_name
    
    def get_primary_mac(self):
        """
        Get the primary MAC address (uppercase, no colons).
        
        Returns:
        str: MAC address
        """
        try:
            if os.name == 'posix':  # Linux/Mac
                mac_output = subprocess.check_output(
                    "ip link show | grep -m 1 'link/ether' | awk '{print $2}'",
                    shell=True, text=True
                ).strip()
            elif os.name == 'nt':  # Windows
                mac_output = subprocess.check_output(
                    "getmac /NH /V | findstr Ethernet | for /f \"tokens=3\" %i in ('more') do @echo %i",
                    shell=True, text=True
                ).strip()
            else:
                # Fallback method using socket and uuid
                import uuid
                mac_output = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff)
                                      for elements in range(0, 48, 8)][::-1])
            
            # Remove colons and convert to uppercase
            mac_clean = re.sub(r'[:]', '', mac_output).upper()
            
            return mac_clean
        except Exception as e:
            print(f"Error getting MAC address: {e}")
            return "UNKNOWNMAC"
    
    def connect(self):
        """
        Connect to the MQTT broker.
        
        Returns:
        bool: True if connection successful, False otherwise
        """
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        
        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Enable automatic reconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        
        # Use TLS if enabled
        if self.mqtt_tls:
            print(f"Enabling TLS for MQTT...")
            self.client.tls_set(ca_certs=self.mqtt_ca_cert if self.mqtt_use_ca_cert else None)
        
        # Set username/password if provided
        if self.mqtt_user and self.mqtt_password:
            self.client.username_pw_set(self.mqtt_user, self.mqtt_password)
        
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            
            # Wait for connection to establish
            timeout = 5  # seconds
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            return self.connected
        except Exception as e:
            print(f"MQTT connection error: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """
        Callback for when the client connects to the broker.
        
        Parameters:
        client: The client instance
        userdata: The user data as set in Client() or userdata_set()
        flags: Response flags sent by the broker
        rc: The connection result
        properties: The MQTT v5.0 properties from connect response (MQTT v5.0 only)
        """
        if rc == 0:
            print("Connected to MQTT broker successfully!")
            self.connected = True
            
            # Subscribe to command topics if any
            for topic in self.command_handlers.keys():
                self.client.subscribe(topic)
        else:
            print(f"Failed to connect to MQTT broker, return code: {rc}")
            self.connected = False
    
    def _on_disconnect(self, client, userdata, rc, *args):
        """
        Callback for when the client disconnects from the broker.
        
        Parameters:
        client: The client instance
        userdata: The user data as set in Client() or userdata_set()
        rc: The disconnection result
        """
        print(f"Disconnected from MQTT broker with code: {rc}")
        self.connected = False
        
        if not self.stopping:
            # Try to reconnect if not stopping intentionally
            print("Attempting to reconnect...")
            try:
                self.client.reconnect()
            except Exception as e:
                print(f"Reconnect failed: {e}")
    
    def _on_message(self, client, userdata, msg):
        """
        Callback for when a message is received from the broker.
        
        Parameters:
        client: The client instance
        userdata: The user data as set in Client() or userdata_set()
        msg: The received message
        """
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            if topic in self.command_handlers:
                handler = self.command_handlers[topic]
                handler(payload)
            else:
                print(f"Received message on topic {topic}: {payload}")
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def publish(self, message_type, payload):
        """
        Publish a message to the appropriate topic.
        
        Parameters:
        message_type (str): Type of message ('anomalies', 'signal_strength', etc.)
        payload: Message payload (string or JSON object)
        
        Returns:
        bool: True if successful, False otherwise
        """
        if not self.connected:
            print("Not connected to MQTT broker. Cannot publish message.")
            return False
        
        if message_type not in self.topics:
            print(f"Unknown message type: {message_type}")
            return False
        
        topic = self.topics[message_type]
        
        # Convert dict/object to JSON string
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        
        try:
            result = self.client.publish(topic, payload)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"Error publishing message: {e}")
            return False
    
    def publish_anomaly(self, frequency, features=None, confidence=None):
        """
        Publish an anomaly detection message.
        
        Parameters:
        frequency (float): Frequency where the anomaly was detected
        features (list, optional): Feature vector of the anomaly
        confidence (float, optional): Confidence level of the anomaly detection
        
        Returns:
        bool: True if successful, False otherwise
        """
        payload = {
            "timestamp": int(time.time()),
            "frequency": frequency,
            "frequency_mhz": frequency / 1e6,
            "device_id": self.client_id
        }
        
        if features is not None:
            payload["features"] = features
        
        if confidence is not None:
            payload["confidence"] = confidence
        
        return self.publish('anomalies', payload)
    
    def publish_signal_strength(self, frequency, strength_db):
        """
        Publish a signal strength measurement.
        
        Parameters:
        frequency (float): Frequency of the measurement
        strength_db (float): Signal strength in dB
        
        Returns:
        bool: True if successful, False otherwise
        """
        payload = {
            "timestamp": int(time.time()),
            "frequency": frequency,
            "frequency_mhz": frequency / 1e6,
            "strength_db": strength_db,
            "device_id": self.client_id
        }
        
        return self.publish('signal_strength', payload)
    
    def publish_coordinates(self, latitude, longitude, altitude=None):
        """
        Publish receiver coordinates.
        
        Parameters:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        altitude (float, optional): Altitude in meters
        
        Returns:
        bool: True if successful, False otherwise
        """
        payload = {
            "timestamp": int(time.time()),
            "latitude": latitude,
            "longitude": longitude,
            "device_id": self.client_id
        }
        
        if altitude is not None:
            payload["altitude"] = altitude
        
        return self.publish('coordinates', payload)
    
    def register_command_handler(self, topic, handler):
        """
        Register a handler for a command topic.
        
        Parameters:
        topic (str): Topic to subscribe to
        handler (callable): Function to call when a message is received on this topic
        """
        if not callable(handler):
            raise ValueError("Handler must be callable")
        
        self.command_handlers[topic] = handler
        
        # Subscribe to the topic if already connected
        if self.connected and self.client:
            self.client.subscribe(topic)
    
    def disconnect(self):
        """
        Disconnect from the MQTT broker.
        """
        self.stopping = True
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False