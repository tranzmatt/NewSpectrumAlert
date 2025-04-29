# rfd900x_detector.py
"""
RFD900x Telemetry Detector with MQTT Reporting
------------------------------------------------

Scans the 902–928 MHz ISM band for RFD900x telemetry bursts using SoapySDR
and reports detections via MQTT. Supports HackRF, RTL-SDR, and any SoapySDR device.

Configuration via env vars or CLI:
    SDR_DEVICE, MQTT_HOST, MQTT_PORT, MQTT_TOPIC,
    MQTT_USER, MQTT_PASSWORD, MQTT_TLS, MQTT_CA_CERT
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from threading import Thread, Event

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import paho.mqtt.client as mqtt

# ----------------------------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('rfd900x_detector')
exit_event = Event()

# ----------------------------------------------------------------------------
# MQTT Handler
# ----------------------------------------------------------------------------
class MQTTHandler:
    def __init__(self, broker, port, topic, username=None, password=None, tls=False, ca_cert=None):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        if username and password:
            self.client.username_pw_set(username, password)
        if tls:
            self.client.tls_set(ca_certs=ca_cert)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

    def connect(self):
        try:
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f"MQTT connect failed: {e}")
            return False

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("MQTT disconnected")

    def publish(self, payload):
        try:
            result = self.client.publish(self.topic, json.dumps(payload))
            result.wait_for_publish()
        except Exception as e:
            logger.error(f"MQTT publish error: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        logger.warning(f"MQTT disconnected (rc={rc})")

    def _on_publish(self, client, userdata, mid):
        logger.info(f"MQTT message published (mid={mid})")

# ----------------------------------------------------------------------------
# RFD900x Detector
# ----------------------------------------------------------------------------
class RFD900xDetector:
    # Frequency scan range
    MIN_FREQ = 902e6
    MAX_FREQ = 928e6
    FREQ_STEP = 2e6

    # SDR defaults
    DEFAULT_SAMPLE_RATE = 8e6
    DEFAULT_BANDWIDTH = 6e6
    DEFAULT_GAIN = 40

    # Detection thresholds
    DB_THRESHOLD = 30.0       # dB above noise floor
    MIN_SNR = 30.0            # minimum SNR in dB
    MIN_BURST_BW_KHZ = 20.0  # filter out narrower spikes
    FFT_SIZE = 4096
    SAMPLES_PER_READ = 32768

    def __init__(self, device_args, mqtt_handler, dwell_time=0.5, gain=None):
        self.device_args = device_args
        self.mqtt = mqtt_handler
        self.dwell_time = dwell_time
        self.sample_rate = self.DEFAULT_SAMPLE_RATE
        self.bandwidth = self.DEFAULT_BANDWIDTH
        self.gain = gain if gain is not None else self.DEFAULT_GAIN

        # Detect serial from SoapySDR enumeration
        self.serial = None
        try:
            for info in SoapySDR.Device.enumerate(self.device_args):
                if 'serial' in info:
                    self.serial = info['serial']
                    break
        except Exception:
            pass

        # Append serial to device args for reporting
        if self.serial:
            self.device_id = f"{self.device_args},serial={self.serial}"
        else:
            self.device_id = self.device_args
        self.freqs = np.arange(self.MIN_FREQ, self.MAX_FREQ + self.FREQ_STEP, self.FREQ_STEP)
        self.current_idx = 0
        self.last_switch = time.time()

        # SDR stream placeholders
        self.sdr = None
        self.rx_stream = None
        self.running = False
        self.thread = None

    def setup(self):
        logger.info(f"Opening SDR: {self.device_id}")
        self.sdr = SoapySDR.Device(self.device_args)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
        self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)
        freq = float(self.freqs[self.current_idx])
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, freq)
        logger.info(f"Tuned to {freq/1e6:.3f} MHz, gain={self.gain} dB")
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

    def start(self):
        self.setup()
        self.sdr.activateStream(self.rx_stream)
        self.running = True
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Detector started")

    def stop(self):
        logger.info("Stopping detector...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.rx_stream:
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)
        if self.sdr:
            self.sdr = None
        self.mqtt.disconnect()
        logger.info("Detector stopped")

    def _loop(self):
        buffer = np.zeros(self.SAMPLES_PER_READ, dtype=np.complex64)
        while self.running and not exit_event.is_set():
            now = time.time()
            if now - self.last_switch >= self.dwell_time:
                self.current_idx = (self.current_idx + 1) % len(self.freqs)
                freq = float(self.freqs[self.current_idx])
                self.sdr.setFrequency(SOAPY_SDR_RX, 0, freq)
                logger.debug(f"Switched to {freq/1e6:.3f} MHz")
                self.last_switch = now

            sr = self.sdr.readStream(self.rx_stream, [buffer], len(buffer), timeoutUs=500000)
            if sr.ret > 0:
                self._analyze(buffer[:sr.ret], float(self.freqs[self.current_idx]))

    def _analyze(self, samples, center_freq):
        # FFT analysis
        spec = np.fft.fftshift(np.fft.fft(samples[:self.FFT_SIZE]))
        spec_db = 10 * np.log10(np.abs(spec)**2 + 1e-12)
        noise = np.percentile(spec_db, 25)
        df = self.sample_rate / len(spec_db)

        for i in range(1, len(spec_db)-1):
            if spec_db[i] > spec_db[i-1] and spec_db[i] > spec_db[i+1] and (spec_db[i] - noise) >= self.DB_THRESHOLD:
                # measure bandwidth
                left = i
                while left > 0 and spec_db[left] > noise + self.DB_THRESHOLD:
                    left -= 1
                right = i
                while right < len(spec_db)-1 and spec_db[right] > noise + self.DB_THRESHOLD:
                    right += 1
                bandwidth_hz = (right - left) * df
                bandwidth_khz = bandwidth_hz / 1000
                if bandwidth_khz < self.MIN_BURST_BW_KHZ:
                    continue
                snr = spec_db[i] - noise
                if snr >= self.MIN_SNR:
                    freq = center_freq + (i - len(spec_db)//2) * df
                    power_db = spec_db[i]
                    self._report(freq, power_db, snr, bandwidth_khz)

    def _report(self, freq_hz, power_db, snr, bandwidth_khz):
        freq_mhz = freq_hz / 1e6
        half_bw_mhz = (self.bandwidth / 1e6) / 2
        msg = {
            "type": "RFD900x",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "frequency_mhz": freq_mhz,
            "power": power_db,
            "snr": snr,
            "device": self.device_id,
            "detection_type": "rfd900x_burst",
            "signal_desc": "RFD900x telemetry burst",
            "bandwidth_khz": round(bandwidth_khz, 2),
            "min_freq_mhz": freq_mhz - half_bw_mhz,
            "max_freq_mhz": freq_mhz + half_bw_mhz
        }
        logger.info(f"Burst @ {freq_mhz:.4f} MHz, SNR={snr:.1f} dB, Power={power_db:.1f} dB, BW={bandwidth_khz:.1f} kHz")
        self.mqtt.publish(msg)

# ----------------------------------------------------------------------------
# Utility: List Devices
# ----------------------------------------------------------------------------
def list_devices():
    print("Available SoapySDR devices:")
    for dev in SoapySDR.Device.enumerate():
        print(dev)
    sys.exit(0)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def signal_handler(sig, frame):
    logger.info("Interrupt received, shutting down...")
    exit_event.set()


def main():
    parser = argparse.ArgumentParser(description="RFD900x Telemetry Detector")
    parser.add_argument('--list-devices', action='store_true', help='List SDR devices')
    parser.add_argument('--device', default=os.getenv('SDR_DEVICE', 'driver=hackrf'),
                        help="SDR device args or SDR_DEVICE env var")
    parser.add_argument('--gain', type=float, default=None, help='SDR gain in dB')
    parser.add_argument('--mqtt-broker', default=os.getenv('MQTT_HOST', 'localhost'), help='MQTT broker (env)')
    parser.add_argument('--mqtt-port', type=int, default=int(os.getenv('MQTT_PORT', 1883)), help='MQTT port (env)')
    parser.add_argument('--mqtt-topic', default=os.getenv('MQTT_TOPIC', 'sensors/rf/rfd900x/detection'), help='MQTT topic (env)')
    parser.add_argument('--mqtt-user', default=os.getenv('MQTT_USER'), help='MQTT user (env)')
    parser.add_argument('--mqtt-password', default=os.getenv('MQTT_PASSWORD'), help='MQTT password (env)')
    parser.add_argument('--mqtt-tls', action='store_true',
                        default=(os.getenv('MQTT_TLS','').lower() in ('true','1')),
                        help='Enable MQTT TLS (env)')
    parser.add_argument('--mqtt-ca-cert', default=os.getenv('MQTT_CA_CERT'), help='MQTT CA cert (env)')
    parser.add_argument('--dwell-time', type=float, default=0.5, help='Seconds per freq step')
    parser.add_argument('--db-threshold', type=float, default=RFD900xDetector.DB_THRESHOLD, help='dB threshold')
    parser.add_argument('--min-snr', type=float, default=RFD900xDetector.MIN_SNR, help='Min SNR')
    args = parser.parse_args()

    if args.list_devices:
        list_devices()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    mqtt = MQTTHandler(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=args.mqtt_topic,
        username=args.mqtt_user,
        password=args.mqtt_password,
        tls=args.mqtt_tls,
        ca_cert=args.mqtt_ca_cert
    )
    if not mqtt.connect():
        sys.exit(1)

    detector = RFD900xDetector(
        device_args=args.device,
        mqtt_handler=mqtt,
        dwell_time=args.dwell_time,
        gain=args.gain
    )
    detector.DB_THRESHOLD = args.db_threshold
    detector.MIN_SNR = args.min_snr

    detector.start()
    while not exit_event.is_set():
        time.sleep(1)

    detector.stop()
    logger.info("Exited cleanly.")

if __name__ == '__main__':
    main()

