# rfd900x_detector.py
"""
RFD900x Telemetry Detector with MQTT Reporting (Full-Band Mode)
---------------------------------------------------------------

Captures the entire 902–928 MHz ISM band in one shot using SoapySDR
and reports real telemetry bursts via MQTT. No frequency hopping—one fixed center.

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
# Logging
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
        self.client = mqtt.Client()
        if username and password:
            self.client.username_pw_set(username, password)
        if tls:
            self.client.tls_set(ca_certs=ca_cert)
        self.client.on_connect = lambda c,u,f,rc: logger.info('Connected to MQTT' if rc==0 else f'MQTT failed rc={rc}')
        self.client.on_disconnect = lambda c,u,rc: logger.warning(f'MQTT disconnected rc={rc}')
        self.client.on_publish = lambda c,u,mid: logger.debug(f'MQTT published mid={mid}')
        self.broker, self.port, self.topic = broker, port, topic

    def connect(self):
        try:
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f'MQTT connect failed: {e}')
            return False

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, msg):
        self.client.publish(self.topic, json.dumps(msg))

# ----------------------------------------------------------------------------
# Full-Band RFD900x Detector with Burst Clustering
# ----------------------------------------------------------------------------
class RFD900xDetector:
    CENTER_FREQ = 915e6     # Center of 902–928 MHz
    SAMPLE_RATE = 20e6      # Capture 20 MHz at once
    BANDWIDTH = 20e6
    FFT_SIZE = 2048 #4096
    SAMPLES_PER_READ = 32768

    POWER_DB_THRESHOLD = 40.0       # only report above this power dB
    DB_ABOVE_NOISE = 35.0          # require dB above noise
    MIN_BURST_BW_KHZ = 100.0        # bursts at least this wide
    DEDUPE_WINDOW = 0.1            # seconds to suppress duplicate bursts

    def __init__(self, device_args, mqtt_handler, gain=None):
        self.device_args = device_args
        self.mqtt = mqtt_handler
        self.gain = gain

        # thresholds and dedupe
        self.POWER_DB_THRESHOLD = RFD900xDetector.POWER_DB_THRESHOLD
        self.DB_ABOVE_NOISE = RFD900xDetector.DB_ABOVE_NOISE
        self.MIN_BURST_BW_KHZ = RFD900xDetector.MIN_BURST_BW_KHZ
        self.dedupe_window = RFD900xDetector.DEDUPE_WINDOW
        self._recent = {}  # freq_key -> last_time

        # Detect serial if available
        self.serial = None
        try:
            for info in SoapySDR.Device.enumerate(self.device_args):
                if 'serial' in info:
                    self.serial = info['serial']
                    break
        except:
            pass
        self.device_id = f"{self.device_args},serial={self.serial}" if self.serial else self.device_args

        self.sdr = None
        self.rx_stream = None

    def setup(self):
        logger.info(f'Opening SDR: {self.device_id}')
        self.sdr = SoapySDR.Device(self.device_args)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.SAMPLE_RATE)
        self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.BANDWIDTH)
        if self.gain is not None:
            self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.CENTER_FREQ)
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

    def start(self):
        self.setup()
        self.sdr.activateStream(self.rx_stream)
        self.running = True
        Thread(target=self._loop, daemon=True).start()
        logger.info('Detector started in full-band mode')

    def stop(self):
        self.running = False
        time.sleep(0.1)
        if self.rx_stream:
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)
        if self.sdr:
            self.sdr = None
        self.mqtt.disconnect()
        logger.info('Detector stopped')

    def _loop(self):
        buf = np.zeros(self.SAMPLES_PER_READ, np.complex64)
        while self.running and not exit_event.is_set():
            sr = self.sdr.readStream(self.rx_stream, [buf], len(buf), timeoutUs=500000)
            if sr.ret > 0:
                self._analyze(buf[:sr.ret])

    def _analyze(self, samples):
        spec = np.fft.fftshift(np.fft.fft(samples[:self.FFT_SIZE]))
        spec_db = 10 * np.log10(np.abs(spec)**2 + 1e-12)
        noise = np.percentile(spec_db, 10)
        df = self.SAMPLE_RATE / len(spec_db)
        half_bw_mhz = (self.BANDWIDTH/1e6)/2
        now = time.time()

        # purge old dedupe entries
        self._recent = {k:v for k,v in self._recent.items() if now-v < self.dedupe_window}

        # create boolean mask of bins above threshold
        mask = (spec_db - noise) >= self.DB_ABOVE_NOISE
        diffs = np.diff(mask.astype(int))
        starts = np.where(diffs==1)[0]+1
        ends = np.where(diffs==-1)[0]+1
        if mask[0]: starts = np.r_[0, starts]
        if mask[-1]: ends = np.r_[ends, len(mask)]

        for s, e in zip(starts, ends):
            bw_khz = ((e - s) * df) / 1000
            if bw_khz < self.MIN_BURST_BW_KHZ:
                continue
            segment = spec_db[s:e]
            idx_offset = np.argmax(segment)
            idx = s + idx_offset
            power_db = spec_db[idx]
            if power_db < self.POWER_DB_THRESHOLD:
                continue
            freq = self.CENTER_FREQ + (idx - len(spec_db)//2) * df
            snr = spec_db[idx] - noise
            # dedupe by rounded kHz
            key = round(freq * 1000) / 1000
            if key in self._recent:
                continue
            self._recent[key] = now
            msg = {
                'type':'RFD900x',
                'timestamp':datetime.utcnow().isoformat()+'Z',
                'frequency_mhz':freq/1e6,
                'power':power_db,
                'snr':snr,
                'device':self.device_id,
                'detection_type':'rfd900x_burst',
                'signal_desc':'RFD900x telemetry burst',
                'bandwidth_khz':round(bw_khz,2),
                'min_freq_mhz':freq/1e6-half_bw_mhz,
                'max_freq_mhz':freq/1e6+half_bw_mhz
            }
            logger.info(f"Burst @ {freq/1e6:.4f} MHz, SNR={snr:.1f} dB, Power={power_db:.1f} dB, BW={bw_khz:.1f} kHz")
            self.mqtt.publish(msg)

# ----------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------

def list_devices():
    for d in SoapySDR.Device.enumerate(): print(d)
    sys.exit(0)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='RFD900x Telemetry Detector (Full-Band)')
    parser.add_argument('--list-devices', action='store_true', help='List SDR devices')
    parser.add_argument('--device', default=os.getenv('SDR_DEVICE','driver=hackrf'), help='SoapySDR device args or SDR_DEVICE env var')
    parser.add_argument('--gain', type=float, default=None, help='SDR gain (dB)')
    parser.add_argument('--db-above-noise', type=float, default=RFD900xDetector.DB_ABOVE_NOISE, help='dB above noise floor to detect')
    parser.add_argument('--power-db-threshold', type=float, default=RFD900xDetector.POWER_DB_THRESHOLD, help='Minimum power dB to report')
    parser.add_argument('--min-burst-bw', type=float, default=RFD900xDetector.MIN_BURST_BW_KHZ, help='Minimum burst bandwidth (kHz)')
    parser.add_argument('--dedupe-window', type=float, default=RFD900xDetector.DEDUPE_WINDOW, help='Dedupe window (s)')
    parser.add_argument('--mqtt-broker', default=os.getenv('MQTT_HOST','localhost'), help='MQTT broker (env)')
    parser.add_argument('--mqtt-port', type=int, default=int(os.getenv('MQTT_PORT',1883)), help='MQTT port (env)')
    parser.add_argument('--mqtt-topic', default=os.getenv('MQTT_TOPIC','sensors/rf/rfd900x/detection'), help='MQTT topic (env)')
    parser.add_argument('--mqtt-user', default=os.getenv('MQTT_USER'), help='MQTT user (env)')
    parser.add_argument('--mqtt-password', default=os.getenv('MQTT_PASSWORD'), help='MQTT password (env)')
    parser.add_argument('--mqtt-tls', action='store_true', default=os.getenv('MQTT_TLS','').lower() in ('true','1'), help='Enable MQTT TLS (env)')
    parser.add_argument('--mqtt-ca-cert', default=os.getenv('MQTT_CA_CERT'), help='MQTT CA cert (env)')
    args = parser.parse_args()

    if args.list_devices:
        list_devices()

    signal.signal(signal.SIGINT, lambda s,f: exit_event.set())
    signal.signal(signal.SIGTERM, lambda s,f: exit_event.set())

    mqtt = MQTTHandler(args.mqtt_broker, args.mqtt_port, args.mqtt_topic,
                       args.mqtt_user, args.mqtt_password, args.mqtt_tls, args.mqtt_ca_cert)
    if not mqtt.connect(): sys.exit(1)

    detector = RFD900xDetector(args.device, mqtt, args.gain)
    detector.DB_ABOVE_NOISE = args.db_above_noise
    detector.POWER_DB_THRESHOLD = args.power_db_threshold
    detector.MIN_BURST_BW_KHZ = args.min_burst_bw
    detector.dedupe_window = args.dedupe_window

    detector.start()
    while not exit_event.is_set():
        time.sleep(1)

    detector.stop()

if __name__=='__main__': main()

