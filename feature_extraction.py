import numpy as np

from config_manager import ConfigManager


class FeatureExtractor:
    def __init__(self, config: ConfigManager):

        self.config = config
        self.lite_mode = self.config.lite_mode

    def extract_features(self, iq_data):
        if self.lite_mode:
            return self.extract_basic_features(iq_data)
        else:
            return self.extract_basic_features(iq_data)

    def extract_basic_features(self, iq_data):
        I = np.real(iq_data)
        Q = np.imag(iq_data)
        amplitude = np.sqrt(I**2 + Q**2)

        mean_amplitude = np.mean(amplitude)
        std_amplitude = abs(np.std(amplitude))  # Avoid negative std due to numerical issues

        return [mean_amplitude, std_amplitude]

    def extract_enhanced_features(self, iq_data):
        I = np.real(iq_data)
        Q = np.imag(iq_data)

        amplitude = np.sqrt(I**2 + Q**2)
        phase = np.unwrap(np.angle(iq_data))
        fft_values = np.fft.fft(iq_data)
        fft_magnitude = np.abs(fft_values)

        mean_amplitude = np.mean(amplitude)
        std_amplitude = np.std(amplitude)

        mean_fft_magnitude = np.mean(fft_magnitude)
        std_fft_magnitude = np.std(fft_magnitude)

        if std_amplitude != 0:
            skew_amplitude = np.mean((amplitude - mean_amplitude) ** 3) / (std_amplitude ** 3)
            kurt_amplitude = np.mean((amplitude - mean_amplitude) ** 4) / (std_amplitude ** 4)
        else:
            skew_amplitude = 0
            kurt_amplitude = 0

        mean_phase = np.mean(phase)
        std_phase = np.std(phase)

        if std_phase != 0:
            skew_phase = np.mean((phase - mean_phase) ** 3) / (std_phase ** 3)
            kurt_phase = np.mean((phase - mean_phase) ** 4) / (std_phase ** 4)
        else:
            skew_phase = 0
            kurt_phase = 0

        if len(amplitude) > 1:
            cyclo_autocorr = np.abs(np.correlate(amplitude, amplitude, mode='full')[len(amplitude) // 2:]).mean()
        else:
            cyclo_autocorr = 0

        fft_magnitude_sum = np.sum(fft_magnitude)
        if fft_magnitude_sum > 0:
            normalized_fft = fft_magnitude / fft_magnitude_sum
            spectral_entropy = -np.sum(normalized_fft * np.log2(normalized_fft + 1e-12))  # add epsilon to avoid log(0)
        else:
            spectral_entropy = 0

        return [
            mean_amplitude, std_amplitude,
            mean_fft_magnitude, std_fft_magnitude,
            skew_amplitude, kurt_amplitude,
            skew_phase, kurt_phase,
            cyclo_autocorr, spectral_entropy
        ]

    # Function to calculate signal strength (simplified)
    def calculate_signal_strength(self, iq_data):
        amplitude = np.abs(iq_data)
        signal_strength_db = 10 * np.log10(np.mean(amplitude ** 2))
        return signal_strength_db

