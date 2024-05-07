import numpy as np
from scipy.signal import lfilter
from statsmodels.regression.linear_model import burg
from pywt import WaveletPacket
from utils import calculate_shannon_entropy, windowing
from mne import Epochs
# from pymultifracs.mfa import mf_analysis_full


class FeatureExtractor:
    def __init__(self, ar=True, wpe=True, window_size=250, ar_order=6, wavelet='haar', max_level=4):
        self.ar = ar
        self.wpe = wpe
        self.window_size = window_size
        self.ar_order = ar_order
        self.wavelet = wavelet
        self.max_level = max_level

    def _compute_ar_coeffs(self, data):
        # Apply Burg's method to estimate AR coefficients
        ar_coefficients, _ = burg(data, self.ar_order)

        return ar_coefficients

    def _compute_wavelet_packet_entropies(self, data):
        # Apply Haar wavelet packet decomposition up to level 4
        wp = WaveletPacket(data, self.wavelet, self.max_level)
        nodes = [node.data for node in wp.get_level(self.max_level)]

        se_vector = []

        # Calculate Shannon's entropy for each node
        for coeffs in nodes:
            entropy = calculate_shannon_entropy(coeffs)
            se_vector.append(entropy)

        return np.array(se_vector)
    
    def transform(self, epoch_data: np.ndarray):
        no_trials = epoch_data.shape[0]
        no_channels = epoch_data.shape[1]

        feature_vectors = np.zeros(epoch_data.shape)

        for trial in range(no_trials):
            for channel in range(no_channels):
                channel_data = epoch_data[trial][channel]
                ar_coeffs = np.array([])
                se_vector = np.array([])

                if self.ar:
                    # Compute AR coefficients
                    ar_coeffs = windowing(channel_data, self.window_size, self._compute_ar_coeffs)

                if self.wpe:
                    # Compute wavelet packet entropies
                    se_vector = windowing(channel_data, self.window_size, self._compute_wavelet_packet_entropies)

                # Concatenate AR coefficients and wavelet packet entropies
                feature_vector = np.concatenate((ar_coeffs, se_vector))

                feature_vectors[trial][channel] = feature_vector

        return np.array(feature_vectors)

    # def compute_multifractal_features(signal):
    #     # Compute multifractal features using MFA
    #     mfa = mf_analysis_full(signal)

    #     return singularity_spectrum, width