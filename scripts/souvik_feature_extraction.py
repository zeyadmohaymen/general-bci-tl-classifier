import numpy as np
from scipy.signal import lfilter
from statsmodels.regression.linear_model import burg
from pywt import WaveletPacket
from utils import calculate_shannon_entropy, windowing
from mne import Epochs
# from pymultifracs.mfa import mf_analysis_full


class SouvikFeatureExtractor:
    """
    The FeatureExtractor class implementing methods described in Souvik's paper.
    Paper can be found at: https://doi.org/10.1016/j.eswa.2022.118901
    Each feature vector is composed of AR coefficients and wavelet packet entropies computed from a sliding window.
    """
    def __init__(self, ar=True, wpe=True, window_size=250, ar_order=6, wavelet='haar', max_level=4):
        """
        Initialize the FeatureExtractor class.

        Parameters:
        - ar (bool): Flag indicating whether to compute AR coefficients.
        - wpe (bool): Flag indicating whether to compute wavelet packet entropies.
        - window_size (int): Size of the window for feature extraction.
        - ar_order (int): Order of the AR model.
        - wavelet (str): Name of the wavelet for wavelet packet decomposition.
        - max_level (int): Maximum level of wavelet packet decomposition.
        """
        self.ar = ar
        self.wpe = wpe
        self.window_size = window_size
        self.ar_order = ar_order
        self.wavelet = wavelet
        self.max_level = max_level

    def _compute_ar_coeffs(self, data):
        """
        Compute AR coefficients using Burg's method.

        Parameters:
        - data (np.ndarray): Input data.

        Returns:
        - ar_coefficients (np.ndarray): AR coefficients.
        """
        ar_coefficients, _ = burg(data, self.ar_order)

        return ar_coefficients

    def _compute_wavelet_packet_entropies(self, data):
        """
        Compute wavelet packet entropies using Haar wavelet packet decomposition and Shannon's Entropy.

        Parameters:
        - data (np.ndarray): Input data.

        Returns:
        - se_vector (np.ndarray): Shannon's entropy for each wavelet packet node.
        """
        wp = WaveletPacket(data, self.wavelet, self.max_level)
        nodes = [node.data for node in wp.get_level(self.max_level)]

        se_vector = []

        for coeffs in nodes:
            entropy = calculate_shannon_entropy(coeffs)
            se_vector.append(entropy)

        return np.array(se_vector)
    
    def transform(self, epoch_data: np.ndarray):
        """
        Transform epoch data into feature vectors. A sliding window is used to extract local features from each channel.

        Parameters:
        - epoch_data (np.ndarray): Input epoch data.

        Returns:
        - feature_vectors (np.ndarray): Transformed feature vectors.
        """
        no_trials = epoch_data.shape[0]
        no_channels = epoch_data.shape[1]

        feature_vectors = np.zeros(epoch_data.shape)

        for trial in range(no_trials):
            for channel in range(no_channels):
                channel_data = epoch_data[trial][channel]
                ar_coeffs = np.array([])
                se_vector = np.array([])

                if self.ar:
                    ar_coeffs = windowing(channel_data, self.window_size, self._compute_ar_coeffs)

                if self.wpe:
                    se_vector = windowing(channel_data, self.window_size, self._compute_wavelet_packet_entropies)

                feature_vector = np.concatenate((ar_coeffs, se_vector))

                feature_vectors[trial][channel] = feature_vector

        return np.array(feature_vectors)

    # def compute_multifractal_features(signal):
    #     # Compute multifractal features using MFA
    #     mfa = mf_analysis_full(signal)

    #     return singularity_spectrum, width