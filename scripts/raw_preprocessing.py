import mne
from mne import pick_types, Epochs
from mne.io import Raw
from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.base import BaseEstimator, TransformerMixin

class FilterRaw(BaseEstimator, TransformerMixin):
    """
    A transformer class for applying bandpass filter to raw data.

    Parameters:
    -----------
    l_freq : float, optional
        The lower frequency of the bandpass filter. Default is 8.0 Hz.
    h_freq : float, optional
        The higher frequency of the bandpass filter. Default is 35.0 Hz.

    Methods:
    transform(raw)
        Transform the raw data by applying bandpass filter.

    Returns:
    --------
    new_raw : mne.io.Raw
        The filtered raw data.
    """

    def __init__(self, l_freq=8.0, h_freq=35.0):
        self.l_freq = l_freq
        self.h_freq = h_freq

    def fit(self, X, y=None):
        return self
    
    def transform(self, raw: Raw):
        print("Applying bandpass filter to raw data...")
        new_raw = raw.copy().filter(self.l_freq, self.h_freq, fir_design='firwin')
        return new_raw
    
class RemoveArtifacts(BaseEstimator, TransformerMixin):
    """
    A transformer class for removing artifacts from raw data.

    Methods:
    --------
    transform(raw)
        Transform the raw data by removing artifacts.

    Returns:
    --------
    new_raw : mne.io.Raw
        The raw data with artifacts removed.
    """

    def fit(self, X, y=None):
        return self
    
    def transform(self, raw: Raw):
        print("Removing artifacts from raw data...")
        # Apply common average reference
        raw = raw.set_eeg_reference('average')

        # Create ICA object
        ica = ICA(n_components=15, random_state=97, max_iter="auto", method="infomax", fit_params=dict(extended=True)) 
        ica.fit(raw)

        # Extract labels
        ica_labels = label_components(raw, ica, method='iclabel')
        labels = ica_labels['labels']

        exclude_idx = [
            idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
        ]

        # Reconstruct clean raw data
        new_raw = ica.apply(raw, exclude=exclude_idx)
        return new_raw
    
class SelectChannels(BaseEstimator, TransformerMixin):
    """
    A transformer class for selecting specific channels from raw data.

    Parameters:
    -----------
    channels : list of str
        A list of channel names to include in the raw data.

    Methods:
    --------
    transform(raw)
        Transform the raw data by selecting specific channels.

    Returns:
    --------
    new_raw : mne.io.Raw
        The raw data with specific channels selected.
    """

    def __init__(self, channels):
        self.channels = channels

    def fit(self, X, y=None):
        return self
    
    def transform(self, raw: Raw):
        print("Selecting specific channels from raw data...")
        selected_channels = pick_types(raw.info, include=self.channels, exclude="bads")
        new_raw = raw.copy().pick_channels(selected_channels)
        return new_raw
    
class Epochify(BaseEstimator, TransformerMixin):
    """
    A transformer class for segmenting raw data into epochs.

    Parameters:
    -----------
    event_ids : dict
        A dictionary mapping event names to event IDs.
    tmin : float, optional
        The start time of each epoch in seconds. Default is -1.
    tmax : float, optional
        The end time of each epoch in seconds. Default is 4.
    baseline : tuple of float, optional
        The time interval to use for baseline correction. Default is None.

    Methods:
    --------
    transform(raw)
        Transform the raw data by segmenting it into epochs.

    Returns:
    --------
    epochs : mne.Epochs
        The segmented epochs.
    """

    def __init__(self, event_ids, channels, tmin=-1, tmax=4, baseline=None):
        self.event_ids = event_ids
        self.channels = channels
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline

    def fit(self, X, y=None):
        return self
    
    def transform(self, raw: Raw):
        print("Segmenting raw data into epochs...")
        selected_channels = pick_types(raw.info, include=self.channels, exclude="bads")
        epochs = Epochs(raw, event_id=self.event_ids, tmin=self.tmin, tmax=self.tmax, picks=selected_channels, baseline=self.baseline, preload=True)
        return epochs
    
class Resampler(BaseEstimator, TransformerMixin):
    """
    A class for resampling epochs data.

    Parameters:
    -----------
    sfreq : int, optional
        The desired sampling frequency for resampling the epochs data.

    Methods:
    --------
    transform(epochs)
        Resample the input epochs data.

    Returns:
    --------
    resampled_epochs : Epochs
        The resampled epochs data.
    """

    def __init__(self, sfreq):
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs: Epochs):
        return epochs.copy().resample(self.sfreq)


# class Preprocessing:
#     """
#     The Preprocessing class provides methods to preprocess Raw MNE object and segment it into MNE Epochs.
#     """
#     def __init__(self, raw: Raw):
#         """
#         Initialize the Preprocessing object.
#         """
#         self.raw = raw
#         self.processed_raw = None
#         self.processed_epochs = None

#     def _filter_raw(self, l_freq=8.0, h_freq=35.0):
#         """
#         Apply bandpass filter to raw data.

#         Parameters:
#         - l_freq: The lower frequency of the bandpass filter (default: 8.0).
#         - h_freq: The higher frequency of the bandpass filter (default: 35.0).
#         """
#         self.processed_raw = self.raw.copy().filter(l_freq, h_freq, fir_design='firwin')

#     def _remove_artifacts(self):
#         """
#         Automatically removes EOG, ECG, and other artifacts from the raw data.
#         """
#         # Apply common average reference
#         self.processed_raw = self.processed_raw.set_eeg_reference('average')

#         # Create ICA object
#         ica = ICA(n_components=15, random_state=97, max_iter="auto", method="infomax", fit_params=dict(extended=True)) 
#         ica.fit(self.processed_raw)

#         # Extract labels
#         ica_labels = label_components(self.processed_raw, ica, method='iclabel')
#         labels = ica_labels['labels']

#         exclude_idx = [
#             idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
#         ]

#         # Reconstruct clean raw data
#         self.processed_raw = ica.apply(self.processed_raw, exclude=exclude_idx)
    
#     def _select_channels(self, channels):
#         """
#         Select specific channels from raw data.
#         """
#         selected_channels = pick_types(self.raw.info, include=channels, exclude="bads")
#         return selected_channels
    
#     def preprocess_raw(self):
#         """
#         Preprocess the raw data.

#         Parameters:
#         - event_ids: A dictionary mapping event names to event IDs.
#         """
#         self._filter_raw()
#         self._remove_artifacts()

#     def segment_into_epochs(self, event_ids, tmin=-1, tmax=4, channels=["C3", "C4", "P3", "P4", "T3", "T4", "T5", "T6"], baseline=None):
#         """
#         Segment the preprocessed data into epochs.

#         Parameters:
#         - event_ids: A dictionary mapping event names to event IDs.
#         - tmin: The start time of each epoch in seconds (default: -1).
#         - tmax: The end time of each epoch in seconds (default: 4).
#         - channels: A list of channel names to include in the epochs (default: ["C3", "C4", "P3", "P4", "T3", "T4", "T5", "T6"]).
#         - baseline: The time interval to use for baseline correction (default: None).

#         Returns:
#         The segmented epochs.
#         """
#         # Specify channels
#         selected_channels = self._select_channels(channels)

#         # Segment raw data into epochs
#         self.processed_epochs = Epochs(self.processed_raw, event_id=event_ids, tmin=tmin, tmax=tmax, picks=selected_channels, baseline=baseline, preload=True)

#         return self.processed_epochs