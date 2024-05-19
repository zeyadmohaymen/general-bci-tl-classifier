import mne
from mne import pick_types, Epochs
from mne.io import Raw
from mne.preprocessing import ICA
from mne_icalabel import label_components


class Preprocessing:
    """
    The Preprocessing class provides methods to preprocess Raw MNE object and segment it into MNE Epochs.
    """
    def __init__(self, raw: Raw):
        """
        Initialize the Preprocessing object.
        """
        self.raw = raw
        self.processed_raw = None
        self.processed_epochs = None

    def _filter_raw(self, l_freq=8.0, h_freq=35.0):
        """
        Apply bandpass filter to raw data.

        Parameters:
        - l_freq: The lower frequency of the bandpass filter (default: 8.0).
        - h_freq: The higher frequency of the bandpass filter (default: 35.0).
        """
        self.processed_raw = self.raw.copy().filter(l_freq, h_freq, fir_design='firwin')

    def _remove_artifacts(self):
        """
        Automatically removes EOG, ECG, and other artifacts from the raw data.
        """
        # Apply common average reference
        self.processed_raw = self.processed_raw.set_eeg_reference('average')

        # Create ICA object
        ica = ICA(n_components=15, random_state=97, max_iter="auto", method="infomax", fit_params=dict(extended=True)) 
        ica.fit(self.processed_raw)

        # Extract labels
        ica_labels = label_components(self.processed_raw, ica, method='iclabel')
        labels = ica_labels['labels']

        exclude_idx = [
            idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
        ]

        # Reconstruct clean raw data
        self.processed_raw = ica.apply(self.processed_raw, exclude=exclude_idx)
    
    def _select_channels(self, channels):
        """
        Select specific channels from raw data.
        """
        selected_channels = pick_types(self.raw.info, include=channels, exclude="bads")
        return selected_channels
    
    def preprocess_raw(self):
        """
        Preprocess the raw data.

        Parameters:
        - event_ids: A dictionary mapping event names to event IDs.
        """
        self._filter_raw()
        self._remove_artifacts()

    def segment_into_epochs(self, event_ids, tmin=-1, tmax=4, channels=["C3", "C4", "P3", "P4", "T3", "T4", "T5", "T6"], baseline=None):
        """
        Segment the preprocessed data into epochs.

        Parameters:
        - event_ids: A dictionary mapping event names to event IDs.
        - tmin: The start time of each epoch in seconds (default: -1).
        - tmax: The end time of each epoch in seconds (default: 4).
        - channels: A list of channel names to include in the epochs (default: ["C3", "C4", "P3", "P4", "T3", "T4", "T5", "T6"]).
        - baseline: The time interval to use for baseline correction (default: None).

        Returns:
        The segmented epochs.
        """
        # Specify channels
        selected_channels = self._select_channels(channels)

        # Segment raw data into epochs
        self.processed_epochs = Epochs(self.processed_raw, event_id=event_ids, tmin=tmin, tmax=tmax, picks=selected_channels, baseline=baseline, preload=True)

        return self.processed_epochs