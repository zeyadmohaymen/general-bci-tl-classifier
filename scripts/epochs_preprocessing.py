from sklearn.base import BaseEstimator, TransformerMixin
from mne import Epochs
import numpy as np
from scripts.utils import reconstruct_epochs

class EventsEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes events in epochs data based on specified counter class.

    Parameters:
    -----------
    counter_class : str, optional
        The counter class to use for encoding. Must be either 'rest' or 'other'.
        Defaults to 'rest'.

    Attributes:
    -----------
    counter_class : str
        The counter class used for encoding.

    Methods:
    --------
    fit(X, y=None)
        Fit the encoder to the data.

    transform(epochs)
        Transform the epochs data by encoding the events.

    Returns:
    --------
    epochs : Epochs
        The transformed epochs data with encoded events.
    """

    def __init__(self, counter_class="rest"):
        if counter_class not in ["rest", "other"]:
            raise ValueError("counter_class must be either 'rest' or 'other'")
        self.counter_class = counter_class

    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs: Epochs):
        """
        Transform the epochs data by encoding the events.

        Parameters:
        -----------
        epochs : Epochs
            The input epochs data.

        Returns:
        --------
        epochs : Epochs
            The transformed epochs data with encoded events.
        """
        # Get the events from the epochs data
        events = epochs.events[:, -1]
        
        # Initialize an array to store the transformed events
        transformed_events = np.zeros_like(events)
        
        # Get the event IDs from the epochs object
        event_ids = epochs.event_id

        # Get the ID and indices of the 'feet' event
        feet_id = event_ids["feet"]
        feet_indices = np.where(events == feet_id)[0]
        
        # Set the transformed events to 1 for 'feet' events
        transformed_events[feet_indices] = 1

        # Check if the counter class is 'rest'
        if self.counter_class == "rest":
            # Get the ID and indices of the 'rest' event
            rest_id = event_ids["rest"]
            rest_indices = np.where(events == rest_id)[0]
            
            # Set the transformed events to 0 for 'rest' events
            transformed_events[rest_indices] = 0
        else:
            # Get the indices of events other than 'feet'
            other_indices = np.where(events != feet_id)[0]
            
            # Set the transformed events to 0 for other events
            transformed_events[other_indices] = 0

        # Update the events in the epochs object with the transformed events
        epochs.events[:, -1] = transformed_events
        
        # Return the modified epochs object
        return epochs
    
class EventsEqualizer(BaseEstimator, TransformerMixin):
    """
    A transformer class to equalize event counts in epochs data.

    Parameters:
    -----------
    None

    Returns:
    --------
    epochs : Epochs
        The input epochs data with equalized event counts.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs: Epochs):
        return epochs.equalize_event_counts()
    
class Resampler(BaseEstimator, TransformerMixin):
    """
    A class for resampling epochs data.

    Parameters:
    -----------
    sfreq : int, optional
        The desired sampling frequency for resampling the epochs data. Default is 160.

    Methods:
    --------
    fit(X, y=None)
        Fit the resampler to the data.

    transform(epochs)
        Resample the input epochs data.

    Returns:
    --------
    resampled_epochs : Epochs
        The resampled epochs data.
    """

    def __init__(self, sfreq=160):
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs: Epochs):
        return epochs.resample(self.sfreq)
    
class Cropper(BaseEstimator, TransformerMixin):
    """
    A transformer class for cropping epochs.

    Parameters:
    -----------
    tmin : float, optional
        The start time of the cropping window in seconds. Default is 0.5.
    tmax : float, optional
        The end time of the cropping window in seconds. Default is 3.5.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the data.

    transform(epochs: Epochs)
        Crop the input epochs using the specified time window.

    Returns:
    --------
    cropped_epochs : Epochs
        The cropped epochs.
    """

    def __init__(self, tmin=0.5, tmax=3.5):
        self.tmin = tmin
        self.tmax = tmax

    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs: Epochs):
        return epochs.crop(self.tmin, self.tmax)
    
class EpochsSegmenter(BaseEstimator, TransformerMixin):
    """
    A transformer class for segmenting epochs into smaller windows.

    Parameters:
    -----------
    window_size : float, optional
        The size of each window in seconds. Default is 1 second.
    overlap : float, optional
        The overlap between consecutive windows as a fraction of the window size.
        Default is 0.5 (50% overlap).

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the data.

    transform(epochs)
        Transform the epochs by segmenting them into smaller windows.

    Returns:
    --------
    new_epochs : mne.Epochs
        The segmented epochs object.

    """

    def __init__(self, window_size=1, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X, y=None):
        return self
    
    def transform(self, epochs):
        """
        Transform the epochs by segmenting them into smaller windows.

        Parameters:
        -----------
        epochs : mne.Epochs
            The input epochs object.

        Returns:
        --------
        new_epochs : mne.Epochs
            The segmented epochs object.

        """
        # Get the sampling frequency of the epochs data
        sfreq = epochs.info['sfreq']
        
        # Calculate the number of samples in each window
        n_samples = int(self.window_size * sfreq)
        
        # Calculate the number of samples to step between windows
        step_samples = int(n_samples * (1 - self.overlap))
        
        # Get the original data from the epochs object
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        # Calculate the number of windows per epoch
        n_windows = (n_times - n_samples) // step_samples + 1
        
        # Initialize lists to store the new data and events
        new_data = []
        new_events = []
        
        # Iterate over each epoch
        for epoch_idx in range(n_epochs):
            # Iterate over each window
            for window_idx in range(n_windows):
                # Calculate the start and stop indices of the window
                start = window_idx * step_samples
                stop = start + n_samples
                
                # Append the data and events of the window to the respective lists
                new_data.append(data[epoch_idx, :, start:stop])
                new_events.append([epochs.events[epoch_idx, 0] + start, 0, epochs.events[epoch_idx, 2]])
        
        # Convert the lists to numpy arrays
        new_data = np.array(new_data)
        new_events = np.array(new_events)
        
        # Create a new epochs object with the segmented data and events
        new_epochs = reconstruct_epochs(epochs, new_data, new_events)
        
        # Return the new epochs object
        return new_epochs
