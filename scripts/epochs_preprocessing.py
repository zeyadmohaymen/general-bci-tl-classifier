from sklearn.base import BaseEstimator, TransformerMixin
from mne import Epochs, merge_events
import numpy as np
from scripts.utils import reconstruct_epochs

class EventsEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes events in epochs data based on specified counter class.

    Parameters:
    -----------
    counter_class : str, optional
        The counter class to use for encoding. Must be either 'rest' or 'other'.

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

    def __init__(self, counter_class):
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
        # Copy the epochs object to avoid modifying the original data
        epochs = epochs.copy()

        # Get the event IDs from the epochs object
        event_ids = epochs.event_id

        # Check if the counter class is 'rest'
        if self.counter_class == "rest":
            # Select only the 'rest' and 'feet' epochs
            selected_epochs = epochs["rest", "feet"]
            events = selected_epochs.events[:, -1]
            
            # Get respective "rest" and "feet" ids
            rest_id = event_ids["rest"]
            feet_id = event_ids["feet"]

            # Extract their indices
            rest_indices = np.where(events == rest_id)[0]
            feet_indices = np.where(events == feet_id)[0]

            # Set "rest" to 0 and "feet" to 1
            transformed_events = np.zeros_like(events)
            transformed_events[rest_indices] = 0
            transformed_events[feet_indices] = 1

            # Update the events in the epochs object with the transformed events
            selected_epochs.events[:, -1] = transformed_events

            # Update the event IDs in the selected epochs object
            selected_epochs.event_id = {"rest": 0, "feet": 1}

            print("Epochs after event encoding [rest vs feet]:")
            print(selected_epochs)

            return selected_epochs
        
        # Check if the counter class is 'other' 
        else:
            events = epochs.events[:, -1]
            feet_id = event_ids["feet"]

            # Extract indices if "feet" vs everything else
            feet_indices = np.where(events == feet_id)[0]
            not_feet_indices = np.where(events != feet_id)[0]

            # Set "not feet" to 0 and "feet" to 1
            transformed_events = np.zeros_like(events)
            transformed_events[not_feet_indices] = 0
            transformed_events[feet_indices] = 1

            # Update the events in the epochs object with the transformed events
            epochs.events[:, -1] = transformed_events

            # Update the event IDs in the epochs object
            epochs.event_id = {"not feet": 0, "feet": 1}

            print("Epochs after event encoding [not feet vs feet]:")
            print(epochs)

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
        return epochs.copy().equalize_event_counts()[0]
    
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
        print("Cropped Epochs to:", self.tmin, "s -", self.tmax, "s")
        return epochs.copy().crop(self.tmin, self.tmax)
    
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
    
    def transform(self, epochs: Epochs):
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
        # Copy the epochs object to avoid modifying the original data
        epochs = epochs.copy()

        # Check if the window size is greater than the epoch length
        epoch_length = epochs.tmax - epochs.tmin
        if self.window_size >= epoch_length:
            return epochs
        
        # Get the sampling frequency of the epochs data
        sfreq = epochs.info['sfreq']
        
        # Calculate the number of samples in each window
        n_samples = int(self.window_size * sfreq)
        
        # Calculate the number of samples to step between windows
        step_samples = int(n_samples * (1 - self.overlap))
        
        # Get the original data from the epochs object
        data = epochs.get_data(copy=False)
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

        print("Applied Epochs Segmentation with", self.window_size, "s and", self.overlap*100, "% overlap")
        print(new_epochs)
        
        # Return the new epochs object
        return new_epochs
