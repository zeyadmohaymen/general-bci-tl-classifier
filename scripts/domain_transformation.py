import numpy as np
import mne
from mne import Epochs
from autoencoder import Autoencoder
from utils import reconstruct_epochs

def transform_domain(epochs: Epochs):
    """
    Transforms the domain of the input epochs to autoencoder weight vectors.

    Parameters:
    epochs (Epochs): The input epochs to be transformed.

    Returns:
    transformed_epochs (ndarray): The transformed epochs.
    """
    epoch_data = epochs.get_data()
    no_trials = epoch_data.shape[0]
    no_channels = epoch_data.shape[1]

    transformed_data = np.zeros(epoch_data.shape)

    for trial in range(no_trials):
        for channel in range(no_channels):
            channel_data = epoch_data[trial][channel]

            # Create an instance of Autoencoder for each channel data
            autoencoder = Autoencoder(epoch=channel_data)
            # Train the autoencoder
            autoencoder.train()
            # Extract optimized weights
            weights = autoencoder.get_weights()

            transformed_data[trial][channel] = weights

    transformed_epochs = reconstruct_epochs(epochs, transformed_data) #??

    return transformed_epochs