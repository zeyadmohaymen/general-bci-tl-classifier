import numpy as np
import mne
from mne import Epochs
from autoencoder import Autoencoder

def transform(epochs: Epochs):
    epoch_data = epochs.get_data()
    no_trials = epoch_data.shape[0]
    no_channels = epoch_data.shape[1]

    transformed_data = np.zeros(epoch_data.shape)

    for trial in range(no_trials):
        for channel in range(no_channels):
            channel_data = epoch_data[trial][channel]

            autoencoder = Autoencoder(epoch=channel_data)
            autoencoder.train()
            weights = autoencoder.get_weights()

            transformed_data[trial][channel] = weights

    return transformed_data