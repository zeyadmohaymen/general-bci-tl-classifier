from mne import Epochs
import numpy as np

def reconstruct_epochs(epochs: Epochs, new_data):
    """
    Copy the input epochs object with new data.

    Parameters:
    epochs (Epochs): The input epochs object.
    new_data (ndarray): The new data to be copied to the epochs object.

    Returns:
    new_epochs (Epochs): The new epochs object with the new data.
    """
    new_epochs = epochs.copy()
    new_epochs._data = new_data
    return new_epochs

def calculate_shannon_entropy(node_coeffs):
    # Calculate Shannon's entropy
    squared_coeffs = np.square(node_coeffs)
    total_energy = np.sum(squared_coeffs)
    coeff_probs = squared_coeffs / total_energy
    entropy = -np.sum(coeff_probs * np.log(coeff_probs))

    return entropy

def windowing(data, window_size, func):
    # Apply windowing to the data
    windowed_data = []

    for i in range(0, len(data), window_size):
        window = data[i:i + window_size]
        windowed_data.extend(func(window))

    return np.array(windowed_data)

# def apply_to_epochs(epochs, func):
#     # Apply a function to each epoch
#     data = epochs.get_data()
#     new_data = []

#     for epoch_data in data:
#         new_epoch_data = func(epoch_data)
#         new_data.append(new_epoch_data)

#     new_data = np.array(new_data)
#     new_epochs = reconstruct_epochs(epochs, new_data)

#     return new_epochs