from mne import Epochs, concatenate_raws
import numpy as np

def reconstruct_epochs(epochs: Epochs, new_data, new_events=None):
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
    if new_events is not None:
        new_epochs.events = new_events
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

def load_moabb_data(dataset):
    # Load data from MOABB dataset
    # data = {'subject_id' :
        #     {'session_id':
        #         {'run_id': run}
        #     }
        # }
    sessions = dataset.get_data()
    raws = []
    for subject_id, sessions in sessions.items():
        for session_id, runs in sessions.items():
            concat = concatenate_raws([run for run in runs.values()])
            raws.append(concat)
    return raws

def load_single_moabb_subject(dataset, subject_id):
    # Load data from MOABB dataset for a single subject
    sessions = dataset.get_data(subjects=[subject_id])
    raws = []
    for session in sessions[subject_id].values():
        for run in session.values():
            raws.append(run)

    return concatenate_raws(raws)