import mne
import numpy as np
from scipy.linalg import fractional_matrix_power
from pyriemann.estimation import Covariances
import logging
logger = logging.getLogger('label-alignment')

class LabelAlignment:
    """
    A domain adaptation technique that transfers a source domain to a target domain by aligning individual source classes to corresponding target classes.

    Parameters:
        target_epochs (mne.Epochs): The target epochs data.
        concat (bool): Whether to concatenate the target epochs with the aligned source. Default is False.

    Methods:
        fit(source_data, source_events):
            Fits the source domain to the target domain by computing alignment matrices.
            Args:
                source_data (ndarray): The source epochs data.
                source_events (ndarray): The source events data.
            Returns:
                self

        transform(source_data):
            Transforms the individual source classes by aligning them with corresponding target classes.
            Args:
                source_data (ndarray): The source epochs data.
            Returns:
                aligned_epochs (ndarray): Aligned epochs.
                aligned_events (ndarray): Aligned events.

        fit_transform(source_data, source_events):
            Fits the source domain to the target domain and transforms the source data.
            Args:
                source_data (ndarray): The source epochs data.
                source_events (ndarray): The source events data.
            Returns:
                aligned_epochs (ndarray): Aligned epochs.
                aligned_events (ndarray): Aligned events.
    """

    def __init__(self, target_data, target_events, concat=True):
        """
        Initialize the LabelAlignment object.

        Parameters:
        - target_epochs (mne.Epochs): The target epochs to align the labels with.
        - concat (bool): Whether to concatenate the source and target data. Default is False.
        """
        self.source_data = None
        self.source_events = None

        self.target_data = target_data
        self.target_events = target_events

        self.concat = concat

        self.alignment_matrices = None

    def _validate(self):
        '''
        Validate source and target domains have the same number of classes.
        '''
        source_classes = np.unique(self.source_events)
        target_classes = np.unique(self.target_events)

        if len(source_classes) != len(target_classes):
            raise ValueError('Source and target domains must have the same number of classes.')

    def _segregate_epochs(self, epoch_events):
        """
        Segregates the epochs based on their labels.

        Parameters:
        epochs (numpy.ndarray): The epochs events.

        Returns:
        dict: A dictionary containing the segregated indices for each label.
        """

        # Get the unique classes
        classes = np.unique(epoch_events)

        # Create an empty dictionary to store the segregated indices
        segregated_indices = {label: [] for label in classes}

        # Add indices to segregate epochs based on their labels
        for i, label in enumerate(epoch_events):
            segregated_indices[label].append(i)

        return segregated_indices
    
    def _classes_mean_covs(self, segregated_indices, covs):
        """
        Compute the mean covariance matrices for each class.

        Parameters:
        segregated_indices (dict): A dictionary containing the segregated indices for each label.
        covs (numpy.ndarray): The covariance matrices of a domain.

        Returns:
        dict: A dictionary containing the mean covariance matrix for each class.
        """

        mean_covs = {}

        for label in segregated_indices:
            indices = segregated_indices[label]
            class_covs = covs[indices]
            mean_cov = np.mean(class_covs, axis=0)
            mean_covs[label] = mean_cov

        return mean_covs
    
    def _compute_alignment_matrices(self, source_mean_covs, target_mean_covs):
        """
        Compute alignment matrix for each source class to its corresponding target class.

        Args:
            source_mean_covs (dict): A dictionary containing source classes and their mean covariance matrices.
            target_mean_covs (dict): A dictionary containing target classes and their mean covariance matrices.

        Returns:
            dict: A dictionary containing alignment matrices for each source label.
        """

        alignment_matrices = {}

        for source_label, target_label in zip(source_mean_covs, target_mean_covs):
            # Exctract corresponding source and target class mean covariance matrices
            source_class_mean_cov = source_mean_covs[source_label]
            target_class_mean_cov = target_mean_covs[target_label]

            # Compute the square root of the target class mean covariance matrix
            target_class_mean_cov_sqrt = fractional_matrix_power(target_class_mean_cov, 0.5)
            # Compute the inverse square root of the source class mean covariance matrix
            source_class_mean_cov_inv_sqrt = fractional_matrix_power(source_class_mean_cov, -0.5)

            # Compute the alignment matrix
            alignment_matrix = np.dot(target_class_mean_cov_sqrt, source_class_mean_cov_inv_sqrt)

            # Store the alignment matrix for the corresponding source label
            alignment_matrices[source_label] = alignment_matrix

        return alignment_matrices
    
    def fit(self, source_data, source_events):
        """
        Fit the source domain to the target domain.

        Args:
            source_data (numpy.ndarray): The source data.
            source_events (numpy.ndarray): The source events.

        Returns:
            self: The fitted LabelAlignment instance.
        """
        self.source_data = source_data
        self.source_events = source_events

        self._validate()

        # Segregate epochs based on labels for source and target domains
        source_segregated_indices = self._segregate_epochs(self.source_events)
        target_segregated_indices = self._segregate_epochs(self.target_events)

        # Compute SPD covariance matrices for source and target domains
        source_covs = Covariances(estimator='oas').fit_transform(self.source_data)
        target_covs = Covariances(estimator='oas').fit_transform(self.target_data)

        # Compute the mean covariance matrices for each class in source and target domains
        source_mean_covs = self._classes_mean_covs(source_segregated_indices, source_covs)
        target_mean_covs = self._classes_mean_covs(target_segregated_indices, target_covs)

        # Compute the alignment matrices for each source class to its corresponding target class
        self.alignment_matrices = self._compute_alignment_matrices(source_mean_covs, target_mean_covs)

        return self

    def transform(self, source_data):
        """
        Transforms the source data by aligning the epochs using the alignment matrices.

        Args:
            source_data (list): The source data containing the epochs to be aligned.

        Returns:
            tuple: A tuple containing the aligned epochs and the corresponding events.
        """
        # Align the source epochs using the alignment matrices
        aligned_source_epochs = []

        for trial, label in zip(source_data, self.source_events):
            alignment_matrix = self.alignment_matrices[label]
            aligned_trial = np.dot(alignment_matrix, trial)
            aligned_source_epochs.append(aligned_trial)

        aligned_epochs = np.array(aligned_source_epochs)
        aligned_events = self.source_events

        if self.concat:
            aligned_epochs = np.concatenate([aligned_epochs, self.target_data])
            aligned_events = np.concatenate([self.source_events, self.target_events])

        logger.info("Source data aligned to target data.")

        return aligned_epochs, aligned_events
    
    def fit_transform(self, source_data, source_events):
        """
        Fits the model to the source data and events, and then transforms the source data.

        Parameters:
            source_data (array-like): The input source data.
            source_events (array-like): The input source events.

        Returns:
            transformed_data (tuple): The transformed source data and events.
        """
        # print("Aligning source data to target data...")
        self.fit(source_data, source_events)
        return self.transform(source_data)


# TODO: Maybe we can refactor the code to use the LabelData class instead of using dictionaries.
# class LabelData:
#     """
#     A class representing individual labels and their associated data.
#     """

#     def __init__(self, label, data):
#         self.label = label
#         self.data = data
#         self.covs = Covariances(estimator='oas').transform(data)
#         self.mean_cov = np.mean(self.covs, axis=0)
#         self.alignment_matrix = None