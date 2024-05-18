import mne
import numpy as np
from scipy.linalg import fractional_matrix_power
from pyriemann.estimation import Covariances

class LabelAlignment:
    """
    A domain adaptation technique that transfers a source domain to a target domain. This is done by aligning 
    individual source classes to corresponding target classes.
    """

    def __init__(self, source_epochs, target_epochs):
        self.source_epochs = source_epochs
        self.target_epochs = target_epochs

        self._validate()

    def _validate(self):
        '''
        Validate source and target domains have the same number of classes.
        '''
        source_classes = np.unique(self.source_epochs.events[:, -1])
        target_classes = np.unique(self.target_epochs.events[:, -1])

        if len(source_classes) != len(target_classes):
            raise ValueError('Source and target domains must have the same number of classes.')

    def _segregate_epochs(self, epochs):
        """
        Segregates the epochs based on their labels.

        Parameters:
        epochs (numpy.ndarray): The epochs data.

        Returns:
        dict: A dictionary containing the segregated indices for each label.
        """

        # Get the labels from the events
        labels = epochs.events[:, -1]

        # Get the unique classes
        classes = np.unique(labels)

        # Create an empty dictionary to store the segregated indices
        segregated_indices = {label: [] for label in classes}

        # Add indices to segregate epochs based on their labels
        for i, label in enumerate(labels):
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

    def transform(self):
        """
        Transforms the individual source class by aligning them with corresponding target classes.

        Returns:
            aligned_source_epochs (ndarray): Aligned source epochs.
        """
        source_data = self.source_epochs.get_data(copy=True)
        source_labels = self.source_epochs.events[:, -1]
        target_data = self.target_epochs.get_data(copy=True)

        # Segregate epochs based on labels for source and target domains
        source_segregated_indices = self._segregate_epochs(self.source_epochs)
        target_segregated_indices = self._segregate_epochs(self.target_epochs)

        # Compute SPD covariance matrices for source and target domains
        source_covs = Covariances(estimator='oas').fit_transform(source_data)
        target_covs = Covariances(estimator='oas').fit_transform(target_data)

        # Compute the mean covariance matrices for each class in source and target domains
        source_mean_covs = self._classes_mean_covs(source_segregated_indices, source_covs)
        target_mean_covs = self._classes_mean_covs(target_segregated_indices, target_covs)

        # Compute the alignment matrices for each source class to its corresponding target class
        alignment_matrices = self._compute_alignment_matrices(source_mean_covs, target_mean_covs)

        # Align the source epochs using the alignment matrices
        aligned_source_epochs = []

        for trial, label in zip(source_data, source_labels):
            alignment_matrix = alignment_matrices[label]
            aligned_trial = np.dot(alignment_matrix, trial)
            aligned_source_epochs.append(aligned_trial)

        aligned_source_epochs = np.array(aligned_source_epochs)

        return aligned_source_epochs


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