from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances

class TangentSpaceMapping:

    def __init__(self):
        self.ts = TangentSpace()

    def transform(self, epochs_data):
        """
        Compute the tangent space mapping for the given epochs data.

        Parameters:
        epochs_data (array-like): The input epochs data.

        Returns:
        array-like: The tangent space mapping of the input epochs data.
        """
        
        # Compute SPD covariance matrices
        covs = Covariances(estimator='oas').fit_transform(epochs_data)

        # Compute tangent space mapping
        tangent_space = self.ts.fit_transform(covs)

        return tangent_space
    
    def fit_transform(self, epochs_data, _=None):
        """
        Fit the given epochs data and compute the tangent space mapping.

        Parameters:
        epochs_data (array-like): The input epochs data.

        Returns:
        array-like: The tangent space mapping of the input epochs data.
        """
        
        return self.transform(epochs_data)