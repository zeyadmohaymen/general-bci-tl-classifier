from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances

def tangent_space_mapping(epochs_data):
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
    ts = TangentSpace()
    tangent_space = ts.fit_transform(covs)

    return tangent_space