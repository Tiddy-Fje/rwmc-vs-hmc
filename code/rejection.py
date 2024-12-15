import numpy as np
from utils import test_example

def get_data_params( X, y, sigmas ):
    '''
    Returns the mean, covariance matrix and C constant associated to the proposal (multivariate gaussian) distribution.

    Parameters
    ----------
    X : np.ndarray
        Design matrix.
    y : np.ndarray 
        Response vector.
    sigmas : np.ndarray
        Standard deviations of the normal distribution.
    '''
    b = X.T @ (y - 1)
    Sigma = np.diag( sigmas**2 )
    mu = Sigma @ b

    p = X.shape[1]
    var_inv = 1 / (sigmas ** 2)
    C2 = (2*np.pi) ** p * np.prod( np.diag(Sigma) ) * np.exp((mu *  var_inv).T @ mu)
    return mu, Sigma, np.sqrt(C2)


def rejection_sampling( X, y, sigmas, alpha, n_samples ):
    '''
    Sample from the posterior distribution using rejection sampling. The proposal is taken to be a multivariate normal distribution.

    Parameters
    ----------
    X : np.ndarray
        Design matrix.
    y : np.ndarray
        Response vector.
    sigmas : np.ndarray
        Standard deviations of the normal distribution.
    n_samples : int
        Number of samples to generate.
    unnorm_logdensity : function   
        Function that returns the log of the unnormalized density.
    '''
    mu, Sigma, C = get_data_params( X, y, sigmas )

    samples = []
    len_samples = 0
    unnorm_logdensity, _ = test_example(alpha)
    while len_samples < n_samples:
        z = np.random.multivariate_normal( mu, Sigma )
        u = np.random.uniform(0, 1)
        proposal_logdensity = -0.5 * (z / np.diag(Sigma)).T @ z
        if u < np.exp(unnorm_logdensity(z) - np.log(C) - proposal_logdensity):
            samples.append(z)
            len_samples += 1
    return np.array(samples)