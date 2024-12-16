import numpy as np
from utils import test_example, preprocess_data, logposterior

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
    b = X.T @ (y - 0)
    Sigma = np.diag( sigmas**2 )
    mu = Sigma @ b

    p = X.shape[1]
    n = X.shape[0]
    var_inv = 1 / (sigmas ** 2)
    logC = 0.5 * ( p * np.log(2*np.pi) + np.sum( np.diag(Sigma) ) + (mu *  var_inv).T @ mu ) + n * np.log(2)
    return mu, Sigma, logC


def rejection_sampling( X, y, sigmas, unnorm_logdensity, n_samples ):
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
    mu, Sigma, logC = get_data_params( X, y, sigmas )

    samples = []
    len_samples = 0
    while len_samples < n_samples:
        z = np.random.multivariate_normal( mu, Sigma )
        u = np.random.uniform(0, 1)
        proposal_logdensity = -0.5 * (z / np.diag(Sigma)).T @ z
        print(logC + proposal_logdensity, unnorm_logdensity(z))
        if u < np.exp(unnorm_logdensity(z) - logC - proposal_logdensity):
            samples.append(z)
            len_samples += 1
    return np.array(samples)


X, y, sigmas = preprocess_data()

mu, Sigma, C = get_data_params( X, y, sigmas )
#x = np.mean( np.log( 1 / ( 1 + np.exp(-X@mu) ) ) )
#print(mu, '\n',Sigma, C)

logdensity = logposterior(X, y, sigmas)
samples = rejection_sampling( X, y, sigmas, logdensity, 10 )
print(np.mean(samples))


'''

mu, Sigma, C = get_data_params( X, y, sigmas )
x = np.mean( np.log( 1 / ( 1 + np.exp(-X@mu) ) ) )
print(x)

mu, Sigma, C = get_data_params( X, y, sigmas )
print(mu, '\n',Sigma, C)

samples = np.random.multivariate_normal( mu, Sigma )
'''
