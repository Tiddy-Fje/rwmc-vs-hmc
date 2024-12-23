import numpy as np
from utils import test_example
import argparse
from matplotlib import pyplot as plt
from comparison import sample_from_info, test_example, get_RWMC_nsamples_from_HMC
from similarity import similarity
import yaml

def main(config_file, ax=None):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file) 
    unnorm_logdensity, unnorm_logdensity_grad = test_example(config['General']['alpha'])
    
    n_samples = np.array( config['General']['n_samples'] )
    config['General']['n_samples'] = n_samples[-1]

    samples = sample_from_info( config['HMC'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
    config['General']['n_samples'] = get_RWMC_nsamples_from_HMC( n_samples[-1], config['HMC'] )
    rwmc_samples = sample_from_info( config['RWMC'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )

    sim = np.zeros(n_samples.shape)
    sim_std = np.zeros(n_samples.shape)
    rwmc_sim = np.zeros(n_samples.shape)
    rwmc_sim_std = np.zeros(n_samples.shape)
    for i,n in enumerate(n_samples):
        sim[i], sim_std[i] = similarity( samples[:,:n,:], unnorm_logdensity, step_x=0.025, step_y=0.025, unorm_log=True)
        rwmc_n = get_RWMC_nsamples_from_HMC( n, config['HMC'] )
        rwmc_sim[i], rwmc_sim_std[i] = similarity( rwmc_samples[:,:rwmc_n,:], unnorm_logdensity, step_x=0.025, step_y=0.025, unorm_log=True)

    # plot the samples
    flag = False
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        flag = True
    
    ax.errorbar( n_samples, sim, yerr=2*sim_std, fmt='s', linestyle='--', color='tab:blue', label='HMC' )
    ax.errorbar( n_samples, rwmc_sim, yerr=2*rwmc_sim_std, fmt='s', linestyle='--', color='tab:red', label='RWMC' )
    ax.set_xlabel('N samples')
    ax.set_ylabel('Similarity')
    ax.legend()

    if flag:
        filename = f'../figures/{config["General"]["fig_name"]}_alpha={config["General"]["alpha"]}.png'
        plt.savefig( filename )

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, help='Path to config file', required=True)
    
    args = parser.parse_args()
    main(args.config_file) # args.config_file is a filename 




'''
def get_data_params( X, y, sigmas ):
    Returns the mean, covariance matrix and C constant associated to the proposal (multivariate gaussian) distribution.

    Parameters
    ----------
    X : np.ndarray
        Design matrix.
    y : np.ndarray 
        Response vector.
    sigmas : np.ndarray
        Standard deviations of the normal distribution.
    b = X.T @ (y - 0)
    Sigma = np.diag( sigmas**2 )
    mu = Sigma @ b

    p = X.shape[1]
    n = X.shape[0]
    var_inv = 1 / (sigmas ** 2)
    logC = 0.5 * ( p * np.log(2*np.pi) + np.sum( np.diag(Sigma) ) + (mu *  var_inv).T @ mu ) + n * np.log(2)
    return mu, Sigma, logC


def rejection_sampling( X, y, sigmas, unnorm_logdensity, n_samples ):
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


mu, Sigma, C = get_data_params( X, y, sigmas )
x = np.mean( np.log( 1 / ( 1 + np.exp(-X@mu) ) ) )
print(x)

mu, Sigma, C = get_data_params( X, y, sigmas )
print(mu, '\n',Sigma, C)

samples = np.random.multivariate_normal( mu, Sigma )
'''
