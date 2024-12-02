import numpy as np
import matplotlib.pyplot as plt
from sampler import RandomWalkMCMC, HamiltonianMCMC
import seaborn as sns

def test_example( alpha, beta=0.25 ):
    '''
    Test function for the unnormalized density.
    '''
    def unnorm_logdensity( x ):
        return -alpha * ( (x[0]**2 + x[1]**2) - beta ) ** 2
    def unnorm_logdensity_grad( x ):
        return -4 * alpha * ( (x[0]**2 + x[1]**2) - beta ) * x
    return unnorm_logdensity, unnorm_logdensity_grad

def simple_test():
    unnorm_logdensity = lambda x: -0.5 * (x[0]**2 + x[1]**2)
    unnorm_logdensity_grad = lambda x: -x
    return unnorm_logdensity, unnorm_logdensity_grad


if __name__ == '__main__':
    seed = 1234
    initial_condition = [0.0, 0.0]
    alpha = 10
    unnorm_logdensity, unnorm_logdensity_grad = test_example(10, beta=0.25)
    step_size = 0.1
    n_samples = 10000

    random_walk_mcmc = RandomWalkMCMC(seed, initial_condition, unnorm_logdensity, step_size)
    random_walk_samples = random_walk_mcmc.sample(n_chains=3, n_samples=n_samples)
    
    dt = 0.1
    t = .5
    hamiltonian_mcmc = HamiltonianMCMC(seed, initial_condition, unnorm_logdensity, unnorm_logdensity_grad, step_size, t, dt)
    hamiltonian_samples = hamiltonian_mcmc.sample(n_chains=3, n_samples=n_samples)


    # plot the kernel density estimate of the samples
    x = np.linspace(-2., 2., 100)
    y = np.linspace(-2., 2., 100)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    Z = np.exp(unnorm_logdensity([X,Y]))
    ax.contourf(X, Y, Z, levels=100)


    def plot_2d_kde( samples, title ):
        '''
        Plots the 2D KDE for a given array of shape (n_chains, n_samples, 2).
        
        Parameters:
        samples (np.ndarray): An array of shape (n_chains, n_samples, 2).
        '''
        n_chains, n_samples, _ = samples.shape
        plt.figure(figsize=(8, 6))
        
        #for chain_idx in range(n_chains):
        #chain_samples = samples[chain_idx]
        sns.kdeplot(
            x=samples[0,:, 0], 
            y=samples[0,:, 1], 
            fill=True, 
            alpha=0.4, 
            #label=f'Chain {chain_idx + 1}'
        )
        
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.legend()
        plt.grid()

    plot_2d_kde(random_walk_samples, 'random_walk')
    plot_2d_kde(hamiltonian_samples, 'hamiltonian')
    plt.show()



