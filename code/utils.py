import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def test_example( alpha, beta=0.25, plot=False ):
    '''
    Test function for the unnormalized density.
    '''
    def unnorm_logdensity( x ):
        return -alpha * ( (x[0]**2 + x[1]**2) - beta ) ** 2
    def unnorm_logdensity_grad( x ):
        return -4 * alpha * ( (x[0]**2 + x[1]**2) - beta ) * x
    
    if plot:
        x = np.linspace(-1., 1., 100)
        y = np.linspace(-1., 1., 100)
        X, Y = np.meshgrid(x, y)
        fig, ax = plt.subplots()
        Z = np.exp(unnorm_logdensity([X,Y]))
        ax.contourf(X, Y, Z, levels=100)
        plt.savefig(f'../figures/density_alpha={alpha}.png')
    
    return unnorm_logdensity, unnorm_logdensity_grad

def simple_test():
    unnorm_logdensity = lambda x: -0.5 * (x[0]**2 + x[1]**2)
    unnorm_logdensity_grad = lambda x: -x
    return unnorm_logdensity, unnorm_logdensity_grad

def plot_2d_kde( samples1, samples2, title1, title2, figname ):
    '''
    Plots the 2D KDE for a given array of shape (n_chains, n_samples, 2).
    Parameters:
    samples (np.ndarray): An array of shape (n_chains, n_samples, 2).
    '''
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    def sub_plot(samples, title, ax):
        sns.kdeplot(
            x=samples[0,:, 0], 
            y=samples[0,:, 1], 
            fill=True, 
            alpha=0.4, 
            ax=ax,
        )
        
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    sub_plot(samples1, title1, ax[0])
    sub_plot(samples2, title2, ax[1])

    plt.tight_layout()
    plt.savefig( f'../figures/{figname}.png' )

    return

if __name__ == '__main__':
    print('Testing stuff ...')

