import yaml
import numpy as np
import argparse
from utils import test_example
from sampler import RandomWalkMCMC, HamiltonianMCMC
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from similarity import similarity

# fix rcParams for plotting
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'lines.linewidth': 2, 'lines.markersize': 10})
plt.rcParams.update({'figure.autolayout': True})


def get_RWMC_nsamples_from_HMC( hmc_nsamples, hmc_info ):
    factor = hmc_info['t'] / hmc_info['dt']
    hmc_nsamples += int( 2 * hmc_nsamples * factor )
    return hmc_nsamples

def plot_U_pot( alpha ):
    '''
    Surface plot of the potential energy associated to the test_example() example.
    
    alpha: float, the parameter of the potential energy.
    '''
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    r = np.linspace(0, .7, 100)
    p = np.linspace(0, 2*np.pi, 100)
    R, P = np.meshgrid(r, p)

    X, Y = R*np.cos(P), R*np.sin(P)
    U_pot = alpha * ( R**2 - 0.25 )**2

    surf = ax.plot_surface(X, Y, U_pot, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha=0.5)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.xaxis.set_major_locator(LinearLocator(5)) # matplotlib documentation
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter('{x:.01f}')
    ax.yaxis.set_major_formatter('{x:.01f}')
    ax.zaxis.set_major_formatter('{x:.01f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    cbar.set_label('$U_{pot}$')
    
    plt.savefig(f'../figures/U_pot_alpha={alpha}.png')
    return


def sample_from_info( info, general, unnorm_logdensity, unnorm_logdensity_grad, return_info=False ):
    sampler = None
    if info['sampler_type'] == 'HMC':
        sampler = HamiltonianMCMC(general['seed'], general['initial_condition'], \
                            unnorm_logdensity, general['burn_in'], unnorm_logdensity_grad, info['mass'], info['t'], info['dt'])
    elif info['sampler_type'] == 'RWMC':
        sampler = RandomWalkMCMC(general['seed'], general['initial_condition'], \
                                 unnorm_logdensity, general['burn_in'], info['step_size'])
    return sampler.sample(general['n_chains'], general['n_samples'], return_info)


def main(config_file, plot_potential):
    if plot_potential:     
        plot_U_pot( 10 )
        plot_U_pot( 1000 )

    config = None
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file) 
    
    unnorm_logdensity, unnorm_logdensity_grad = test_example(config['General']['alpha'], plot=True)
    if config['Case1']['sampler_type'] == 'RWMC' and config['Case2']['sampler_type'] == 'HMC':
        # we adapt the number of samples for the RWMC case so that function evaluations match
        case2_samples = sample_from_info( config['Case2'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
        factor = config['Case2']['t'] / config['Case2']['dt']
        config['General']['n_samples'] += int( 2 * config['General']['n_samples'] * factor )
        case1_samples = sample_from_info( config['Case1'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
    else : 
        case1_samples = sample_from_info( config['Case1'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
        case2_samples = sample_from_info( config['Case2'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sim1, sim_std1 = similarity( case1_samples, unnorm_logdensity, step_x=0.025, step_y=0.025, unorm_log=True, ax=ax[0] )
    sim2, sim_std2 = similarity( case2_samples, unnorm_logdensity, step_x=0.025, step_y=0.025, unorm_log=True, ax=ax[1] ) 
    ax[0].set_title(f'Similarity : {sim1:.2f} pm {2*sim_std1:.2f}')
    ax[1].set_title(f'Similarity : {sim2:.2f} pm {2*sim_std2:.2f}')
    plt.savefig(f'../figures/{config['General']['fig_name']}_alpha={config['General']['alpha']}.png')

    #plot_2d_kde( case1_samples, case2_samples, config['Case1']['title'], \
    #            config['Case2']['title'], config['General']['fig_name'] )
    # this was quite slow and too qualitative
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, help='Path to config file', required=True)
    parser.add_argument('-p', '--plot-potential', type=bool, help='Whether to plot potential energy', \
     required=False, default=False)
    
    args = parser.parse_args()
    main(args.config_file, args.plot_potential) # args.config_file is a filename 
    #plt.show()

