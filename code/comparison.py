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


def sample_from_info( info, general, unnorm_logdensity, unnorm_logdensity_grad, return_info=False ):
    sampler = None
    if info['sampler_type'] == 'HMC':
        sampler = HamiltonianMCMC(general['seed'], general['initial_condition'], \
                            unnorm_logdensity, general['burn_in'], unnorm_logdensity_grad, info['mass'], info['t'], info['dt'])
    elif info['sampler_type'] == 'RWMC':
        sampler = RandomWalkMCMC(general['seed'], general['initial_condition'], \
                                 unnorm_logdensity, general['burn_in'], info['step_size'])
    return sampler.sample(general['n_chains'], general['n_samples'], return_info)


def main(config_file, ax=None):
    config = None
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file) 
    
    unnorm_logdensity, unnorm_logdensity_grad = test_example(config['General']['alpha'])
    if config['Case1']['sampler_type'] == 'RWMC' and config['Case2']['sampler_type'] == 'HMC':
        # we adapt the number of samples for the RWMC case so that function evaluations match
        case2_samples = sample_from_info( config['Case2'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
        config['General']['n_samples'] = get_RWMC_nsamples_from_HMC( config['General']['n_samples'], config['Case2'] )
        case1_samples = sample_from_info( config['Case1'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
    else : 
        case1_samples = sample_from_info( config['Case1'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
        case2_samples = sample_from_info( config['Case2'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )

    flag = False
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        flag = True

    sim1, sim_std1 = similarity( case1_samples, unnorm_logdensity, step_x=0.025, step_y=0.025, unorm_log=True, ax=ax[0] )
    sim2, sim_std2 = similarity( case2_samples, unnorm_logdensity, step_x=0.025, step_y=0.025, unorm_log=True, ax=ax[1] ) 
    ax[0].set_xlabel('$q_1$')
    ax[1].set_xlabel('$q_1$')
    ax[0].set_ylabel('$q_2$')
    ax[1].set_ylabel('$q_2$')
    ax[0].set_title(f'Similarity : {sim1:.2f} pm {2*sim_std1:.2f}')
    ax[1].set_title(f'Similarity : {sim2:.2f} pm {2*sim_std2:.2f}')


    if flag:
        filename = f'../figures/{config["General"]["fig_name"]}_alpha={config["General"]["alpha"]}.png'
        plt.savefig( filename )

    #plot_2d_kde( case1_samples, case2_samples, config['Case1']['title'], \
    #            config['Case2']['title'], config['General']['fig_name'] )
    # this was quite slow and too qualitative
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, help='Path to config file', required=True)
    
    args = parser.parse_args()
    main(args.config_file) # args.config_file is a filename 

