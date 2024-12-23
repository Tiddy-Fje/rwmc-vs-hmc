import yaml
import numpy as np
import argparse
from utils import test_example
from matplotlib import pyplot as plt
from comparison import sample_from_info
from similarity import similarity


def main(config_file, ax=None):
    config = None
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file) 

    unnorm_logdensity, unnorm_logdensity_grad = test_example(config['General']['alpha'])

    param2_scan = config['General']['param_to_scan']
    params = config['Sampler'][param2_scan]

    sims = np.empty(len(params))
    stds = np.empty_like(sims)
    accept_rate = np.empty_like(sims)
    for i, param in enumerate(params):
        config['Sampler'][param2_scan] = param
        samples, accept_rate[i], _ = sample_from_info( config['Sampler'], config['General'], unnorm_logdensity, unnorm_logdensity_grad, return_info=True )

        sims[i], stds[i] = similarity( samples, unnorm_logdensity, step_x=0.025, step_y=0.025, plot=False, unorm_log=True )
    
    if type(param) is list:
        params = [params[i][0] for i in range(len(params))]

    # inspired from an example in the matplotlib documentation
    flag = False

    ax1 = ax
    if ax is None:
        _, ax1 = plt.subplots()
        flag = True

    #ax1.set_xscale('log')
    upper_param2_scan = param2_scan.upper()
    ax1.set_xlabel(f'{upper_param2_scan[:1]}{param2_scan[1:].replace('_', ' ')}')

    color = 'tab:blue'
    ax1.errorbar( params, sims, yerr=2*stds, fmt='s', linestyle='--', color=color )
    ax1.set_ylabel('Target similarity')
    ax1.tick_params( axis='y', labelcolor=color )

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot( params, accept_rate, color=color )
    ax2.tick_params( axis='y', labelcolor=color )
    ax2.set_ylabel('Acceptance rete')

    if flag:
        filename = f'../figures/{config["General"]["fig_name"]}_alpha={config["General"]["alpha"]}.png'
        plt.savefig( filename )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, help='Path to config file', required=True)
    #parser.add_argument('-p', '--plot-potential', type=bool, help='Whether to plot potential energy', \
    #required=False, default=False)
    
    args = parser.parse_args()
    main(args.config_file) # args.config_file is a filename 
