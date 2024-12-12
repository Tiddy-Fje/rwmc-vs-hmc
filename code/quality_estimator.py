import yaml
import numpy as np
import argparse
from utils import plot_acf, test_example, plot_eff_sample_size
from sampler import RandomWalkMCMC, HamiltonianMCMC
from comparison import sample_from_info

def main(config_file, plot_potential):
    config = None
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file) 
    
    unnorm_logdensity, unnorm_logdensity_grad = test_example(100)
    case1_samples = sample_from_info( config['Case1'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
    case2_samples = sample_from_info( config['Case2'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
    
    plot_acf( case1_samples, case2_samples, config['Case1']['title'], config['Case2']['title'] )
    plot_eff_sample_size( case1_samples, case2_samples, config['Case1']['title'], \
            config['Case2']['title'], config['Case2']['dt'], config['Case2']['t'] )
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, help='Path to config file', required=True)
    parser.add_argument('-p', '--plot-potential', type=bool, help='Whether to plot potential energy', \
     required=False, default=False)
    
    args = parser.parse_args()
    main(args.config_file, args.plot_potential)