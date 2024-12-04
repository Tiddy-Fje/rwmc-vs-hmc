import yaml
import argparse
from main import test_example, plot_2d_kde
from sampler import RandomWalkMCMC, HamiltonianMCMC
from matplotlib import pyplot as plt

def sample_from_info( info, general, unnorm_logdensity, unnorm_logdensity_grad ):
    sampler = None
    if info['sampler_type'] == 'HMC':
        sampler = HamiltonianMCMC(general['seed'], general['initial_condition'], \
                            unnorm_logdensity, unnorm_logdensity_grad, info['mass'], info['t'], info['dt'])
    elif info['sampler_type'] == 'RWMC':
        sampler = RandomWalkMCMC(general['seed'], general['initial_condition'], \
                                 unnorm_logdensity, info['step_size'])
    
    return sampler.sample(general['n_chains'], general['n_samples'])


def main(config_file):
    config = None
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file) 
    
    unnorm_logdensity, unnorm_logdensity_grad = test_example(100, beta=0.25)
    case1_samples = sample_from_info( config['Case1'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
    case2_samples = sample_from_info( config['Case2'], config['General'], unnorm_logdensity, unnorm_logdensity_grad )
    
    plot_2d_kde(case1_samples, config['Case1']['sampler_type'])
    plot_2d_kde(case2_samples, config['Case2']['sampler_type'])
    plt.show()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, help='Path to config file', required=True)
    #parser.add_argument('-c', '--comparison', type=bool, help='Whether to run comparison', \
    # required=False, default=False)
    
    args = parser.parse_args()
    main(args.config_file) # args.config_file is a filename 
