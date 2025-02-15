import matplotlib.pyplot as plt
from scan import main as scan_main
from nsample_evolution import main as nsample_main
from comparison import main as comparison_main
from matplotlib import cm
from utils import test_example
import yaml
from matplotlib.ticker import LinearLocator
import numpy as np

config_dir = '../configs/'
fig_dir = '../figures/'
png = '.png'
yaml_ = '.yaml'

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

def fig_from_param( function, param, x_label, ax=None ): 
    ax_none = None
    if ax is None:
        param_fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax_none = True
    function(f'{config_dir}{param}_alpha=10{yaml_}', ax=ax[0])
    function(f'{config_dir}{param}{yaml_}', ax=ax[1])
    if x_label != None:
        ax[0].set_xlabel(x_label)
        ax[1].set_xlabel(x_label)
    ax[0].set_title(r'$\alpha$ = 10')
    ax[1].set_title(r'$\alpha$ = 1000')
    if ax_none :
        filename = f'{fig_dir}{param}{png}'
        plt.savefig( filename )
    print(f'{param} figure done !')


param_fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
param = 'mass_sym'
comparison_main(f'{config_dir}{param}{yaml_}', ax=ax)
title0 = ax[0].get_title()
title1 = ax[1].get_title()
with open(f'{config_dir}{param}{yaml_}', 'r') as file:
    config = yaml.safe_load(file) 
m1 = config['Case1']['mass']
m2 = config['Case2']['mass']
alpha = config['General']['alpha']
m1_lab = f'(m_1,m_2)=({m1[0]:.2f},{m1[1]:.2f})'
m2_lab = f'(m_1,m_2)=({m2[0]:.2f},{m2[1]:.2f})'
ax[0].set_title( r'$'+m1_lab+r'$'+f'\n{title0}' )
ax[1].set_title( r'$'+m2_lab+r'$'+f'\n{title1}' )
plt.savefig( f'{fig_dir}{param}{png}' )
print(f'{param} figure done !')


fig_from_param( scan_main, 'mass_scan', '$m=m_1=m_2$' )
fig_from_param( scan_main, 't_scan', '$t$' )
fig_from_param( scan_main, 'dt_scan', r'$\Delta t$' )
fig_from_param( nsample_main, 'nsamples_evolution', 'Number of samples' )
fig_from_param( scan_main, 'rwmc_scan', None )

plot_U_pot( 10 )
plot_U_pot( 1000 )

param_fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
_, __ = test_example( 10, ax=ax[0] )
_, __ = test_example( 1000, ax=ax[1] )
ax[0].set_title(r'$\alpha$ = 10')
ax[1].set_title(r'$\alpha$ = 1000')
plt.savefig(f'../figures/alpha_density.png')