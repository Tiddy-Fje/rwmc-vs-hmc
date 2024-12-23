import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from utils import two_dim_input_fun

def get_discretization(x, y, step_x, step_y):
    '''
    Returns a 2D grid that spans a centered rectangle.
    
    Parameters:
        x (float): Length of the rectangle in the x-direction.
        y (float): Length of the rectangle in the y-direction.
        step_x (float): Size of the grid cell in the x-direction.
        step_y (float): Size of the grid cell in the y-direction.

    Returns:
        grid_x, grid_y (2D arrays): Meshgrid representing the grid points.
    '''
    x_range = np.arange(-x / 2, x / 2 + step_x, step_x)
    y_range = np.arange(-y / 2, y / 2 + step_y, step_y)
    
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    return grid_x, grid_y

def get_frequencies(samples, grid_x, grid_y):
    '''
    Returns the frequencies of samples appearing in each rectangle in the grid.

    Parameters:
        samples (ndarray): 2 or 3 D array with last two dimensions = chain_length, 2. If present, the firs dimension is n_chains.
        grid_x, grid_y (2D arrays): The grid generated by get_discretization.

    Returns:
        freq_array (2D array): Frequency of samples in each grid cell.
    '''
    x_edges = grid_x[0, :]
    y_edges = grid_y[:, 0]
    
    if len(samples.shape) == 2 : 
        samples = samples.reshape((1,*samples.shape))

    # count sample frequencies in each cell
    freq_array = []
    for i in range(samples.shape[0]):
        temp, _, _ = np.histogram2d(samples[i,:, 0], samples[i,:, 1], bins=[x_edges, y_edges])
        temp = temp.T # make grid orientation right
        freq_array.append(temp)
    
    return np.array(freq_array)  

def discretize(fun, grid_x, grid_y, unorm_log=False):
    '''
    Returns the numerically approximated probability of the target function in each rectangle.

    Parameters:
        fun (callable): A probability density function, fun(q), where q = (x, y).
        grid_x, grid_y (2D arrays): The grid generated by get_discretization.
        unorm_log : if true, we consider the density not to be normalised.

    Returns:
        prob_array (2D array): Approximated probabilities in each grid cell.
    '''
    step_x = grid_x[0, 1] - grid_x[0, 0]
    step_y = grid_y[1, 0] - grid_y[0, 0]
    
    centers_x = grid_x[:-1, :-1] + step_x / 2
    centers_y = grid_y[:-1, :-1] + step_y / 2
        
    # evaluating the function at rectangle centers
    prob_density = None
    if unorm_log:
        # no need to account for cell size as they are all the same
        prob_density = two_dim_input_fun( fun, centers_x, centers_y )
        prob_density = np.exp(prob_density)
        prob_density /= np.sum(prob_density)
    else:
        prob_density = two_dim_input_fun( fun, centers_x, centers_y ) * step_x * step_y
    
    return prob_density

def similarity( samples, fun, x_length=2, y_length=2, step_x=0.1, step_y=0.1, plot=False, unorm_log=False, ax=None ):
    '''
    Routine for computing similarity associated to arguments.

    Parameters:
        samples : as above
        fun : as above
        x_length : as above
        y_length : as above
        step_x : as above
        step_y : as above
        plot : whether to generate the associated plot
        unorm_log : as above 

    Returns:
        sim : mean similarity over the chains
        sim_std : std of similarity over the chains 
    '''
    grid_x, grid_y = get_discretization(x_length, y_length, step_x, step_y)
    
    probabilities = discretize(fun, grid_x, grid_y, unorm_log=unorm_log)
    
    frequencies = get_frequencies(samples, grid_x, grid_y)
    n_chains = frequencies.shape[0]
    n_samples = frequencies.shape[1]
    
    dists = np.array( [distance.jensenshannon(frequencies[i,:,:].flatten()/n_samples, probabilities.flatten(), 2.0) for i in range(n_chains)] )

    sim = np.mean( 1 - dists )
    sim_std = np.std( 1 - dists ) / np.sqrt(n_chains)

    if plot or ax is not None:
        ax.set_title('Sample Distribution')
        # plotting only the first chain
        l = ax.pcolormesh(grid_x, grid_y, frequencies[0,:,:]/n_samples, shading='auto', cmap='Blues')
        plt.colorbar(l, ax=ax, label='Density')

        #plt.subplot(1, 2, 2)
        #plt.title('Target Distribution')
        #plt.pcolormesh(grid_x, grid_y, probabilities, shading='auto', cmap='Reds')
        #plt.colorbar()
        #plt.title(f'Similarity: {sim:.2f} pm {2*sim_std:.2f}')

    return sim, sim_std


