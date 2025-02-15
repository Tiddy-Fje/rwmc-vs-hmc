import numpy as np
from abc import ABC, abstractmethod

class MCMCSampler(ABC):
    '''
    Abstract base class for MCMC Samplers. Provides a common interface for different MCMC algorithms.
    '''
    def __init__(self, seed, initial_condition, unnorm_logdensity, burn_in):
        self.seed = seed
        self.initial_condition = np.array(initial_condition)
        self.unnorm_logdensity = unnorm_logdensity
        # we use log densities here to avoid underflow
        
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)
        self.state = self.initial_condition.copy() # current state
        self.n_evaluations = 0 # number of function evaluations
    
    def reset_state(self, only_state=False):
        '''
        Resets the RNG and state to their initial values.

        Parameters:
        only_state (bool): If True, only resets the state, not the RNG. This is useful for creating independent chains starting with the same initial condition.
        '''
        self.state = self.initial_condition.copy()
        if not only_state:
            self.rng = np.random.default_rng(self.seed)
    
    def MH_acceptance_rule(self, current_state, proposed_state, current_logdensity, proposed_logdensity):
        '''
        Accepts or rejects a proposed state based on the Metropolis-Hastings acceptance rule.

        Returns:
        np.ndarray: The new state.
        bool: Whether the proposed state was accepted.
        '''
        prob = np.exp( proposed_logdensity - current_logdensity )
        acceptance_prob = min(1, prob)
        if self.rng.uniform() < acceptance_prob:
            return proposed_state, True
        else:
            return current_state, False

    def sample(self, n_chains, n_samples, return_info=False):
        '''
        Generates samples using the Metropolis-Hastings algorithm.

        Parameters:
        n_chains (int): Number of independent chains.
        n_samples (int): Number of samples per chain.
        return_info (bool): If True, returns acceptance_rate and n_evaluations associated to the sampling process.

        Returns:
        np.ndarray: Samples of shape (n_chains, n_samples, dim).
        float: Acceptance rate.
        int: Number of function evaluations.
        '''
        assert self.burn_in < n_samples, 'Burn-in period must be less than number of samples.'

        dim = len(self.initial_condition)
        samples = np.zeros((n_chains, n_samples, dim))
        acceptance_rate = 0
        
        # This could be made more efficient, especially for RWMC, but 
        # will only do this if runtime is too long.
        for chain_idx in range(n_chains):
            self.reset_state(only_state=True) # make chains start at the same place
            current_state = self.state.copy()
            for sample_idx in range(n_samples):
                current_state, accepted = self.proposal_step(current_state)
                acceptance_rate += float(accepted)
                samples[chain_idx, sample_idx, :] = current_state
        
        acceptance_rate /= (n_chains * n_samples)
        if return_info:
            return samples[:, self.burn_in:, :], acceptance_rate, self.n_evaluations

        print(f'Acceptance Rate: {acceptance_rate:.2f}')
        print(f'Number of function evaluations: {self.n_evaluations}')
        return samples[:, self.burn_in:, :]
    
    @abstractmethod
    def proposal_step(self, current_state):
        '''
        Generate a proposed state based on the current state, and accept it according to associated MH rule.
        Must be implemented in subclasses.
        '''
        pass

class RandomWalkMCMC(MCMCSampler):
    '''
    Random Walk MCMC sampler.
    '''
    def __init__(self, seed, initial_condition, unnorm_logdensity, burn_in, step_size):
        super().__init__(seed, initial_condition, unnorm_logdensity, burn_in)
        # to allow for step_size to be a scalar or an array
        try:
            iterator = iter(step_size)
            self.step_size = np.array(step_size)
        except TypeError:
            self.step_size = step_size
    
    def proposal_step(self, current_state):
        '''
        Proposes a new state using a random walk.
        '''
        proposed_state = current_state + self.step_size * self.rng.normal(0, 1, size=current_state.shape)
        current_logdensity = self.unnorm_logdensity(current_state)
        proposed_logdensity = self.unnorm_logdensity(proposed_state)
        self.n_evaluations += 1 
        # we only add one since in principle could evaluate density of energy difference

        return self.MH_acceptance_rule(current_state, proposed_state, current_logdensity, proposed_logdensity)

class HamiltonianMCMC(MCMCSampler):
    '''
    Hamiltonian Monte Carlo sampler.
    '''
    def __init__(self, seed, initial_condition, unnorm_logdensity, burn_in, unnorm_logdensity_grad, mass, leapfrog_time, dt):
        super().__init__(seed, initial_condition, unnorm_logdensity, burn_in)
        assert (np.array(mass) != 0).all(), 'Mass must be non-zero.'
        # to allow for mass to be a scalar or an array
        try:
            iterator = iter(mass)
            self.mass = np.array(mass)
        except TypeError:
            self.mass = mass
        self.dt = dt
        self.num_steps = int(leapfrog_time / dt)
        self.unnorm_logdensity_grad = unnorm_logdensity_grad
    
    def proposal_step(self, current_state):
        '''
        Proposes a new state using Hamiltonian dynamics.
        '''
        momentum = np.sqrt(self.mass) * self.rng.normal(0, 1, size=current_state.shape)
        proposed_state = current_state.copy()
        proposed_momentum = momentum.copy()
        
        # integrate Hamiltonian dynamics using leapfrog
        for _ in range(self.num_steps):
            # we use : grad U = - grad log p
            proposed_momentum += 0.5 * self.dt * self.unnorm_logdensity_grad(proposed_state) 
            proposed_state += self.dt * proposed_momentum / self.mass
            proposed_momentum += 0.5 * self.dt * self.unnorm_logdensity_grad(proposed_state)
            self.n_evaluations += 2

        current_energy = -self.unnorm_logdensity(current_state) + 0.5 * np.dot(momentum/self.mass,momentum)
        proposed_energy = -self.unnorm_logdensity(proposed_state) + 0.5 * np.dot(proposed_momentum/self.mass,proposed_momentum)
        self.n_evaluations += 1
        # we only add one since in principle could evaluate density of energy difference
        
        return self.MH_acceptance_rule(current_state, proposed_state, -current_energy, -proposed_energy)