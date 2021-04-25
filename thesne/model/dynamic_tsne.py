import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import check_random_state

from .core import floath
from .core import cost_var
from .core import find_sigma


def movement_penalty(all_step_visible_data, N):
    penalties = []
    for t in range(len(all_step_visible_data) - 1):
        penalties.append(T.sum((all_step_visible_data[t] - all_step_visible_data[t + 1]) ** 2))

    return T.sum(penalties) / (2 * N)


def find_all_step_visible_data(all_step_original_data_shared, all_step_visible_data_shared, sigmas_shared, N, steps,
                               output_dims, n_epochs, initial_lr, final_lr, lr_switch, initial_momentum,
                               final_momentum, momentum_switch, penalty_lambda, metric, verbose=0):
    """Optimize cost wrt all_step_visible_data[t], simultaneously for all t"""

    # Optimization hyper-parameters
    initial_lr = np.array(initial_lr, dtype=floath)
    final_lr = np.array(final_lr, dtype=floath)
    initial_momentum = np.array(initial_momentum, dtype=floath)
    final_momentum = np.array(final_momentum, dtype=floath)

    lr = T.fscalar('lr')
    lr_shared = theano.shared(initial_lr)

    momentum = T.fscalar('momentum')
    momentum_shared = theano.shared(initial_momentum)

    # Penalty hyper-parameter
    penalty_lambda_var = T.fscalar('penalty_lambda')
    penalty_lambda_shared = theano.shared(np.array(penalty_lambda, dtype=floath))

    # Yv velocities
    all_step_visible_progress_shared = []
    zero_velocities = np.zeros((N, output_dims), dtype=floath)
    for t in range(steps):
        all_step_visible_progress_shared.append(theano.shared(np.array(zero_velocities)))

    # Cost
    all_step_original_data_vars = T.fmatrices(steps)
    all_step_visible_data_vars = T.fmatrices(steps)
    all_step_visible_progress_vars = T.fmatrices(steps)
    sigmas_vars = T.fvectors(steps)

    c_vars = []
    for t in range(steps):
        c_vars.append(cost_var(all_step_original_data_vars[t], all_step_visible_data_vars[t], sigmas_vars[t], metric))

    cost = T.sum(c_vars) + penalty_lambda_var * movement_penalty(all_step_visible_data_vars, N)

    # Setting update for all_step_visible_data velocities
    grad_Y = T.grad(cost, all_step_visible_data_vars)

    givens = {lr: lr_shared, momentum: momentum_shared, penalty_lambda_var: penalty_lambda_shared}
    updates = []
    for t in range(steps):
        updates.append((all_step_visible_progress_shared[t], momentum * all_step_visible_progress_vars[t] - lr * grad_Y[t]))

        givens[all_step_original_data_vars[t]] = all_step_original_data_shared[t]
        givens[all_step_visible_data_vars[t]] = all_step_visible_data_shared[t]
        givens[all_step_visible_progress_vars[t]] = all_step_visible_progress_shared[t]
        givens[sigmas_vars[t]] = sigmas_shared[t]

    update_Yvs = theano.function([], cost, givens=givens, updates=updates)

    # Setting update for all_step_visible_data positions
    updates = []
    givens = dict()
    for t in range(steps):
        updates.append((all_step_visible_data_shared[t], all_step_visible_data_vars[t] + all_step_visible_progress_vars[t]))
        givens[all_step_visible_data_vars[t]] = all_step_visible_data_shared[t]
        givens[all_step_visible_progress_vars[t]] = all_step_visible_progress_shared[t]

    update_all_step_visible_data = theano.function([], [], givens=givens, updates=updates)

    # Momentum-based gradient descent
    for epoch in range(n_epochs):
        if epoch == lr_switch:
            lr_shared.set_value(final_lr)
        if epoch == momentum_switch:
            momentum_shared.set_value(final_momentum)

        c = update_Yvs()
        update_all_step_visible_data()
        if verbose:
            print('Epoch: {0}. Cost: {1:.6f}.'.format(epoch + 1, float(c)))

    all_step_visible_data = []
    for t in range(steps):
        all_step_visible_data.append(np.array(all_step_visible_data_shared[t].get_value(), dtype=floath))

    return all_step_visible_data


def dynamic_tsne(all_step_original_data, perplexity=30, all_step_visible_data=None, output_dims=2, n_epochs=1000,
                 initial_lr=2400, final_lr=200, lr_switch=250, init_stdev=1e-4,
                 sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
                 momentum_switch=250, penalty_lambda=0.1, metric='euclidean',
                 random_state=None, verbose=1):
    """Compute sequence of projections from a sequence of matrices of
    observations (or distances) using dynamic t-SNE.
    
    Parameters
    ----------
    all_step_original_data : list of array-likes, each with shape (n_observations, n_features), \
            or (n_observations, n_observations) if `metric` == 'precomputed'.
        List of matrices containing the observations (one per row). If `metric` 
        is 'precomputed', list of pairwise dissimilarity (distance) matrices. 
        Each row in `all_step_original_data[t + 1]` should correspond to the same row in `all_step_original_data[t]`, 
        for every time step t > 1.
    
    perplexity : float, optional (default = 30)
        Target perplexity for binary search for sigmas.
        
    all_step_visible_data : list of array-likes, each with shape (n_observations, output_dims), \
            optional (default = None)
        List of matrices containing the starting positions for each point at
        each time step.
    
    output_dims : int, optional (default = 2)
        Target dimension.
        
    n_epochs : int, optional (default = 1000)
        Number of gradient descent iterations.
        
    initial_lr : float, optional (default = 2400)
        The initial learning rate for gradient descent.
        
    final_lr : float, optional (default = 200)
        The final learning rate for gradient descent.
        
    lr_switch : int, optional (default = 250)
        Iteration in which the learning rate changes from initial to final.
        This option effectively subsumes early exaggeration.
        
    init_stdev : float, optional (default = 1e-4)
        Standard deviation for a Gaussian distribution with zero mean from
        which the initial coordinates are sampled.
        
    sigma_iters : int, optional (default = 50)
        Number of binary search iterations for target perplexity.
        
    initial_momentum : float, optional (default = 0.5)
        The initial momentum for gradient descent.
        
    final_momentum : float, optional (default = 0.8)
        The final momentum for gradient descent.
        
    momentum_switch : int, optional (default = 250)
        Iteration in which the momentum changes from initial to final.
        
    penalty_lambda : float, optional (default = 0.0)
        Movement penalty hyperparameter. Controls how much each point is
        penalized for moving across time steps.
        
    metric : 'euclidean' or 'precomputed', optional (default = 'euclidean')
        Indicates whether `original_data[t]` is composed of observations ('euclidean') 
        or distances ('precomputed'), for all t.
    
    random_state : int or np.RandomState, optional (default = None)
        Integer seed or np.RandomState object used to initialize the
        position of each point. Defaults to a random seed.

    verbose : bool (default = 1)
        Indicates whether progress information should be sent to standard 
        output.
        
    Returns
    -------
    all_step_visible_data : list of array-likes, each with shape (n_observations, output_dims)
        List of matrices representing the sequence of projections. 
        Each row (point) in `all_step_visible_data[t]` corresponds to a row (observation or 
        distance to other observations) in the input matrix `all_step_original_data[t]`, for all t.
    """
    random_state = check_random_state(random_state)

    steps = len(all_step_original_data)
    N = all_step_original_data[0].shape[0]

    if all_step_visible_data is None:
        initial_visible_data = random_state.normal(0, init_stdev, size=(N, output_dims))
        all_step_visible_data = [initial_visible_data] * steps

    for t in range(steps):
        if all_step_original_data[t].shape[0] != N or all_step_visible_data[t].shape[0] != N:
            raise Exception('Invalid datasets.')

        all_step_original_data[t] = np.array(all_step_original_data[t], dtype=floath)

    all_step_original_data_shared, all_step_visible_data_shared, sigmas_shared = [], [], []
    for t in range(steps):
        original_data_shared = theano.shared(all_step_original_data[t])
        sigma_shared = theano.shared(np.ones(N, dtype=floath))

        find_sigma(original_data_shared, sigma_shared, N, perplexity, sigma_iters,
                   metric=metric, verbose=verbose)

        all_step_original_data_shared.append(original_data_shared)
        all_step_visible_data_shared.append(theano.shared(np.array(all_step_visible_data[t], dtype=floath)))
        sigmas_shared.append(sigma_shared)

    all_step_visible_data = find_all_step_visible_data(all_step_original_data_shared, all_step_visible_data_shared,
                                                       sigmas_shared, N, steps, output_dims,
                                                       n_epochs, initial_lr, final_lr, lr_switch, initial_momentum,
                                                       final_momentum, momentum_switch, penalty_lambda, metric, verbose)

    return all_step_visible_data
