import numpy as np
import torch

from sklearn.utils import check_random_state
from model.core import floath, cost_var, find_sigma, calc_original_simul_prob, calc_visible_simul_prob, calc_square_euclidean_norms


def movement_penalty(all_step_visible_data_tensor, N, device='cpu'):
    penalties = torch.zeros(all_step_visible_data_tensor.size()[0], device=device)
    for t in range(all_step_visible_data_tensor.size()[0] - 1):
        penalties[t] = torch.sum((all_step_visible_data_tensor[t] - all_step_visible_data_tensor[t + 1]) ** 2)

    return torch.sum(penalties) / (2 * N)


def create_subtract_matrix(one_step_tensor, device='cpu'):
    N, dims = one_step_tensor.size()
    subtract_matrix = torch.zeros(N, N, dims).float().to(device)
    for i in range(N):
        for j in range(dims):
            subtract_matrix[i, j] = one_step_tensor[i] - one_step_tensor[j]

    return subtract_matrix


def find_all_step_visible_data(all_step_original_data, all_step_visible_data, all_step_sigmas, N, steps,
                               output_dims, n_epochs, initial_lr, final_lr, lr_switch, initial_momentum,
                               final_momentum, momentum_switch, penalty_lambda, metric, square_distances, verbose=0, device='cpu'):
    """Optimize cost wrt all_step_visible_data[t], simultaneously for all t"""

    # Optimization hyper-parameters
    lr_tensor = torch.from_numpy(np.array(initial_lr, dtype=floath)).to(device)
    momentum_tensor = torch.from_numpy(np.array(initial_momentum, dtype=floath)).to(device)

    # Penalty hyper-parameter
    penalty_lambda_tensor = torch.from_numpy(np.array(penalty_lambda, dtype=floath)).to(device)

    # Cost
    all_step_original_data_tensors = torch.from_numpy(all_step_original_data).clone().to(device)
    all_step_visible_data_tensors = torch.tensor(all_step_visible_data.copy()).float().to(device)
    all_step_visible_progress_tensors = torch.from_numpy(np.zeros((steps, N, output_dims), dtype=floath)).to(device)
    all_step_sigmas_tensors = torch.from_numpy(all_step_sigmas).clone().to(device)

    all_step_visible_data_tensors.requires_grad_(True)

    # Momentum-based gradient descent
    epoch = 0
    while True:
        if epoch == lr_switch:
            lr_tensor = torch.from_numpy(np.array(final_lr, dtype=floath)).to(device)
        if epoch == momentum_switch:
            momentum_tensor = torch.from_numpy(np.array(final_momentum, dtype=floath)).to(device)

        c_vars = torch.zeros(steps).to(device)
        for t in range(steps):
            c_vars[t] = cost_var(all_step_original_data_tensors[t], all_step_visible_data_tensors[t], all_step_sigmas_tensors[t],
                                 metric, square_distances, device=device)

        penalty = movement_penalty(all_step_visible_data_tensors, N, device=device)
        kl_loss = torch.sum(c_vars)
        cost = kl_loss + penalty_lambda_tensor * penalty
        cost.backward()

        with torch.no_grad():

            # Setting update for all_step_visible_data velocities
            all_step_visible_progress_tensors = \
                momentum_tensor * all_step_visible_progress_tensors - lr_tensor * all_step_visible_data_tensors.grad

            # Setting update for all_step_visible_data positions
            all_step_visible_data_tensors += all_step_visible_progress_tensors
            all_step_visible_data_tensors.grad.zero_()

        if verbose:
            print(f'Epoch: {epoch + 1}. KL_loss: {kl_loss}, penalty: {penalty}')
        epoch += 1

        if epoch >= n_epochs:
            break

    final_all_step_visible_data = []

    for t in range(steps):
        final_all_step_visible_data.append(all_step_visible_data_tensors[t].to('cpu').detach().numpy().copy())

    return final_all_step_visible_data


def dynamic_tsne(all_step_original_data, perplexity=30, all_step_visible_data=None, output_dims=2, n_epochs=1000,
                 initial_lr=2400, final_lr=200, lr_switch=250, init_stdev=1e-4,
                 sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
                 momentum_switch=250, penalty_lambda=0.1, metric='euclidean',
                 random_state=None, square_distances='legacy', verbose=1, device='cpu'):
    """Compute sequence of projections from a sequence of matrices of
    observations (or distances) using dynamic t-SNE.

    Parameters
    ----------
    all_step_original_data : list of array-likes,xxx each with shape (n_observations, n_features), \
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
    if verbose:
        print(f'dynamic_tsne is using {device}')

    steps = len(all_step_original_data)
    N = all_step_original_data[0].shape[0]

    if all_step_visible_data is None:
        initial_visible_data = random_state.normal(0, init_stdev, size=(N, output_dims))
        all_step_visible_data = [initial_visible_data] * steps

    for t in range(steps):
        if all_step_original_data[t].shape[0] != N or all_step_visible_data[t].shape[0] != N:
            raise Exception('Invalid datasets.')

        all_step_original_data[t] = np.array(all_step_original_data[t], dtype=floath)
        all_step_visible_data[t] = np.array(all_step_visible_data[t], dtype=floath)

    if isinstance(all_step_visible_data, list):
        all_step_visible_data = np.stack(all_step_visible_data, axis=0)

    all_step_sigmas = []
    for t in range(steps):
        original_data = all_step_original_data[t]

        sigma = find_sigma(original_data, np.ones(N, dtype=floath), N, perplexity, sigma_iters,
                           metric, square_distances, verbose=verbose, device=device)

        all_step_sigmas.append(sigma)

    all_step_sigmas = np.stack(all_step_sigmas, axis=0)
    all_step_visible_data = find_all_step_visible_data(all_step_original_data, all_step_visible_data,
                                                       all_step_sigmas, N, steps, output_dims,
                                                       n_epochs, initial_lr, final_lr, lr_switch, initial_momentum,
                                                       final_momentum, momentum_switch, penalty_lambda, metric,
                                                       square_distances, verbose, device=device)

    return all_step_visible_data
