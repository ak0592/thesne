import numpy as np
import torch

from sklearn.utils import check_random_state
from model.core import floath, cost_var, find_sigma, calc_original_simul_prob, calc_visible_simul_prob, calc_square_euclidean_norms


def find_visible_data(original_data, visible_data, sigma, N, output_dims, n_epochs,
                      initial_lr, final_lr, lr_switch, initial_momentum,
                      final_momentum, momentum_switch, metric, square_distances, verbose, device='cpu'):
    """Optimize cost wrt visual_data"""
    # Optimization hyper-parameters
    lr_tensor = torch.from_numpy(np.array(initial_lr, dtype=floath)).to(device)
    momentum_tensor = torch.from_numpy(np.array(initial_momentum, dtype=floath)).to(device)

    # Cost
    original_data_tensor = torch.from_numpy(original_data).clone().to(device)
    visible_data_tensor = torch.tensor(visible_data.copy()).float().to(device)
    visible_progress_tensor = torch.from_numpy(np.zeros((N, output_dims), dtype=floath)).to(device)
    sigma_tensor = torch.from_numpy(sigma).clone().to(device)

    visible_data_tensor.requires_grad_(True)

    # Momentum-based gradient descent
    epoch = 0
    while True:
        if epoch == lr_switch:
            lr_tensor = torch.from_numpy(np.array(final_lr, dtype=floath)).to(device)
        if epoch == momentum_switch:
            momentum_tensor = torch.from_numpy(np.array(final_momentum, dtype=floath)).to(device)

        kl_loss = cost_var(original_data_tensor, visible_data_tensor, sigma_tensor,
                           metric, square_distances, device=device)

        kl_loss.backward()

        with torch.no_grad():

            # Setting update for visible_data velocities
            visible_progress_tensor = \
                momentum_tensor * visible_progress_tensor - lr_tensor * visible_data_tensor.grad

            # Setting update for visible_data positions
            visible_data_tensor += visible_progress_tensor
            visible_data_tensor.grad.zero_()

        if verbose:
            print(f'Epoch: {epoch + 1}. KL_loss: {kl_loss}')
        epoch += 1

        if epoch >= n_epochs:
            break

    return visible_data_tensor.to('cpu').detach().numpy().copy()


def tsne(original_data, perplexity=30, visible_data=None, output_dims=2, n_epochs=1000,
         initial_lr=2400, final_lr=200, lr_switch=250, init_stdev=1e-4,
         sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
         momentum_switch=250, metric='euclidean',
         random_state=None, square_distances='legacy', verbose=1, device='cpu'):
    """Compute sequence of projections from a matriX of
    observations (or distances) using t-SNE.

    Parameters
    ----------
    original_data : array-likes with shape (n_observations, n_features), \
            or (n_observations, n_observations) if `metric` == 'precomputed'.
        matrix containing the observations (one per row). If `metric`
        is 'precomputed', pairwise dissimilarity (distance) matrices.

    perplexity : float, optional (default = 30)
        Target perplexity for binary search for sigmas.

    visible_data : array-likes, each with shape (n_observations, output_dims), \
            optional (default = None)
        matrix containing the starting positions for each point.

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
    visible_data : array-likes, each with shape (n_observations, output_dims)
        matrix representing the projection.
        Each row (point) in `visible_data` corresponds to a row (observation or
        distance to other observations) in the input matrix `original_data`, for all t.
    """
    random_state = check_random_state(random_state)
    if verbose:
        print(f't_sne is using {device}')

    if not isinstance(original_data, np.ndarray):
        original_data = np.array(original_data)

    N = original_data.shape[0]

    if visible_data is None:
        visible_data = random_state.normal(0, init_stdev, size=(N, output_dims))

    sigma = find_sigma(original_data, np.ones(N, dtype=floath), N, perplexity, sigma_iters,
                       metric, square_distances, verbose=verbose, device=device)

    visible_data = find_visible_data(original_data, visible_data, sigma, N, output_dims, n_epochs,
                                     initial_lr, final_lr, lr_switch, initial_momentum,
                                     final_momentum, momentum_switch, metric, square_distances, verbose, device=device)

    return visible_data
