import numpy as np
import torch

epsilon = 1e-16
floath = np.float32


def calc_square_euclidean_norms(X):
    N = X.size()[0]
    ss = (X ** 2).sum(dim=1)

    return ss.reshape(N, 1) + ss.reshape(1, N) - 2 * torch.mm(X, torch.t(X))


def calc_original_cond_prob(X, sigma, metric):
    N = X.size()[0]

    if metric == 'euclidean':
        data_distances = calc_square_euclidean_norms(X)
    elif metric == 'precomputed':
        data_distances = X ** 2
    else:
        raise Exception('Invalid metric')

    esqdistance = torch.exp(-data_distances / ((2 * (sigma ** 2)).reshape(N, 1)))
    esqdistance_zd = esqdistance.clone().fill_diagonal_(0)

    row_sum = torch.sum(esqdistance_zd, dim=1).reshape((N, 1))

    return esqdistance_zd / row_sum  # Possibly dangerous


def calc_original_simul_prob(original_data_tensor, sigma_tensor, metric):
    p_Xp_given_X = calc_original_cond_prob(original_data_tensor, sigma_tensor, metric)

    return (p_Xp_given_X + torch.t(p_Xp_given_X)) / (2 * p_Xp_given_X.size()[0])


def calc_visible_simul_prob(Y):
    numerators = 1 / (calc_square_euclidean_norms(Y) + 1)
    numerators.fill_diagonal_(0)

    return numerators / numerators.sum()  # Possibly dangerous


def cost_var(original_data_tensor, visible_data_tensor, sigma_tensor, metric, device='cpu'):
    epsilon_tensor = torch.tensor(epsilon).float().to(device)

    original_simul_prob = calc_original_simul_prob(original_data_tensor, sigma_tensor, metric)
    visible_simul_prob = calc_visible_simul_prob(visible_data_tensor)

    PXc = torch.maximum(original_simul_prob, epsilon_tensor)
    PYc = torch.maximum(visible_simul_prob, epsilon_tensor)

    # Possibly dangerous (clipped)
    return torch.sum(original_simul_prob * torch.log(PXc / PYc))


def find_sigma(original_data, sigma, N, perplexity, sigma_iters,
               metric, verbose=0, device='cpu'):
    """Binary search on sigma for a given perplexity."""
    original_data_tensor = torch.from_numpy(original_data).clone().to(device)
    sigma_tensor = torch.from_numpy(sigma).clone().to(device)

    target = torch.tensor(np.log(perplexity)).float().to(device)
    epsilon_tensor = torch.tensor(epsilon).float().to(device)

    # Setting update for binary search interval
    sigmin = torch.from_numpy(np.full(N, np.sqrt(epsilon), dtype=floath)).to(device)
    sigmax = torch.from_numpy(np.full(N, np.inf, dtype=floath)).to(device)

    for i in range(sigma_iters):
        P = torch.maximum(calc_original_cond_prob(original_data_tensor, sigma_tensor, metric), epsilon_tensor)
        entropy = -torch.sum(P * torch.log(P), dim=1)

        sigmin = torch.where(torch.lt(entropy, target), sigma_tensor, sigmin)
        sigmax = torch.where(torch.gt(entropy, target), sigma_tensor, sigmax)

        # Setting update for sigma_tensor according to search interval
        sigma_tensor = torch.where(torch.isinf(sigmax), sigma_tensor * 2, (sigmin + sigmax) / 2.)

        if verbose:
            print('Iteration: {0}.'.format(i + 1))
            print('Perplexities in [{0:.4f}, {1:.4f}].'.format(np.exp(entropy.to('cpu').detach().numpy().copy().min()),
                                                               np.exp(entropy.to('cpu').detach().numpy().copy().max())))

    if np.any(np.isnan(np.exp(entropy.to('cpu').detach().numpy().copy()))):
        raise Exception('Invalid sigmas. The perplexity is probably too low.')

    return sigma_tensor.to('cpu').detach().numpy().copy()
