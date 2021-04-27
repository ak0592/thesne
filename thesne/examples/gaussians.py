import numpy as np
import torch
from sklearn.utils import check_random_state
import os
import sys

sys.path.append('/home/lab/akira/MusicVisualization/thesne/thesne')
from model.dynamic_tsne import dynamic_tsne
from examples import plot


def create_blobs(classes=10, dims=100, class_size=100, variance=0.1, steps=4, advection_ratio=0.5, random_state=None):
    random_state = check_random_state(random_state)
    initial_original_data = []

    indices = random_state.permutation(dims)[0:classes]
    means = []
    for c in range(classes):
        mean = np.zeros(dims)
        mean[indices[c]] = 1.0
        means.append(mean)

        initial_original_data.append(random_state.multivariate_normal(mean, np.eye(dims) * variance, class_size))
    initial_original_data = np.concatenate(initial_original_data)
    class_label_list = np.concatenate([[i] * class_size for i in range(classes)])

    all_step_original_data = [np.array(initial_original_data)]
    for step in range(steps - 1):
        next_step_original_data = np.array(all_step_original_data[step])
        for c in range(classes):
            start, end = class_size * c, class_size * (c + 1)
            next_step_original_data[start: end] += advection_ratio * (means[c] - next_step_original_data[start: end])

        all_step_original_data.append(next_step_original_data)

    all_step_original_data = np.stack(all_step_original_data, axis=0)

    return all_step_original_data, class_label_list


def main():
    seed = 0
    steps = 10
    all_step_original_data, class_label_list = create_blobs(classes=10, class_size=200, dims=100, advection_ratio=0.1,
                                                            steps=steps, random_state=seed)

    all_step_visible_data = dynamic_tsne(all_step_original_data, perplexity=70, penalty_lambda=0.1, verbose=1,
                                         sigma_iters=50, random_state=seed, device=device)

    for step, visible_data in enumerate(all_step_visible_data):
        plot.plot(step, visible_data, class_label_list, save_path)

    print('visualize is completed.')


if __name__ == "__main__":
    save_path = '/home/data/akira/music_visualization_res/thesne_res'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device(f'cuda:{os.environ["CUDA_VISIBLE_DEVICES"]}') if torch.cuda.is_available() else torch.device('cpu')
    main()
