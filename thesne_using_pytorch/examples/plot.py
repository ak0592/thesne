import numpy as np
import matplotlib
import matplotlib.pyplot as plt


hsv_colors = [(0.56823266219239377, 0.82777777777777772, 0.70588235294117652),
              (0.078146611341632088, 0.94509803921568625, 1.0),
              (0.33333333333333331, 0.72499999999999998, 0.62745098039215685),
              (0.99904761904761907, 0.81775700934579443, 0.83921568627450982),
              (0.75387596899224807, 0.45502645502645506, 0.74117647058823533),
              (0.028205128205128216, 0.4642857142857143, 0.5490196078431373),
              (0.8842592592592593, 0.47577092511013214, 0.8901960784313725),
              (0.0, 0.0, 0.49803921568627452),
              (0.16774193548387095, 0.82010582010582012, 0.74117647058823533),
              (0.51539855072463769, 0.88888888888888884, 0.81176470588235294)]

rgb_colors = matplotlib.colors.hsv_to_rgb(np.array(hsv_colors).reshape(10, 1, 3))
colors = matplotlib.colors.ListedColormap(rgb_colors.reshape(10, 3))


def plot(step, Y, labels, save_path):
    figure = plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], s=30, c=labels, cmap=colors, linewidth=0)
    plt.colorbar()
    plt.title(f'{step} step projection figure')
    figure.savefig(f'{save_path}/{step}_step.png')