import os
import csv
import umap
import torch
import umap.plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight


def plot_data_std():
    data = torch.load("biopsy_data2.pt").numpy()
    train_indices = np.load("indices/Train_indices_CnC.npy")
    # train_data = data[train_indices]

    data_std = np.std(data, axis=0)
    print(data_std.shape)
    d_std_min = np.min(data_std[0])
    d_std_max = np.max(data_std[0])
    c = plt.imshow(data_std[0], cmap="CMRmap", vmin = d_std_min, vmax = d_std_max)
    plt.colorbar(c)
    plt.show()

def plot_std_hist():
    data = torch.load("biopsy_data2.pt").numpy()
    data_std = np.std(data, axis=0)
    data_std_flat = data_std.flatten()
    values = np.unique(data_std_flat)
    counts = []

    for value in values:
        count = np.sum(data_std_flat == value)
        counts.append(count)

    print(np.max(values))
    print(np.min(values))
    n, bins, patches = plt.hist(data_std_flat, color="green")
    # plt.plot(bins, values, '--', color='black')
    #
    plt.show()

def umap_plot():
    # reducer = umap.UMAP(n_neighbors=20,
    #     min_dist=0.1)
    data = torch.load("biopsy/biopsy_data2.pt").numpy()[:, 0, :, :]
    labels = np.load("biopsy/biopsy_labels.npy")

    nsamples, nx, ny = data.shape
    data_reshaped = data.reshape((nsamples, nx * ny))
    print(data_reshaped.shape)
    mapper = umap.UMAP(n_components=2).fit(data_reshaped)
    umap.plot.points(mapper, labels=labels)
    # plt.scatter(
    #     embedding[:, 0],
    #     embedding[:, 1],
    #     c=[["blue", "red"][x] for x in labels])
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the biopsy dataset', fontsize=10)
    plt.show()


umap_plot()