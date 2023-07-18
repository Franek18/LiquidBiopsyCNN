import os
import csv
import wandb
import torch
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

from hparams import hparams
from data import CancerDataset, save_data_reduced
from training import training_loop
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Select type of Model", default='vanilla', type=str)
    parser.add_argument("--test", help="Select type data split", default='random', type=str)
    args = parser.parse_args()

    # model = None
    # dataset = None
    # save_data_reduced()
    if args.model == "Flat":
        # model = FlatCNN()
        model_type = "Flat"
        dataset = CancerDataset("biopsy/biopsy_data_vectors.pt", "biopsy/biopsy_labels.npy")
        model_hparams = hparams["FlatCnn"]
        wandb.init(project="mgr_biopsy", name="FlatCNN" + "_" + args.test + "_" + str(datetime.date.today()))
    elif args.model == "Hybrid":
        # model = HybridCNN()
        model_type = "Hybrid"
        dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")
        model_hparams = hparams["HybridCnn"]
        wandb.init(project="mgr_biopsy", name="HybridCnn" + "_" + args.test + "_" + str(datetime.date.today()))
    else:
        # model = VanillaCNN()
        model_type = "Vanilla"
        dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")
        model_hparams = hparams["VanillaCnn"]
        wandb.init(project="mgr_biopsy", name="VanillaCNN" + "_" + args.test + "_" + str(datetime.date.today()))

    class_weights = torch.Tensor(dataset.getClassWeights())

    if args.test == "hospital":
        train_indices = np.load("indices/New_train_indices_CnC.npy")
        test_indices = np.load("indices/New_test_indices_CnC.npy")
    elif args.test == "cancer":
        train_indices = np.load("indices/New_disease_train_indices_CnC.npy")
        test_indices = np.load("indices/New_disease_test_indices_CnC.npy")
    else:
        train_indices = np.load("indices/Train_indices_CnC.npy")
        test_indices = np.load("indices/Test_indices_CnC.npy")

    # train_indices, val_indices = train_test_split(train_indices, test_size=0.1, shuffle=True)
    # print(len(val_indices[0]))
    # print(len(val_indices[1]))
    # print(train_indices.shape)
    y_train = dataset.labels[train_indices]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    # val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # print("Test size = ", len(test_dataset))

    datasets = {
        "non-test": train_dataset,
        # "val": val_dataset,
        "test": test_dataset
    }

    training_loop(model_type, class_weights, datasets, y_train, model_hparams)


