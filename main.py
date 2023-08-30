import os
import csv

import pandas as pd

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
    parser.add_argument("--data", help="Select type of data's representation", default='standard', type=str)
    parser.add_argument("--test", help="Select type data's split", default='random', type=str)
    args = parser.parse_args()
    project_name = None
    standard = False
    # model = None
    # dataset = None
    # save_data_reduced()
    if args.model == "Flat":
        # model = FlatCNN()
        model_type = "Flat"
        dataset = CancerDataset("biopsy/biopsy_data_vectors.pt", "biopsy/biopsy_labels.npy")
        model_hparams = hparams["FlatCNN"]
        data_type = "vector"
        project_name = "FlatCNN" + "_" + data_type + '_' + args.test + "_" + str(datetime.date.today())
        wandb.init(project="mgr_biopsy", name=project_name)
    elif args.model == "Hybrid":
        # model = HybridCNN()
        model_type = "Hybrid"
        data_type = args.data
        if data_type == "standard":
            standard = True
            dataset = CancerDataset("biopsy/biopsy_data2.pt", "biopsy/biopsy_labels.npy")
        else:
            dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")
        model_hparams = hparams["HybridCNN"]
        project_name = "HybridCNN" + "_" + data_type + '_' + args.test + "_" + str(datetime.date.today())
        wandb.init(project="mgr_biopsy", name=project_name)
    elif args.model == "Resnet":
        model_type = "Resnet18"
        model_hparams = hparams["ResNet18"]
        data_type = args.data
        if data_type == "standard":
            standard = True
            dataset = CancerDataset("biopsy/biopsy_data2.pt", "biopsy/biopsy_labels.npy")
        else:
            dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")
    else:
        # model = VanillaCNN()
        model_type = "Vanilla"
        data_type = args.data
        if data_type == "standard":
            standard = True
            dataset = CancerDataset("biopsy/biopsy_data2.pt", "biopsy/biopsy_labels.npy")
        else:
            dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")
        model_hparams = hparams["VanillaCNN"]
        project_name = "VanillaCNN" + "_" + data_type + '_' + args.test + "_" + str(datetime.date.today())
        wandb.init(project="mgr_biopsy", name=project_name)

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
    exp_results = []
    for _ in range(3):
        run_results = training_loop(standard, model_type, class_weights, datasets, y_train, model_hparams)
        exp_results.append(run_results)

    final_result = pd.concat(exp_results, ignore_index=True)
    mean_std_result = pd.DataFrame({'mean': final_result.mean(), 'std': final_result.std()})

    if not os.path.exists("csv_results"):
        os.mkdir("csv_results")

    mean_std_result.to_csv(os.path.join("csv_results", project_name + ".csv"))

    # wandb.log({"Mean Test Loss": np.mean(test_loss), "Mean Test AUC": np.mean(test_auc),
    #            "Mean Test Bal Acc": np.mean(test_bal_acc)})
    # wandb.log({"Test Bal Acc std": std_test_bal_acc, "Test AUC std": std_test_auc})
    #
    # wandb.log({"Mean Train AUC": mean_best_train_auc, "Mean Train Bal Acc": mean_best_train_bal_acc,
    #            "Train Bal Acc std": std_best_train_bal_acc, "Train AUC std": std_best_train_auc})
    #
    # wandb.log({"Mean Val AUC": mean_best_val_auc, "Mean Val Bal Acc": mean_best_val_bal_acc,
    #            "Val Bal Acc std": std_best_val_bal_acc, "Val AUC std": std_best_val_auc})
