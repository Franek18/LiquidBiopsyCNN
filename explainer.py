import copy
import glob
import os.path
import time

import matplotlib.pyplot as plt
import shap
import tqdm
import torch
import argparse
import datetime
import numpy as np
import pandas as pd

from torch import nn
from tqdm import tqdm
from lime import lime_image
from hparams import hparams
from data import CancerDataset
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from skimage.segmentation import mark_boundaries
from sklearn.model_selection import train_test_split
from models import get_ResNet, VanillaCNN, FlatCNN, HybridCNN
# from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = None


def train_model(model, num_epochs, dataloaders, optimizer, criterion, scheduler, dataset_sizes):
    best_model = copy.deepcopy(model.state_dict())
    # validation balanced accuracy
    best_auc = 0.0
    best_bal_acc = 0.0
    best_epoch = 0

    global device

    for epoch in tqdm(range(num_epochs), desc=f"Training [{num_epochs} epochs]"):
        # print("Epoch no = ", epoch)
        start_time = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            all_outputs = []
            all_preds = []
            all_labels = []
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # print("batch size = ", inputs.shape[0])
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    max_outputs = outputs[:, 1]
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    # print(max_outputs)
                    all_outputs.extend(outputs.detach().cpu().numpy())
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            all_outputs = np.array(all_outputs, dtype=float)
            all_preds = np.array(all_preds, dtype=int)
            all_labels = np.array(all_labels, dtype=int)

            if phase == 'train':
                scheduler.step()
                end_time = time.time()
                epoch_training_time = end_time - start_time
                print(f"Epoch trainign time = {epoch_training_time}")

            epoch_loss = running_loss / dataset_sizes[phase]
            auc = roc_auc_score(all_labels, all_outputs[:, 1])
            bal_acc = balanced_accuracy_score(all_labels, all_preds)

            if phase == 'val' and (auc > best_auc):
                # deep copy the model
                best_epoch = epoch
                best_auc = auc
                best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)

    return model, epoch_training_time


def train_and_save_model():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Select type of Model", default='vanilla', type=str)
    parser.add_argument("--data", help="Select type data split", default='standard', type=str)
    parser.add_argument("--test", help="Select type data split", default='random', type=str)
    args = parser.parse_args()

    standard = False

    if args.model == "Flat":
        model = FlatCNN().to(device)
        dataset = CancerDataset("biopsy/biopsy_data_vectors.pt", "biopsy/biopsy_labels.npy")
        model_hparams = hparams["FlatCNN"]
        data_type = "vector"
    elif args.model == "Hybrid":

        data_type = args.data
        if data_type == "standard":
            standard = True
            dataset = CancerDataset("biopsy/biopsy_data2.pt", "biopsy/biopsy_labels.npy")
        else:
            dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")

        model = HybridCNN(standard).to(device)
        model_hparams = hparams["HybridCNN"]
    elif args.model == "Resnet":
        model_hparams = hparams["ResNet18"]
        model = get_ResNet(model_hparams["Dropout"]).to(device)
        data_type = args.data
        if data_type == "standard":
            standard = True
            dataset = CancerDataset("biopsy/biopsy_data2.pt", "biopsy/biopsy_labels.npy")
        else:
            dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")
    else:

        data_type = args.data
        if data_type == "standard":
            standard = True
            dataset = CancerDataset("biopsy/biopsy_data2.pt", "biopsy/biopsy_labels.npy")
        else:
            dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")

        model = VanillaCNN(standard).to(device)
        model_hparams = hparams["VanillaCNN"]


    class_weights = torch.Tensor(dataset.getClassWeights())

    all_train_indices = np.load("indices/Train_indices_CnC.npy")
    test_indices = np.load("indices/Test_indices_CnC.npy")

    train_indices, val_indices = train_test_split(all_train_indices)

    y_train = dataset.labels[train_indices]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(class_weights.to(device))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=model_hparams["lr"],
        weight_decay=model_hparams["weight_decay"]
    )
    # and SteLR
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=model_hparams["step_size"],
                                           gamma=model_hparams["gamma"])

    train_size = len(datasets["train"])
    val_size = len(datasets["val"])
    test_size = len(datasets["test"])

    dataset_sizes = {'train': train_size, 'val': val_size}

    # Get dataloaders
    trainloader = DataLoader(datasets["train"], batch_size=model_hparams["train_batch_size"])
    validloader = DataLoader(datasets["val"], batch_size=model_hparams["val_batch_size"])
    testloader = DataLoader(datasets["test"], batch_size=model_hparams["val_batch_size"])

    dataloaders = {'train': trainloader, 'val': validloader}

    # train num_epochs = 1
    model, epoch_training_time = train_model(model, 1, dataloaders,
                        optimizer, loss_fn, exp_lr_scheduler, dataset_sizes)

    test_sample = dataset.data.float().to(device)[0:1]
    print(test_sample.shape)
    model.eval()
    start_time = time.time()
    model(test_sample)
    end_time = time.time()
    inf_time = end_time - start_time

    if not os.path.exists("models"):
        os.mkdir("models")

    df = pd.DataFrame({"Training epoch time": [epoch_training_time], "Inference time": [inf_time]})
    results_csv_file = 'times.csv'

    if not os.path.exists(results_csv_file):
        df.to_csv(results_csv_file, index=False)
    else:
        df.to_csv(results_csv_file, mode='a', index=False, header=False)

    # torch.save(model.state_dict(), os.path.join("models", args.model + ".pt"))


def shap_explanation(model_name, model, X_train, X_test):
    explainer = shap.DeepExplainer(model, X_train[:100])
    # explainer = shap.Explainer(model)


    # shap.summary_plot(shap_values, X_test[100:101], show=False)
    # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    # test_numpy = np.swapaxes(np.swapaxes(X_test[100:101].cpu().numpy(), 1, -1), 1, 2)
    # shap.image_plot(shap_numpy, -test_numpy)

    no_tests = len(X_test)

    avg_negative_shap_value = np.zeros((267, 531))
    avg_positive_shap_value = np.zeros((267, 531))

    for i in range(no_tests):
        shap_values = np.array(explainer.shap_values(X_test[i:i+1]))

        negative_shap_value = shap_values[0][0][0]
        positive_shap_value = shap_values[1][0][0]

        avg_negative_shap_value += negative_shap_value
        avg_positive_shap_value += positive_shap_value

    avg_negative_shap_value /= no_tests
    avg_positive_shap_value /= no_tests

    neg_importance_pd = pd.DataFrame()
    pos_importance_pd = pd.DataFrame()

    # plt.savefig("Resnet_shap.png")
    flat_avg_neg_shap = avg_negative_shap_value.flatten()
    flat_avg_pos_shap = avg_positive_shap_value.flatten()

    top_avg_neg_indices = np.argsort(flat_avg_neg_shap)[::-1][:100]
    top_avg_pos_indices = np.argsort(flat_avg_pos_shap)[::-1][:100]

    neg_importance_pd['row'] = top_avg_neg_indices // 531
    neg_importance_pd['column'] = top_avg_neg_indices % 531
    neg_importance_pd['shap_value'] = flat_avg_neg_shap[top_avg_neg_indices]

    pos_importance_pd['row'] = top_avg_pos_indices // 531
    pos_importance_pd['column'] = top_avg_pos_indices % 531
    pos_importance_pd['shap_value'] = flat_avg_pos_shap[top_avg_pos_indices]

    neg_importance_pd.to_csv(model_name + "_shap_neg_importance.csv")
    pos_importance_pd.to_csv(model_name + "_shap_pos_importance.csv")


def shap_explanation_vis(model_name, model, X_train, X_test):
    explainer = shap.DeepExplainer(model, X_train[:100])
    # explainer = shap.Explainer(model)


    # shap.summary_plot(shap_values, X_test[100:101], show=False)
    # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    # test_numpy = np.swapaxes(np.swapaxes(X_test[100:101].cpu().numpy(), 1, -1), 1, 2)
    # shap.image_plot(shap_numpy, -test_numpy)

    no_tests = len(X_test)

    avg_negative_shap_value = np.zeros((267, 531))
    avg_positive_shap_value = np.zeros((267, 531))

    for i in range(no_tests):
        shap_values = np.array(explainer.shap_values(X_test[i:i+1]))

        negative_shap_value = shap_values[0][0][0]
        positive_shap_value = shap_values[1][0][0]

        avg_negative_shap_value += negative_shap_value
        avg_positive_shap_value += positive_shap_value

    avg_negative_shap_value /= no_tests
    avg_positive_shap_value /= no_tests

    np.save(model_name + "_avg_pos_shap_values.npy", avg_positive_shap_value)
    np.save(model_name + "_avg_neg_shap_values.npy", avg_negative_shap_value)


def explain():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Select type of Model", default='vanilla', type=str)
    parser.add_argument("--data", help="Select type data split", default='standard', type=str)
    args = parser.parse_args()

    global model
    standard = False
    print(args.data)

    if args.model == "Flat":
        model = FlatCNN().to(device)
        dataset = CancerDataset("biopsy/biopsy_data_vectors.pt", "biopsy/biopsy_labels.npy")
        model_hparams = hparams["FlatCNN"]
    elif args.model == "Hybrid":

        data_type = args.data
        if data_type == "standard":
            standard = True
            dataset = CancerDataset("biopsy/biopsy_data2.pt", "biopsy/biopsy_labels.npy")
        else:
            dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")

        print(standard)
        model = HybridCNN(standard).to(device)
        model_hparams = hparams["HybridCNN"]
    elif args.model == "Resnet":
        model_hparams = hparams["ResNet18"]
        model = get_ResNet(model_hparams["Dropout"]).to(device)
        data_type = args.data
        if data_type == "standard":
            standard = True
            dataset = CancerDataset("biopsy/biopsy_data2.pt", "biopsy/biopsy_labels.npy")
        else:
            dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")
    else:

        data_type = args.data
        if data_type == "standard":
            standard = True
            dataset = CancerDataset("biopsy/biopsy_data2.pt", "biopsy/biopsy_labels.npy")
        else:
            dataset = CancerDataset("biopsy/new_biopsy_data_matrices.pt", "biopsy/biopsy_labels.npy")

        model = VanillaCNN(standard).to(device)


    train_indices = np.load("indices/Train_indices_CnC.npy")
    test_indices = np.load("indices/Test_indices_CnC.npy")
    X_train = dataset.data.float().to(device)[train_indices]
    # X_test = dataset.data.float().to(device)[test_indices]
    X_test = dataset.data.float()[test_indices]
    print(X_test.shape)
    model.load_state_dict(torch.load(os.path.join("models", args.model + ".pt")))
    model.eval()

    # shap_explanation(args.model, model, X_train, X_test)
    shap_explanation_vis(args.model, model, X_train, X_test)
    # lime_explanation(args.model, model, X_test)


def summary_results():
    vanilla_standard_results = sorted(glob.glob("csv_results/Vanilla*standard*.csv"))
    vanilla_results = sorted([f for f in glob.glob("csv_results/Vanilla*.csv") if f not in vanilla_standard_results])
    flat_results = sorted(glob.glob("csv_results/Flat*.csv"))
    hybrid_standard_results = sorted(glob.glob("csv_results/Hybrid*standard*.csv"))
    hybrid_results = sorted([f for f in glob.glob("csv_results/Hybrid*.csv") if f not in hybrid_standard_results])

    vis_dir = "results_vis"

    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    resnet_standard_auroc_results = [0.764, 0.958, 0.950]
    resnet_reduced_auroc_results = [0.755, 0.904, 0.912]

    all_csv_files = [vanilla_standard_results, vanilla_results, flat_results, hybrid_standard_results, hybrid_results]
    all_labels = ["VanillaCNN_standard", "VanillaCNN_reduced", "FlatCNN", "HybridCNN_standard", "HybridCNN_reduced"]

    domains = ["cancer", "hospital", "random"]

    # for csv_files, label in zip(all_csv_files, all_labels):
    #     auroc_results = []
    #     for csv_file in csv_files:
    #         csv_results = pd.read_csv(csv_file, sep=",")
    #         auroc_results.append(float(csv_results["mean"][5]))
    #
    #     plt.scatter(domains, auroc_results, label=label)
    #
    # plt.scatter(domains, resnet_standard_auroc_results, label="Resnet18_standard")
    # plt.scatter(domains, resnet_reduced_auroc_results, label="Resnet18_reduced")
    #
    # plt.xlabel("Wykorzystywany podział danych")
    # plt.ylabel("Testowe AUROC")
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.tight_layout()
    # plt.savefig(os.path.join(vis_dir, "test_auroc.png"))

    resnet_standard_bal_results = [0.755, 0.857, 0.883]
    resnet_reduced_bal_results = [0.748, 0.843, 0.864]

    for csv_files, label in zip(all_csv_files, all_labels):
        auroc_results = []
        for csv_file in csv_files:
            csv_results = pd.read_csv(csv_file, sep=",")
            auroc_results.append(float(csv_results["mean"][4]))

        plt.scatter(domains, auroc_results, label=label)

    plt.scatter(domains, resnet_standard_bal_results, label="Resnet18_standard")
    plt.scatter(domains, resnet_reduced_bal_results, label="Resnet18_reduced")

    plt.xlabel("Wykorzystywany podział danych")
    plt.ylabel("Testowa zbilansowana dokładność")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "test_bal_acc.png"))


def summary_resources():
    vis_dir = "results_vis"
    # In seconds
    #                     "Vanilla_standard":
    #                     "Vanilla_reduced":
    #                     "Resnet_standard":
    #                     "Resnet_reduced":
    #                     "Flat":
    #                     "Hybrid_standard":
    #                     "Hybrid_reduced"
    times_pd = pd.read_csv("times.csv", sep=',')
    domain = ["Vanilla", "Vanilla_red", "Resnet", "Resnet_red", "Flat",
              "Hybrid", "Hybrid_red"]

    plt.scatter(domain, times_pd["Training epoch time"], color="red", label="Trening epoki")
    plt.scatter(domain, times_pd["Inference time"], color="green", label="Predykcja")

    plt.xlabel("Architektura")
    plt.ylabel("Czas trwania [s]")
    plt.legend()
    # plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "times.png"))


def summary_models():
    vis_dir = "results_vis"
    times_pd = pd.read_csv("times.csv", sep=',')
    domain = ["Flat", "Hybrid_red", "Hybrid", "Vanilla_red", "Resnet", "Resnet_red", "Vanilla"]
    values = [369250, 653570, 1656194, 5463586, 11171266, 11171266, 34082338]

    plt.bar(domain, values, color='maroon', width=0.4)

    plt.xlabel("Architektura")
    plt.ylabel("Liczba parametrów")
    plt.savefig(os.path.join(vis_dir, "models.png"))

def check_group(row):
    if row < 87:
        return 1
    elif row < 102:
        return 2
    elif row < 132:
        return 3
    elif row < 149:
        return 4
    elif row < 170:
        return 5.1
    elif row < 212:
        return 5.2
    elif row < 240:
        return 6.1
    else:
        return 6.2

def create_table(csv_filename, table_filename):
    vis_dir = "results_vis"
    shap_df = pd.read_csv(csv_filename, sep=',')

    with open(table_filename, "w") as f:
        i = 1
        for _, shap_row in shap_df.iterrows():
            group = check_group(int(shap_row["row"]))
            table_row = str(i) + "  &  " + str(shap_row["row"]) + "  &  " + str(shap_row["column"]) + "  &  "\
                        + str(group) + "  &  " + str(shap_row["shap_value"]) + "  \\\\" + "\n"
            f.write(table_row)
            i += 1


def shap_vis():
    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = 2

    resnet_shap_values = np.load("Resnet_avg_neg_shap_values.npy")
    vanilla_shap_values = np.load("vanilla_avg_neg_shap_values.npy")

    # Adds a subplot at the 1st position
    # fig.add_subplot(rows, columns, 1)

    # showing image
    c = plt.imshow(resnet_shap_values, cmap="inferno", vmin=0.0)

    # Adds a subplot at the 2nd position
    # fig.add_subplot(rows, columns, 2)

    # showing image
    # c = plt.imshow(vanilla_shap_values, cmap="inferno", vmin=0.0)

    plt.colorbar(c)
    plt.savefig("SHAP/Resnet_shap_neg_values.png")
    # plt.savefig("SHAP/Vanilla_shap_neg_values.png")
    plt.show()
    # print(shap_pos_values.shape)
    # print(shap_pos_values[0])
    # shap.image_plot(shap_pos_values)

if __name__ == '__main__':
    # train_and_save_model()
    # explain()
    shap_vis()
    # summary_resources()
    # summary_results()
    # summary_models()
    # create_table("SHAP/Resnet_shap_neg_importance.csv", "SHAP/Resnet_shap_neg_importance.txt")
    # create_table("SHAP/Resnet_shap_pos_importance.csv", "SHAP/Resnet_shap_pos_importance.txt")
    # create_table("SHAP/vanilla_shap_neg_importance.csv", "SHAP/vanilla_shap_neg_importance.txt")
    # create_table("SHAP/vanilla_shap_pos_importance.csv", "SHAP/vanilla_shap_pos_importance.txt")
