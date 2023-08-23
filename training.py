import copy
import time
import datetime
import torch
import wandb
import numpy as np
import pandas as pd

from torch import nn
from tqdm import tqdm
from torchsummary import summary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models import VanillaCNN, FlatCNN, HybridCNN
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_bal_accs, train_aucs, train_losses = np.array([]), np.array([]), np.array([])
val_bal_accs, val_aucs, val_losses = np.array([]), np.array([]), np.array([])
test_bal_acc, test_auc, test_loss = np.array([]), np.array([]), np.array([])

def train_model(model, num_epochs, dataloaders, optimizer, criterion, scheduler, dataset_sizes, fold):
    best_model = copy.deepcopy(model.state_dict())
    # validation balanced accuracy
    best_auc = 0.0
    best_bal_acc = 0.0
    best_epoch = 0

    global device
    global train_bal_accs, train_aucs, train_losses
    global val_bal_accs, val_aucs, val_losses

    wandb.watch(model)


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

            epoch_loss = running_loss / dataset_sizes[phase]
            auc = roc_auc_score(all_labels, all_outputs[:, 1])
            bal_acc = balanced_accuracy_score(all_labels, all_preds)
            # auc = roc_auc_score(all_labels, all_outputs, average='weighted', multi_class='ovo')

            # log_file.write('{} Loss: {:.4f} Auc: {:.4f}\n'.format(phase, epoch_loss, auc))
            # wandb.log({f"{phase} Fold {fold} Loss": loss, f"{phase} Fold {fold} AUC": auc, f"{phase} Fold {fold} Bal Acc": bal_acc, "epoch": epoch})
            if phase == "train":
                train_bal_accs[epoch, fold] = bal_acc
                train_aucs[epoch, fold] = auc
                train_losses[epoch, fold] = epoch_loss
            else:
                val_bal_accs[epoch, fold] = bal_acc
                val_aucs[epoch, fold] = auc
                val_losses[epoch, fold] = epoch_loss

            if phase == 'val' and (auc > best_auc):
                # deep copy the model
                best_epoch = epoch
                best_auc = auc
                best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)

    return model, best_epoch


def test(model, dataloader, size, criterion, fold):
    '''
        Function for test model.
    '''
    global test_bal_acc, test_auc, test_loss

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    all_outputs = []
    all_preds = []
    all_labels = []
    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            max_outputs = outputs[:, 1]
            loss = criterion(outputs, labels)

        all_outputs.extend(outputs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    all_outputs = np.array(all_outputs, dtype=float)
    all_preds = np.array(all_preds, dtype=int)
    all_labels = np.array(all_labels, dtype=int)


    test_loss[fold] = running_loss / size
    test_auc[fold] = roc_auc_score(all_labels, all_outputs[:, 1])
    test_bal_acc[fold] = balanced_accuracy_score(all_labels, all_preds)
    # test_auc = roc_auc_score(all_labels, all_outputs, average='weighted', multi_class='ovo')

    # log_file.write('Test Loss: {:.4f} Auc: {:.4f}\n'.format(test_loss, test_auc))

    # wandb.log({"Test Loss": loss, "Test AUC": test_auc, "Test Bal Acc": test_bal_acc})

    # return test_loss, test_auc

def report_metrics(best_epochs):
    global train_bal_accs, train_aucs, train_losses
    global val_bal_accs, val_aucs, val_losses
    global test_bal_acc, test_auc, test_loss

    mean_best_train_bal_acc = np.mean([train_bal_accs[best_epoch][i] for i, best_epoch in enumerate(best_epochs)])
    mean_best_train_auc = np.mean([train_aucs[best_epoch][i] for i, best_epoch in enumerate(best_epochs)])

    mean_best_val_bal_acc = np.mean([val_bal_accs[best_epoch][i] for i, best_epoch in enumerate(best_epochs)])
    mean_best_val_auc = np.mean([val_aucs[best_epoch][i] for i, best_epoch in enumerate(best_epochs)])

    std_best_train_bal_acc = np.std([train_bal_accs[best_epoch][i] for i, best_epoch in enumerate(best_epochs)])
    std_best_train_auc = np.std([train_aucs[best_epoch][i] for i, best_epoch in enumerate(best_epochs)])

    std_best_val_bal_acc = np.std([val_bal_accs[best_epoch][i] for i, best_epoch in enumerate(best_epochs)])
    std_best_val_auc = np.std([val_aucs[best_epoch][i] for i, best_epoch in enumerate(best_epochs)])

    std_test_bal_acc = np.std(test_bal_acc)
    std_test_auc = np.std(test_auc)

    run_results = pd.DataFrame({
        "train_bal_acc": mean_best_train_bal_acc,
        "train_auc": mean_best_train_auc,
        "val_bal_acc": mean_best_val_bal_acc,
        "val_auc": mean_best_val_auc,
        "test_bal_acc": 0,
        "test_auc": 0
    }, index=[0])

    # wandb.log({"Mean Test Loss": np.mean(test_loss), "Mean Test AUC": np.mean(test_auc),
    #            "Mean Test Bal Acc": np.mean(test_bal_acc)})
    # wandb.log({"Test Bal Acc std": std_test_bal_acc, "Test AUC std": std_test_auc})
    #
    # wandb.log({"Mean Train AUC": mean_best_train_auc, "Mean Train Bal Acc": mean_best_train_bal_acc,
    #            "Train Bal Acc std": std_best_train_bal_acc, "Train AUC std": std_best_train_auc})
    #
    # wandb.log({"Mean Val AUC": mean_best_val_auc, "Mean Val Bal Acc": mean_best_val_bal_acc,
    #            "Val Bal Acc std": std_best_val_bal_acc, "Val AUC std": std_best_val_auc})

    for i in range(100):
        wandb.log({f"Train Loss": np.mean(train_losses[i]), f"Train AUC": np.mean(train_aucs[i]), f"Train Bal Acc": np.mean(train_bal_accs[i]), "epoch": i + 1})
        wandb.log({f"Val Loss": np.mean(val_losses[i]), f"Val  AUC": np.mean(val_aucs[i]), f"Val Bal Acc": np.mean(val_bal_accs[i]), "epoch": i + 1})

    return run_results

def training_loop(standard, model_type, class_weights, datasets, y_train, hparams):
    no_folds = 5
    skf = StratifiedKFold(n_splits=no_folds)

    global train_bal_accs, train_aucs, train_losses
    global val_bal_accs, val_aucs, val_losses
    global test_bal_acc, test_auc, test_loss

    train_bal_accs = np.zeros((hparams["num_of_epochs"], no_folds))
    train_aucs = np.zeros((hparams["num_of_epochs"], no_folds))
    train_losses = np.zeros((hparams["num_of_epochs"], no_folds))

    val_bal_accs = np.zeros((hparams["num_of_epochs"], no_folds))
    val_aucs = np.zeros((hparams["num_of_epochs"], no_folds))
    val_losses = np.zeros((hparams["num_of_epochs"], no_folds))

    test_bal_acc = np.zeros((no_folds,))
    test_auc = np.zeros((no_folds,))
    test_loss = np.zeros((no_folds,))

    best_epochs = []



    # for fold, (train_ids, val_ids) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
    for fold, (train_ids, val_ids) in tqdm(enumerate(skf.split(np.zeros(len(y_train)), y_train)),
                                           desc=f"{no_folds}-fold cross-validation"):
        # Get a model and put it onto a device

        if model_type == "Flat":
            model = FlatCNN().to(device)
        elif model_type == "Hybrid":
            model = HybridCNN(standard).to(device)
        else:
            model = VanillaCNN(standard).to(device)

        # Loss function and optimizer
        loss_fn = nn.CrossEntropyLoss(class_weights.to(device))
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        # and SteLR
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=hparams["step_size"],
                                               gamma=hparams["gamma"])

        datasets["val"] = torch.utils.data.Subset(datasets["non-test"], val_ids)
        datasets["train"] = torch.utils.data.Subset(datasets["non-test"], train_ids)

        train_size = len(datasets["train"])
        val_size = len(datasets["val"])
        test_size = len(datasets["test"])

        # print("Train size = ", train_size)
        # print("val size = ", val_size)

        dataset_sizes = {'train': train_size, 'val': val_size}

        # Get dataloaders
        trainloader = DataLoader(datasets["train"], batch_size=hparams["train_batch_size"])
        validloader = DataLoader(datasets["val"], batch_size=hparams["val_batch_size"])
        testloader = DataLoader(datasets["test"], batch_size=hparams["val_batch_size"])

        dataloaders = {'train': trainloader, 'val': validloader}

        # train
        fold_model, fold_epoch = train_model(model,
                                hparams["num_of_epochs"],
                                dataloaders,
                                optimizer,
                                loss_fn,
                                exp_lr_scheduler,
                                dataset_sizes,
                                fold)

        best_epochs.append(fold_epoch)

        # test
        test(fold_model, testloader, test_size, loss_fn, fold)

    run_results = report_metrics(best_epochs)

    run_results["test_bal_acc"] = np.mean(test_bal_acc)
    run_results["test_auc"] = np.mean(test_auc)

    return run_results

    #torch.save(best_model, "best_model.pt")