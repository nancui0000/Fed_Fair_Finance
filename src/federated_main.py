#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch

from options import args_parser
from update import LocalModel, test_inference
from models import SimpleMLP, LogisticRegression
from utils import (get_dataset, average_weights, exp_details, convert_and_send_to_device, plot_curve,
                   record_experiment_results_with_args, fair_metric, aggregate_weights)


def print_metrics(prefix, accuracy, recall=None, f1=None, auc=None, parity=None, equality=None):
    print(f"|---- {prefix} Accuracy: {100 * accuracy:.2f}%")
    if recall is not None:
        print(f"|---- {prefix} Recall: {recall:.2f}")
    if f1 is not None:
        print(f"|---- {prefix} F1: {f1:.2f}")
    if auc is not None:
        print(f"|---- {prefix} AUC: {auc:.2f}")
    if parity is not None:
        print(f"|---- {prefix} Parity: {100 * parity:.2f}%")
    if equality is not None:
        print(f"|---- {prefix} Equality: {100 * equality:.2f}%")
    print("-----------------------------------------------")


def calculate_metrics(X, y, indices=None):
    """
    Calculate metrics for a given dataset
    :param X: dataset features
    :param y: dataset labels
    :param indices: indices of the dataset to be used
    :return: accuracy, loss, recall, f1, auc
    """
    if indices is not None:
        X_subset, y_subset = X[indices], y[indices]
    else:
        X_subset, y_subset = X, y
    return test_inference(args, global_model, X_subset, y_subset, device)


def calculate_fair_metrics(X, y, sens, indices=None):
    if indices is not None:
        X_subset, y_subset, sens_subset = X[indices], y[indices], sens[indices]
    else:
        X_subset, y_subset, sens_subset = X, y, sens
    return fair_metric(args, global_model, X_subset, y_subset, sens_subset)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    if args.gpu != 'cpu':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True)
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    print("device: ", device)

    set_seed(args.seed)

    # load dataset and user groups
    if args.dataset == 'adult':
        (X_train, y_train, X_test, y_test, sens_train, sens_test, X_train_with_sens, X_test_with_sens, sens_position,
         user_groups) = get_dataset(args)

    # Assuming X_train, y_train, X_test, y_test are either DataFrames or NumPy arrays
    X_train = convert_and_send_to_device(X_train, device)
    y_train = convert_and_send_to_device(y_train, device)
    X_test = convert_and_send_to_device(X_test, device)
    y_test = convert_and_send_to_device(y_test, device)
    if args.dataset == 'adult':
        sens_train = convert_and_send_to_device(sens_train, device)
        sens_test = convert_and_send_to_device(sens_test, device)
        X_train_with_sens = convert_and_send_to_device(X_train_with_sens, device)
        X_test_with_sens = convert_and_send_to_device(X_test_with_sens, device)

    # BUILD MODEL
    if args.model == 'simple_MLP':
        len_in = X_train.shape[1]
        global_model = SimpleMLP(dim_in=len_in, dim_out=args.num_classes)

    elif args.model == 'logistic_regression':
        len_in = X_train.shape[1]
        global_model = LogisticRegression(dim_in=len_in)

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    local_models = {}

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        local_sp, local_eo = [], []

        for idx in idxs_users:
            if idx not in local_models:
                local_model = LocalModel(args, X_train=X_train, y_train=y_train,
                                     idxs=user_groups[idx], device=device, len_in=len_in)
                local_model.to(device)
                local_models[idx] = local_model
            else:
                local_model = local_models[idx]

            local_model.model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            w, loss, local_model_temp = local_model.train(args, local_model.model, X_train[local_model.idx_train],
                                                          y_train[local_model.idx_train], sens_train[local_model.idx_train],
                                                          X_train_with_sens[local_model.idx_train]
                                                          if args.dataset == 'adult' else None,
                                                          sens_position if args.dataset == 'adult' else None)

            locall_sp, locall_eo = fair_metric(args, local_model_temp, X_train[local_model.idx_train],
                                               y_train[local_model.idx_train],
                                               sens_train[local_model.idx_train])

            local_sp.append(locall_sp.item())
            local_eo.append(locall_eo.item())
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

        # update global weights
        if args.aggregate_method == 'avg':
            global_weights = average_weights(local_weights)
        elif args.aggregate_method == 'fair':
            weights = [(1 / (local_sp[i] + local_eo[i] + 1E-5)) for i in range(len(local_sp))]
            # weights = [(1 / (local_sp[i] + 1E-5)) for i in range(len(local_sp))]
            weights = [float(i) / args.tau for i in weights]
            weights = torch.tensor(weights, device=device)

            weights = torch.softmax(weights, dim=-1)

            global_weights = aggregate_weights(local_weights, weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        for idx in idxs_users:
            local_model = local_models[idx]
            acc, loss = local_model.inference(global_model, X_train[local_model.idx_val], y_train[local_model.idx_val])
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    female_indices = torch.nonzero(sens_test == 0).squeeze().tolist()
    male_indices = torch.nonzero(sens_test == 1).squeeze().tolist()

    if args.dataset == 'adult':
        income_more_than_50k_indices = torch.nonzero(y_test == 1).squeeze().tolist()
        income_less_than_50k_indices = torch.nonzero(y_test == 0).squeeze().tolist()

    global_model.eval()

    # Test inference after completion of training
    test_metrics = calculate_metrics(X_test, y_test)
    test_metrics_female = calculate_metrics(X_test, y_test, female_indices)
    test_metrics_male = calculate_metrics(X_test, y_test, male_indices)

    # Fairness metrics calculation for specific datasets
    if args.dataset == 'adult' or args.dataset == 'german':
        test_fair_metrics = fair_metric(args, global_model, X_test, y_test, sens_test)

    # Print results
    print(f'\nResults after {args.epochs} global rounds of training:')
    print_metrics("Avg Train", train_accuracy[-1], None, None, None)
    print_metrics("Test", test_metrics[0], test_metrics[2], test_metrics[3], test_metrics[4],
                  *(test_fair_metrics if args.dataset in ['adult', 'german'] else (None, None)))

    print(f"|---- Sensitive Attribute Drop: {args.sens_drop_flag}")
    print(f"|---- Imbalance Method: {args.imbalance_method}")
    print(f"|---- Aggregate Method: {args.aggregate_method}")
    print(f"|---- Male to Female ratio: {args.male_to_female_ratio}")

    # Directory to save the objects
    save_dir = '../save/objects/'

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # File name for saving the objects
    file_name = '{}{}_{}_{}_C[{}]_iid[{}]_E[{}].pkl'.format(
        save_dir, args.dataset, args.model, args.epochs, args.frac, args.iid,
        args.local_ep)

    # Rest of your code to save the objects
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    if args.imbalance_method is not None:
        print(f'Imbalance method: {args.imbalance_method}')
    else:
        print(f'No imbalance method used')

    results = {
        "Dataset Name": args.dataset,
        "Training Accuracy (%)": 100 * train_accuracy[-1],
        "Test Accuracy (%)": 100 * test_metrics[0],
        "Test Recall": test_metrics[2],
        "Test F1": test_metrics[3],
        "Test AUC": test_metrics[4],
    }

    for index_sec, metrics in [('Females', test_metrics_female), ('Males', test_metrics_male)]:
        print_metrics(f"Test over {index_sec}", metrics[0], metrics[2], metrics[3], metrics[4])
        results[f"Test Accuracy ({index_sec}) (%)"] = 100 * metrics[0]
        results[f"Test Recall ({index_sec})"] = metrics[2]
        results[f"Test F1 ({index_sec})"] = metrics[3]
        results[f"Test AUC ({index_sec})"] = metrics[4]

    if args.dataset == 'adult':
        for index_sec, metrics in [('Income > 50k', calculate_metrics(X_test, y_test, income_more_than_50k_indices)),
                                   ('Income < 50k', calculate_metrics(X_test, y_test, income_less_than_50k_indices))]:
            print_metrics(f"Test over {index_sec}", metrics[0], metrics[2], metrics[3], metrics[4])
            results[f"Test Accuracy ({index_sec}) (%)"] = 100 * metrics[0]
            results[f"Test Recall ({index_sec})"] = metrics[2]
            results[f"Test F1 ({index_sec})"] = metrics[3]
            results[f"Test AUC ({index_sec})"] = metrics[4]

    if args.dataset == 'adult':
        results["Test Global Parity (%)"] = 100 * test_fair_metrics[0]
        results["Test Global Equality (%)"] = 100 * test_fair_metrics[1]

    record_experiment_results_with_args("experiment_results.xlsx", results, args)

    # Plot the Loss curve
    plot_curve(train_loss, 'Training Loss', 'Training Loss vs Communication rounds', '../save/loss_curve.png')

    # Plot the Average Accuracy curve
    plot_curve(train_accuracy, 'Average Accuracy', 'Average Accuracy vs Communication rounds',
               '../save/accuracy_curve.png')
