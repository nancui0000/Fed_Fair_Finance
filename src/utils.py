#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
# from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
# from sampling import cifar_iid, cifar_noniid
from sampling import adult_iid, adult_non_iid
from sampling import creditcard_iid
from sampling import german_iid, german_non_iid
from sampling import give_me_some_credit_iid
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter


def fairness_loss(model, data, original_labels, sens):
    """
    Calculate fairness metrics: Demographic Parity (DP) and Equal Opportunity (EO).

    Args:
    - model: The current model being trained.
    - data: The input data tensor.
    - original_labels: The ground truth labels tensor.
    - sens: The sensitive attribute tensor.

    Returns:
    - parity: Fairness loss based on demographic parity.
    - equality: Fairness loss based on equal opportunity.
    """

    def calculate_distribution(probs, mask):
        """Compute the distribution of positive instances under a given mask."""
        N_positive = torch.sum(probs * mask)
        total_masked = torch.sum(mask.float()) + 1E-5

        return N_positive / total_masked

    # Obtain continuous predictions from the model
    node_preds = model(data)
    pred_probs = torch.sigmoid(node_preds).view(-1)  ### Convert logits to probabilities
    # pred_probs = node_preds.view(-1)

    # Assume male and female masks are binary 1/0 tensors of the same size as sens
    s1_mask = sens.eq(1).float()
    s0_mask = sens.eq(0).float()

    # DP (Demographic Parity)
    s1_dist_dp = calculate_distribution(pred_probs, s1_mask)  ### Use probabilities directly
    s0_dist_dp = calculate_distribution(pred_probs, s0_mask)  ### Use probabilities directly
    fairness_loss_dp = torch.abs(s1_dist_dp - s0_dist_dp)
    # fairness_loss_dp = (s1_dist_dp - s0_dist_dp).pow(2).sqrt()

    # EO (Equal Opportunity)
    correct_probs = pred_probs * original_labels.float()  ### Only consider positive instances using probabilities
    s1_dist_eo = calculate_distribution(correct_probs, s1_mask)
    s0_dist_eo = calculate_distribution(correct_probs, s0_mask)
    fairness_loss_eo = torch.abs(s1_dist_eo - s0_dist_eo)
    # fairness_loss_eo = (s1_dist_eo - s0_dist_eo).pow(2).sqrt()

    return fairness_loss_dp, fairness_loss_eo


def aggregate_weights(w, norm_w):
    """
    Returns the weighted weights.
    """
    w_new = copy.deepcopy(w[0])
    for key in w_new.keys():
        temp = 0
        for i in range(0, len(w)):
            # w_new[key] = torch.add(w_new[key], torch.mul(w[i][key], norm_w[i]))
            # temp += w[i][key] * norm_w[i]
            temp += torch.mul(w[i][key], norm_w[i])
        w_new[key] = temp
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_new


def is_imbalanced(labels, threshold=0.1):
    """
    Function to determine if a dataset is imbalanced.

    Parameters
    ----------
    labels : array-like
        The labels of the dataset.
    threshold : float, optional (default=0.1)
        The ratio threshold for minority class over majority class.
        If the ratio is below this threshold, the dataset is considered imbalanced.

    Returns
    -------
    bool
        Returns True if the dataset is imbalanced, False otherwise.
    """
    # Calculate the class distribution of the labels
    class_distribution = Counter(labels)

    # Calculate the ratio of the minority class over the majority class
    ratio = min(class_distribution.values()) / max(class_distribution.values())

    # Return True if the ratio is below the threshold, False otherwise
    return ratio < threshold


def fair_metric(args, model, X, labels, sens):
    # Inference
    outputs = model(X)

    # Prediction
    pred_y = (outputs > 0.5).float().view(-1)

    true_y = labels
    # pred_y = pred_y
    idx_s0 = sens.eq(0)
    idx_s1 = sens.eq(1)

    idx_s0_y1 = torch.bitwise_and(idx_s0, true_y.eq(1))
    idx_s1_y1 = torch.bitwise_and(idx_s1, true_y.eq(1))

    if torch.sum(idx_s0) > 1 and torch.sum(idx_s1) > 1:
        parity = abs(torch.sum(pred_y[idx_s0]) / (torch.sum(idx_s0)) - torch.sum(pred_y[idx_s1]) / (torch.sum(idx_s1)))
    elif torch.sum(idx_s0) == 0 and torch.sum(idx_s1) > 1:
        parity = abs(torch.sum(pred_y[idx_s1]) / (torch.sum(idx_s1)))
    elif torch.sum(idx_s1) == 0 and torch.sum(idx_s0) > 1:
        parity = abs(torch.sum(pred_y[idx_s0]) / (torch.sum(idx_s0)))
    else:
        parity = torch.tensor(0., device='cuda:0')
    if torch.sum(idx_s0_y1) > 1 and torch.sum(idx_s1_y1) > 1:
        equality = abs(
            torch.sum(pred_y[idx_s0_y1]) / (torch.sum(idx_s0_y1)) - torch.sum(pred_y[idx_s1_y1]) / (
                torch.sum(idx_s1_y1)))
    elif torch.sum(idx_s0_y1) == 0 and torch.sum(idx_s1_y1) > 1:
        equality = abs(torch.sum(pred_y[idx_s1_y1]) / (torch.sum(idx_s1_y1)))
    elif torch.sum(idx_s0_y1) > 1 and torch.sum(idx_s1_y1) == 0:
        equality = abs(torch.sum(pred_y[idx_s0_y1]) / (torch.sum(idx_s0_y1)))
    else:
        equality = torch.tensor(0., device='cuda:0')

    return parity, equality


def resample_with_smote(data, labels, k_neighbors_list=[5, 2, 1]):
    for k_neighbors in k_neighbors_list:
        try:
            sm = SMOTE(k_neighbors=k_neighbors)
            data_resampled, labels_resampled = sm.fit_resample(data, labels)
            return data_resampled, labels_resampled
        except ValueError:
            pass
    else:
        print('SMOTE failed to generate synthetic samples. Using original dataset instead.')
    return data, labels


def resample_with_adasyn(data, labels, n_neighbors_list=[5, 2, 1]):
    for n_neighbors in n_neighbors_list:
        try:
            adasyn = ADASYN(n_neighbors=n_neighbors)
            data_resampled, labels_resampled = adasyn.fit_resample(data, labels)
            return data_resampled, labels_resampled
        except ValueError:
            pass
        except RuntimeError:
            pass
    else:
        print('ADASYN failed, using SMOTE instead.')
        return resample_with_smote(data, labels)


def record_experiment_results_with_args(file_name, results, args):
    # Check if the file exists
    if os.path.exists(file_name):
        # If the file exists, load the existing dataframe
        df = pd.read_excel(file_name)
    else:
        # If the file doesn't exist, create a new dataframe with columns for all parameters
        columns = ["Model Name", "Dataset Name", "Training Accuracy (%)", "Test Accuracy (%)"] + list(vars(args).keys())
        df = pd.DataFrame(columns=columns)

    # Combine the results with the parameters
    combined_results = {**results, **vars(args)}

    # Convert any Tensor objects on GPU to CPU
    for key, value in combined_results.items():
        if torch.is_tensor(value) and value.is_cuda:
            # combined_results[key] = value.cpu()
            combined_results[key] = value.item()

    # Append the new results
    df = df.append(combined_results, ignore_index=True)

    # Save the dataframe to an excel file
    df.to_excel(file_name, index=False)


def plot_curve(data, ylabel, title, filename):
    plt.figure(figsize=(8, 6))  # Set the size of the figure
    plt.title(title, fontsize=16)  # Set the title and font size
    plt.plot(range(len(data)), data, linestyle='-', marker='o', color='b', linewidth=2,
             label=ylabel)  # Set the line style and color
    plt.ylabel(ylabel, fontsize=14)  # Set the y-axis label and font size
    plt.xlabel('Communication Rounds', fontsize=14)  # Set the x-axis label and font size
    plt.legend(fontsize=12)  # Add the legend and set the font size
    plt.grid(True)  # Show gridlines
    plt.show()  # Show the figure
    # plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the image with adjusted resolution and bounding box
    plt.close()


def convert_and_send_to_device(data, device):
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    elif isinstance(data, np.ndarray):
        data_array = data
    elif isinstance(data, pd.Series):
        data_array = data.values
    else:
        print("Data type: ", type(data))
        raise ValueError("Unsupported data type. Only DataFrame and ndarray are supported.")

    tensor_data = torch.tensor(data_array).float().to(device)
    return tensor_data


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    def clean_data(data):
        missing_values = data.isnull().sum()
        print("Missing values statistics：\n", missing_values)

        for column in data.columns:
            if data[column].isnull().any():
                if data[column].dtype == 'float64' or data[column].dtype == 'int64':
                    # Numeric columns fill in missing values with the mean
                    data[column].fillna(data[column].mean(), inplace=True)
                else:
                    # Text columns mark missing values as "unknown"
                    data[column].fillna("unknown", inplace=True)
        return data

    if args.dataset == 'adult':
        data = pd.read_csv('../data/{}.csv'.format(args.dataset))
        if args.dataset == 'adult':
            # Fill '?' values with the most frequent value in each column
            for col in ['workclass', 'occupation', 'native.country']:
                data[col] = data[col].replace('?', data[col].mode()[0])

            data = clean_data(data)

            data_encoded = pd.get_dummies(data,
                                          columns=['workclass', 'education', 'marital.status', 'occupation',
                                                   'relationship',
                                                   'race', 'native.country'])

            # Convert the income feature to binary
            data_encoded['income'] = data_encoded['income'].map({'<=50K': 0, '>50K': 1})

            # Convert the sex feature to binary
            data_encoded['sex'] = data_encoded['sex'].map({'Male': 0, 'Female': 1})

            # Initialize the StandardScaler
            scaler = StandardScaler()

            # Scale the numerical features
            numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
            data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

            # Split the data into training set and test set
            data_train, data_test = train_test_split(data_encoded, test_size=0.2, random_state=42)

            # # Reset the index of data_train
            data_train = data_train.reset_index(drop=True)
            data_test = data_test.reset_index(drop=True)

            # Separate the features and the target variable
            X_train = data_train.drop('income', axis=1)
            y_train = data_train['income']
            X_test = data_test.drop('income', axis=1)
            y_test = data_test['income']

            sens_position = X_train.columns.get_loc('sex')

            if args.sens_drop_flag == 1:
                sens_train = X_train['sex']
                sens_test = X_test['sex']
                X_train_with_sens = X_train
                X_test_with_sens = X_test
                X_train = X_train.drop('sex', axis=1)
                X_test = X_test.drop('sex', axis=1)
            elif args.sens_drop_flag == 0:
                sens_train = X_train['sex']
                sens_test = X_test['sex']
                X_train_with_sens = X_train
                X_test_with_sens = X_test

            if args.iid == 1:
                user_groups = adult_iid(data_train, args.num_users)
            elif args.iid == 0:
                user_groups = adult_non_iid(data_train, args.num_users, args.male_to_female_ratio)

            return (X_train, y_train, X_test, y_test, sens_train, sens_test, X_train_with_sens, X_test_with_sens,
                    sens_position, user_groups)
        else:
            raise ValueError("Unsupported dataset. Only adult is supported.")


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:\n')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    # print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
