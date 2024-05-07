#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import torch
import os
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score
from torch import nn
import torch.nn.functional as F

from utils import resample_with_smote, resample_with_adasyn, fair_metric, fairness_loss
from models import SimpleMLP


class FocalLoss(nn.Module):
    def __init__(self, args, size_average=True):
        """
        :param alpha: Alpha value, weighting factor for class imbalance
        :param gamma: Gamma value, focusing parameter
        :param reduction: Reduction method to apply ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.size_average = size_average

        if args.gpu != 'cpu':
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            torch.use_deterministic_algorithms(True)
            self.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'

    def forward(self, input, target):
        # Use sigmoid instead of softmax for binary classification
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if target=1, else pt = 1-p
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.size_average:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)


class LocalModel(nn.Module):
    def __init__(self, args, X_train, y_train, idxs, device, len_in, epochs=10):
        super().__init__()
        self.model = SimpleMLP(dim_in=len_in, dim_out=args.num_classes).to(device)
        # self.args = args
        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.device = device
        self.epochs = epochs
        self.idx_train = list(idxs)[:int(0.9 * len(idxs))]
        self.idx_val = list(idxs)[int(0.9 * len(idxs)):]
        if args.loss_type == 'focal':
            self.criterion = FocalLoss(args)
        else:
            self.criterion = nn.BCELoss()
        self.X_train = X_train
        self.y_train = y_train
        self.loss = 0

    def train(self, args, model, data, labels, sens, data_with_sens, sens_position):
        sens_balanced = None
        data_balanced = None

        original_labels = labels

        if args.imbalance_method == 'smote':
            if args.fair_regulization == 1:
                data_with_sens, labels = resample_with_smote(data_with_sens.cpu(), labels.cpu())
                sens_balanced = data_with_sens[:, sens_position]
                if args.sens_drop_flag == 1:
                    data_balanced = np.concatenate(
                        (data_with_sens[:, :sens_position], data_with_sens[:, sens_position + 1:]), axis=1)
                elif args.sens_drop_flag == 0:
                    data_balanced = data_with_sens

            else:
                data, labels = resample_with_smote(data.cpu(), labels.cpu())
        elif args.imbalance_method == 'adasyn':
            if args.fair_regulization == 1:
                data_with_sens, labels = resample_with_adasyn(data_with_sens.cpu(), labels.cpu())
                sens_balanced = data_with_sens[:, sens_position]
                if args.sens_drop_flag == 1:
                    data_balanced = np.concatenate(
                        (data_with_sens[:, :sens_position], data_with_sens[:, sens_position + 1:]), axis=1)
                elif args.sens_drop_flag == 0:
                    data_balanced = data_with_sens
            else:
                data, labels = resample_with_adasyn(data.cpu(), labels.cpu())

        # Convert the resampled data and labels to PyTorch tensors and move them to GPU
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if sens_balanced is not None:
            sens_balanced = torch.from_numpy(sens_balanced).float()
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels).float()
        if data_balanced is not None:
            data_balanced = torch.from_numpy(data_balanced).float()
        if data_balanced is not None and sens_balanced is not None:
            data_balanced, sens_balanced, labels = data_balanced.to(self.device), sens_balanced.to(
                self.device), labels.to(self.device)
        else:
            data, labels = data.to(self.device), labels.to(self.device)
        if args.fair_regulization == 1:
            if sens_balanced is not None:
                if not isinstance(sens_balanced, torch.Tensor):
                    sens_balanced = torch.from_numpy(sens_balanced).float()
            else:
                if not isinstance(sens, torch.Tensor):
                    sens = torch.from_numpy(sens).float()
                sens = sens.to(self.device)

        model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            if data_balanced is not None:
                output = model(data_balanced).squeeze()
            else:
                output = model(data).squeeze()
            labels = labels.squeeze()

            if args.fair_regulization == 1:
                self.parity, self.equality = fairness_loss(model, data, original_labels, sens)

                self.loss = (1 - args.beta) * self.criterion(output, labels) + args.beta * (
                        self.parity + self.equality)

            else:
                self.loss = self.criterion(output, labels)
            self.loss.backward()
            self.optimizer.step()

        return model.state_dict(), self.loss.item(), model

    def inference(self, model, data, labels):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        criterion = nn.BCEWithLogitsLoss().to(self.device)

        # Inference
        outputs = model(data)
        loss = criterion(outputs.squeeze(), labels.squeeze())

        # Prediction
        pred_labels = (outputs > 0.5).float().view(-1)
        correct = torch.sum(torch.eq(pred_labels, labels)).item()

        accuracy = correct / len(labels)
        loss = loss.item()
        return accuracy, loss


def test_inference(args, model, X_test, y_test, device, label=None):
    """
    Returns the test accuracy and loss.

    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.BCELoss().to(device)

    # Inference
    outputs = model(X_test)
    loss = criterion(outputs.squeeze(), y_test.squeeze())

    unique_labels = torch.unique(y_test)

    if len(unique_labels) == 1 and unique_labels[0] == 0:
        pred_labels = (outputs <= 0.5).float().view(-1)
        y_test = 1 - y_test
    else:
        pred_labels = (outputs > 0.5).float().view(-1)

    # # Prediction

    correct = torch.sum(torch.eq(pred_labels, y_test)).item()

    if label is not None:
        condition_pos = (pred_labels == label)
        condition_true = (y_test == label)

        # Ensure conditions are tensors of dtype torch.bool
        if not isinstance(condition_pos, torch.Tensor):
            condition_pos = torch.tensor(condition_pos, dtype=torch.bool, device=device)
        if not isinstance(condition_true, torch.Tensor):
            condition_true = torch.tensor(condition_true, dtype=torch.bool, device=device)

        true_positive = torch.sum(torch.logical_and(condition_pos, condition_true)).item()
        condition_neg = (pred_labels != label)
        condition_false = (y_test != label)

        # Same checks and conversion for negative conditions
        if not isinstance(condition_neg, torch.Tensor):
            condition_neg = torch.tensor(condition_neg, dtype=torch.bool, device=device)
        if not isinstance(condition_false, torch.Tensor):
            condition_false = torch.tensor(condition_false, dtype=torch.bool, device=device)

        true_negative = torch.sum(torch.logical_and(condition_neg, condition_false)).item()
        accuracy = (true_positive + true_negative) / y_test.numel()
    else:
        accuracy = correct / len(y_test)
    loss = loss.item()

    # Convert the model outputs and the test labels to NumPy arrays
    # outputs_numpy = outputs.cpu().detach().numpy()
    labels_numpy = y_test.cpu().numpy()
    preds_numpy = pred_labels.cpu().numpy()

    if len(unique_labels) > 1:  # Check if there are at least two unique values
        auc = roc_auc_score(labels_numpy, preds_numpy)
    else:
        auc = np.nan
    # Calculate recall, F1-score, and AUC
    if label is not None:
        recall = recall_score(labels_numpy, preds_numpy, labels=[label], average=None)
        f1 = f1_score(labels_numpy, preds_numpy, labels=[label], average=None)
        precision = precision_score(labels_numpy, preds_numpy, labels=[label], average=None)

        return accuracy, loss, recall[0], f1[0], auc, precision[0]
    else:
        recall = recall_score(labels_numpy, preds_numpy)
        f1 = f1_score(labels_numpy, preds_numpy)
        precision = precision_score(labels_numpy, preds_numpy)

        return accuracy, loss, recall, f1, auc, precision
