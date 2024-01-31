#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def give_me_some_credit_iid(dataset, num_users):
    """
    Sample I.I.D. client data from Adult dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def german_non_iid(dataset, num_users, male_to_female_ratio):
    """
    Sample non-IID client data from Adult dataset based on gender ratio.
    :param dataset: The dataset to draw samples from.
    :param num_users: The number of clients.
    :param male_to_female_ratio: The desired ratio of male to female samples for each client.
    :return: A dictionary where keys are client IDs and values are lists of indices.
    """
    male_data = dataset[dataset['Gender'] == 0]
    female_data = dataset[dataset['Gender'] == 1]

    male_indices = male_data.index.tolist()
    female_indices = female_data.index.tolist()

    np.random.shuffle(male_indices)
    np.random.shuffle(female_indices)

    dict_users = {}
    male_count = 0
    female_count = 0

    for i in range(num_users):
        if len(male_indices) - male_count == 0:
            male_to_female_ratio = 0
        elif len(female_indices) - female_count == 0:
            male_to_female_ratio = male_to_female_ratio + 1

        # *****************************************************************************************
        # num_samples = len(dataset) // (num_users - i)
        num_samples = len(dataset) // num_users
        # *****************************************************************************************
        # print("\n num_samples: ", num_samples)
        # print("\n male_to_female_ratio: ", male_to_female_ratio)
        num_males = int((male_to_female_ratio / (male_to_female_ratio + 1)) * num_samples)
        num_females = num_samples - num_males

        sampled_males = male_indices[male_count: male_count + min(num_males, len(male_indices) - male_count)]
        male_count += len(sampled_males)

        sampled_females = female_indices[
                          female_count: female_count + min(num_females, len(female_indices) - female_count)]
        female_count += len(sampled_females)

        dict_users[i] = sampled_males + sampled_females

    return dict_users


def german_iid(dataset, num_users):
    """
    Sample I.I.D. client data from Adult dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def creditcard_iid(dataset, num_users):
    """
    Sample I.I.D. client data from Adult dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def adult_non_iid(dataset, num_users, male_to_female_ratio):
    """
    Sample non-IID client data from Adult dataset based on gender ratio.
    :param dataset: The dataset to draw samples from.
    :param num_users: The number of clients.
    :param male_to_female_ratio: The desired ratio of male to female samples for each client.
    :return: A dictionary where keys are client IDs and values are lists of indices.
    """
    male_data = dataset[dataset['sex'] == 0]
    female_data = dataset[dataset['sex'] == 1]

    male_indices = male_data.index.tolist()
    female_indices = female_data.index.tolist()

    np.random.shuffle(male_indices)
    np.random.shuffle(female_indices)

    dict_users = {}
    male_count = 0
    female_count = 0

    for i in range(num_users):
        if len(male_indices) - male_count == 0:
            male_to_female_ratio = 0
        elif len(female_indices) - female_count == 0:
            male_to_female_ratio = male_to_female_ratio + 1

        # *****************************************************************************************
        # num_samples = len(dataset) // (num_users - i)
        num_samples = len(dataset) // num_users
        # *****************************************************************************************
        # print("\n num_samples: ", num_samples)
        # print("\n male_to_female_ratio: ", male_to_female_ratio)
        num_males = int((male_to_female_ratio / (male_to_female_ratio + 1)) * num_samples)
        num_females = num_samples - num_males

        sampled_males = male_indices[male_count: male_count + min(num_males, len(male_indices) - male_count)]
        male_count += len(sampled_males)

        sampled_females = female_indices[
                          female_count: female_count + min(num_females, len(female_indices) - female_count)]
        female_count += len(sampled_females)

        dict_users[i] = sampled_males + sampled_females

    return dict_users


def adult_iid(dataset, num_users):
    """
    Sample I.I.D. client data from Adult dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users





if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    # d = mnist_noniid(dataset_train, num)
