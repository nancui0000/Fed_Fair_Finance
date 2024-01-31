#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    # parser.add_argument('--local_bs', type=int, default=10,
    #                     help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='logistic_regression', help='model name (simple_MLP, '
                                                                                 'logistic_regression)')

    # other arguments
    parser.add_argument('--dataset', type=str, default='german', help="name \
                        of dataset (adult)")
    parser.add_argument('--num_classes', type=int, default=1, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--imbalance_method', type=str, default=None,
                        help='The type of SMOTE algorithm to use. Options are: smote, adasyn')
    parser.add_argument('--sens_drop_flag', type=int, default=0,
                        help='Whether to use sensitive attribute dropout')
    parser.add_argument('--aggregate_method', type=str, default='avg',
                        help='The method to aggregate the local models. Options are: avg, fair')
    parser.add_argument('--male_to_female_ratio', type=float, default=1.0, help='The ratio of male to female samples '
                                                                                'for each client.')
    parser.add_argument('--loss_type', type=str, default='bce', help='The type of loss function to use. Options are: '
                                                                     'bce, focal')
    parser.add_argument('--gamma', type=float, default=2.0, help='The gamma parameter for focal loss.')
    parser.add_argument('--alpha', type=float, default=0.25, help='The alpha parameter for focal loss.')
    parser.add_argument('--beta', type=float, default=0.5, help='The beta parameter for loss.')
    parser.add_argument('--fair_regulization', type=int, default=0, help='Whether to use fair regulization.')
    parser.add_argument('--tau', type=float, default=2, help='The tau parameter for fair regulization.')
    args = parser.parse_args()
    return args
