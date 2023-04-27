import sys
import random
import argparse
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from multimodelserver import *
from singlemodelserver import *
from singlemodelserver_weightedagg import *

from UserEdgeDevice import *

from net import Net

from dataset import *
from plotting import *

if __name__=="__main__":
    print("Federated Serving")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", default=10, type=int)
    parser.add_argument("--n_points_per_device", default=5000, type=int)
    parser.add_argument("--n_sampled_points_per_update", default=100, type=int)
    parser.add_argument("--n_rounds", default=100, type=int)
    args = parser.parse_args()

    full_scan(args.n_devices, args.n_rounds, args.n_points_per_device, args.n_sampled_points_per_update)
    """
    # Load MNIST
    dataset = get_dataset()

    # Create edge devices
    devices = []
    for i in range(args.n_devices):
        sampled_dataset = sample_dataset_non_iid(dataset, args.n_points_per_device)
        #sampled_dataset = sample_dataset_one_class(dataset, args.n_points_per_device)
        #sampled_dataset = sample_dataset_iid(dataset, args.n_points_per_device)
        devices.append(UserEdgeDevice(sampled_dataset, sample=args.n_sampled_points_per_update))

    run_and_plot_methods([
        #SingleModelServer_weightedagg(devices, n_rounds=args.n_rounds),
        MultiModelServer(devices, n_rounds=args.n_rounds, lr=1e-4),
        SingleModelServer(devices, n_rounds=args.n_rounds, lr=1e-4)])

    # Create server and runs
    #server = SingleModelServer(devices)
    #server.run()

    #server = MultiModelServer(devices)
    #server.run()
    """
