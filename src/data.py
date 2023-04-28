import numpy as np

import random
import ssl
import sys
import torch

ssl._create_default_https_context = ssl._create_unverified_context

def sample_dataset(dataset, num_samples, sampling_method):
    n_classes = len(dataset.keys())
    if sampling_method == 'iid':
        p_distr   = np.ones(n_classes)
    elif sampling_method == 'non-iid':
        p_distr = np.abs(np.random.randn(n_classes))
    else:
        sys.exit('Invalid sampling method.')
    p_distr  /= np.sum(p_distr)

    dkeys            = list(dataset.keys())
    index_to_class   = {i:dkeys[i] for i in range(n_classes)}
    selected_classes = np.random.choice(n_classes, num_samples, p=p_distr)

    images, labels = [], []
    for i in selected_classes:
        class_images, class_labels = dataset[index_to_class[i]]
        n_class_images = class_labels.shape[0]
        idx = random.randint(0, n_class_images-1)

        images.append(class_images[idx])
        labels.append(class_labels[idx])

    images = np.stack(images)
    labels = np.stack(labels)

    return images, labels

def get_dataset(dataset_name, num_samples, sampling_method):
    # Download and preprocess dataset
    if dataset_name == 'mnist':
        from torchvision.datasets import MNIST
        dataset = MNIST(root='.', train=True, download=True)
        X_train, Y_train = dataset.data, dataset.targets
        X_train = X_train.reshape((X_train.shape[0], -1))
        
        active_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        data = {}
        for label in active_labels:
            indices = np.where(Y_train == label)
            data[label] = (X_train[indices], Y_train[indices])

    elif dataset_name == 'cifar10':
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root='.', train=True, download=True)
        X_train, Y_train = dataset.data, dataset.targets
        X_train = np.swapaxes(X_train, 1, 3)
        Y_train = np.array(Y_train)

        active_labels = [0, 1, 2, 3, 4, 5, 6] # only do first 7 classes of CIFAR-10

        data = {}
        for label in active_labels:
            indices = np.where(Y_train == label)
            data[label] = (X_train[indices], Y_train[indices])
    else:
        sys.exit('Dataset not supported.')

    # Sample dataset
    images, labels = sample_dataset(data, num_samples, sampling_method)
    images, labels = torch.from_numpy(images).float(), torch.from_numpy(labels).long()

    return images, labels