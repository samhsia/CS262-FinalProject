import random
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_dataset():

    # Load mnist
    # (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = mnist.load_data()
    (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = cifar10.load_data()
    # print(X_test_orig.shape)
    # print(Y_train_orig.shape)

    # Flatten
    #X_train_orig = X_train_orig.reshape(X_train_orig.shape[0], -1)

    #X_train_orig = X_train_orig.astype(float)
    #X_train_orig /= 255 WAT?

    X_train_orig = np.swapaxes(X_train_orig, 1, 3)
    Y_train_orig = Y_train_orig.flatten()


    # Get distinct labels
    labels = set(Y_train_orig.flatten().tolist())
    labels = set([0, 1, 2, 3, 4, 5, 6])

    # Separate out classes
    d = {}
    for label in labels:
        indices = np.where(Y_train_orig == label)
        d[label] = (X_train_orig[indices], Y_train_orig[indices])

    return d

def sample_dataset_iid(dataset, n):
    n_classes = len(dataset.keys())
    p_distr = np.ones(n_classes)
    p_distr /= np.sum(p_distr)

    dkeys = list(dataset.keys())
    index_to_class = {i:dkeys[i] for i in range(n_classes)}
    selected_classes = np.random.choice(n_classes, n, p=p_distr)
    images, labels = [], []
    for i in selected_classes:
        class_images, class_labels = dataset[index_to_class[i]]
        n_class_images = class_labels.shape[0]
        indx = random.randint(0, n_class_images-1)

        images.append(class_images[indx])
        labels.append(class_labels[indx])

    images = np.stack(images)
    labels = np.stack(labels)

    return images, labels

def sample_dataset_non_iid(dataset, n):
    n_classes = len(dataset.keys())
    p_distr = np.abs(np.random.randn(n_classes))
    p_distr /= np.sum(p_distr)

    dkeys = list(dataset.keys())
    index_to_class = {i:dkeys[i] for i in range(n_classes)}
    selected_classes = np.random.choice(n_classes, n, p=p_distr)
    images, labels = [], []
    for i in selected_classes:
        class_images, class_labels = dataset[index_to_class[i]]
        n_class_images = class_labels.shape[0]
        indx = random.randint(0, n_class_images-1)

        images.append(class_images[indx])
        labels.append(class_labels[indx])

    images = np.stack(images)
    labels = np.stack(labels)

    return images, labels