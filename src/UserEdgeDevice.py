import sys
import random
import argparse
import numpy as np
from tensorflow.python.keras.datasets import mnist, cifar10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class UserEdgeDevice(object):

    def __init__(self, dataset, sample=10):
        self.dataset = dataset
        images, labels = self.dataset
        images, labels = torch.from_numpy(images).float(), torch.from_numpy(labels).long()
        self.dataset = images, labels
        self.n_compute = 0
        self.sample = sample

        # Compute label distribution
        label_set = sorted(list(set(labels.numpy().flatten())))
        distr = {}
        total = 0
        for label in label_set:
            distr[label] = np.sum((labels.numpy()==label))
            total += distr[label]
        distr = {k:v/total for k,v in distr.items()}
        self.class_distr = distr

    def reset(self):
        self.n_compute = 0

    def evaluate(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            images, labels = self.dataset
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        acc = 100. * correct / labels.shape[0]

        return acc

    def gradient(self, model):
        criterion = nn.CrossEntropyLoss()
        model.train()

        model.zero_grad()
        images, labels = self.dataset
        #sample_indices = np.random.randint(0, labels.shape[0], size=(self.sample,))
        sample_indices = np.random.choice(list(range(labels.shape[0])), size=(self.sample,), replace=False)
        images, labels = images[sample_indices], labels[sample_indices]
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

        self.n_compute += 1

        return [x.grad for x in model.parameters()]
