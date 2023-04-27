import numpy as np
import random

from net import Net


class FederatedServer(object):

    def __init__(self, devices, lr=.0001, n_rounds=100, perc_devices_per_round=.5, aggregate_data_interval=20):
        self.aggregate_data_interval = 20
        self.devices = devices
        self.n_rounds = n_rounds
        self.perc_devices_per_round = perc_devices_per_round
        self.lr = lr

        # Running stats
        self.accs = []
        self.mean_accs = []
        self.total_computes = []

    def evaluate_device_accuracies(self):
        assert(0)

    def reset(self):
        for device in self.devices:
            device.reset()

    def aggregate_stats(self):
        total_compute = np.sum([x.n_compute for x in self.devices])
        device_accuracies = self.evaluate_device_accuracies()
        mean_device_accuracy = np.mean(device_accuracies)
        class_distributions = [x.class_distr for x in self.devices]

        print("*"*50)
        print("Total Devices: %d" % len(device_accuracies))
        print("Total Mean Accuracy: %f" % mean_device_accuracy)
        print("Total Compute: %f" % total_compute)
        for i in range(len(device_accuracies)):
            print("Device: %d, Class Distr: %s, Acc: %f" % (i, str(class_distributions[i]), device_accuracies[i]))
        print("*"*50)

        self.mean_accs.append(mean_device_accuracy)
        self.accs.append(device_accuracies)
        self.total_computes.append(total_compute)

    def run(self):
        assert(0)
