import numpy as np
import random
import torch.nn as nn

from net import Net

from FederatedServer import FederatedServer


class MultiModelServer(FederatedServer):

    def __init__(self, devices, lr=.0001, n_rounds=100, perc_devices_per_round=.5):
        super().__init__(devices, lr, n_rounds, perc_devices_per_round)

        # Create model
        self.model = Net()

        # Create model
        self.models = [Net() for i in range(len(self.devices))]

    def evaluate_device_accuracies(self):
        return [x.evaluate(self.models[i]) for i,x in enumerate(self.devices)]


    def run(self):

        for r in range(self.n_rounds):
            self.aggregate_stats()

            # Get participating devices
            participating_devices_indices = [i for i in range(len(self.devices)) if random.uniform(0, 1) < self.perc_devices_per_round]
            participating_devices = [self.devices[i] for i in participating_devices_indices]

            # Aggregate gradients
            for indx, d in zip(participating_devices_indices, participating_devices):

                # Compute + aggregate gradients individually
                gradients = d.gradient(self.models[indx])

                # Apply aggregated gradient
                for variable, grad in zip(self.models[indx].parameters(), gradients):
                    variable.data.sub_(self.lr * grad)
        for device in self.devices:
            device.n_compute = 0
