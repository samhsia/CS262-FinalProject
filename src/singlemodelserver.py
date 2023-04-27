import numpy as np
import random

from net import Net

from FederatedServer import FederatedServer


class SingleModelServer(FederatedServer):

    def __init__(self, devices, lr=.0001, n_rounds=100, perc_devices_per_round=.5):
        super().__init__(devices, lr, n_rounds, perc_devices_per_round)

        # Create model
        self.model = Net()

    def evaluate_device_accuracies(self):
        return [x.evaluate(self.model) for x in self.devices]


    def run(self):
        for r in range(self.n_rounds):
            self.aggregate_stats()

            # Get participating devices
            participating_devices = [x for x in self.devices if random.uniform(0, 1) < self.perc_devices_per_round]

            # Aggregate gradients
            sum_of_gradients = None
            for d in participating_devices:

                # Compute + aggregate gradient
                gradients = d.gradient(self.model)
                if sum_of_gradients is None:
                    sum_of_gradients = gradients
                else:
                    sum_of_gradients = [x.data+gradients[i].data for i,x in enumerate(sum_of_gradients)]

            if sum_of_gradients is None:
                continue

            # Apply aggregated gradient
            for variable, grad in zip(self.model.parameters(), sum_of_gradients):
                variable.data.sub_((self.lr * grad) / len(participating_devices))

        for device in self.devices:
            device.n_compute = 0
