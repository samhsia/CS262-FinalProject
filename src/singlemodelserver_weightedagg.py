import numpy as np
import random
import tensorflow as tf
from net import Net

from FederatedServer import FederatedServer


class SingleModelServer_weightedagg(FederatedServer):

    def __init__(self, devices, lr=.0001, n_rounds=100, perc_devices_per_round=.5):
        self.devices = devices
        self.n_rounds = n_rounds
        self.perc_devices_per_round = perc_devices_per_round
        self.lr = lr
        self.reweight = 0.5
        self.decay = 0.99
        super().__init__(devices, lr, n_rounds, perc_devices_per_round)

        # Create model
        self.model = Net()

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
            gradient_abs_avg = []
            gradients_list = []

            for d in participating_devices:

                # Compute + aggregate gradient
                gradients = d.gradient(self.model)
                # Get the average of absolute gradient value of participating_devices
                avg = []
                for i,x in enumerate(gradients):
                    avg.append(np.average(np.absolute(x.numpy())))
                gradient_abs_avg.append(np.average(avg))
                # Record the gradient value of participating_devices
                gradients_list.append(gradients)
                if sum_of_gradients is None:
                    sum_of_gradients = gradients
                else:
                    sum_of_gradients = [x.data+gradients[i].data for i,x in enumerate(sum_of_gradients)]
            if sum_of_gradients is None:
                continue
            # Assume that the device with maximum average of absolute gradient value is underrepresented
            sorted_gradient_abs_avg = np.sort(gradient_abs_avg)
            underrepresented_device_index = gradient_abs_avg.index(sorted_gradient_abs_avg[len(gradient_abs_avg)-1])
            # Scale the gradient value of the underrepresented device
            sum_of_gradients = [x.data+gradients_list[underrepresented_device_index][i].data * (len(participating_devices) * self.reweight) for i,x in enumerate(sum_of_gradients)]
            # Apply aggregated gradient
            for variable, grad in zip(self.model.parameters(), sum_of_gradients):
                # The summation of gradients divided by participating_devices + the scaling factor
                variable.data.sub_((self.lr * grad) / (len(participating_devices) + len(participating_devices) * self.reweight))
        self.reweight = self.decay * self.reweight
        for device in self.devices:
            device.n_compute = 0
