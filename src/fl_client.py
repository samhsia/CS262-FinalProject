# Import packages
import argparse
import numpy as np
import pickle

from socket import socket, AF_INET, SOCK_STREAM
import sys
import torch

from data import get_dataset
from net import Net_CIFAR, Net_MNIST
import torch
import random

# Constants/configurations
ENCODING    = 'utf-8' # message encoding
BUFFER_SIZE = 2048 # fixed 2KB buffer size
PORT        = 1234 # fixed application port
PRINT_INFO  = True

class SingleModelClient:
    def compute_gradient(self):
        self.model.train()
        self.model.zero_grad()

        images, labels = self.dataset
        sample_indices = np.random.choice(list(range(labels.shape[0])), size=(self.num_samples_per_update, ), replace=False)
        images, labels = images[sample_indices], labels[sample_indices]
        
        outputs = self.model(images)
        loss    = self.model.loss_fn(outputs, labels)
        loss.backward()

        self.num_compute += 1

        return [x.grad for x in self.model.parameters()]
    
    def update_model(self):
        # Recieve updated gradient
        new_model_weights = []
        while True:
            msg = self.client.recv(BUFFER_SIZE)
            if not msg:
                print('ERROR: server disconnected')
                break
            if len(msg) != 2048:
                try:
                    if msg[-6:].decode(encoding=ENCODING) == 'FINISH':
                        new_model_weights.append(msg[:-6])
                        break
                except:
                    new_model_weights.append(msg)
                    continue
            new_model_weights.append(msg)
        # Try to update model. Return True if successful.
        try:
            new_model_weights = pickle.loads(b"".join(new_model_weights))
            
            # Update model
            for variable, new_weights in zip(self.model.parameters(), new_model_weights):
                variable.data = new_weights # *** + NOISE
            return True
        except:
            return False

    def evaluate_model(self):
        self.model.eval()
        correct_samples = 0
        with torch.no_grad():
            images, labels = self.dataset
            outputs        = self.model(images)
            predictions    = outputs.argmax(dim=1, keepdim=True)
            
            correct_samples += predictions.eq(labels.view_as(predictions)).sum().item()

        acc = 100. * correct_samples / labels.shape[0]
        return acc

    def __init__(
        self,
        device_num,
        dataset_name,
        lr,
        num_samples_per_device,
        num_samples_per_update,
        sampling_method,
        server_ip,
    ):
        self.device_num             = device_num
        self.dataset_name           = dataset_name
        self.lr                     = lr
        self.num_samples_per_device = num_samples_per_device
        self.num_samples_per_update = num_samples_per_update
        self.sampling_method        = sampling_method
        self.server_ip              = server_ip
        self.num_compute = 0 # number of times calculating gradient

        # Create model
        self.model = Net_MNIST()

        # Create local dataset
        self.dataset = get_dataset(self.dataset_name, self.num_samples_per_device, self.sampling_method)

        # Connect to server
        self.client = socket(family=AF_INET, type=SOCK_STREAM)
        self.client.connect((self.server_ip, PORT))

def main():
    print("********** Federated Serving: devices **********")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default='mnist', help='Dataset')
    parser.add_argument("--lr", type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--num-devices", type=int, default=10, help='Number of devices.')
    parser.add_argument("--num-rounds", type=int, default=100, help='Number of rounds.')
    parser.add_argument("--num-samples-per-device", type=int, default=5000, help='Number of training samples per device.')
    parser.add_argument("--num-samples-per-update", type=int, default=100, help='Number of training samples per device update.')
    parser.add_argument("--sampling-method", type=str, default='iid', help='Dataset sampling method')
    parser.add_argument("--server-ip", type=str, default='localhost', help='Server IP address')
    parser.add_argument("--enable-malicious-agent", type=str, default='False', help='Enable malicious agents.')
    parser.add_argument("--num-malicious-agents", type=int, default=1, help='Number of malicious agents.')
    parser.add_argument("--noise-level", type=int, default=10, help='Multiple of random.random (0~1)')
    parser.add_argument("--noise-type", type=int, default=0, help='0: white noise (random noise for each value); 1: uniform noise (one random noise value for each layer)')
    args = parser.parse_args()

    devices        = []
    all_accuracies = []

    for device_num in range(args.num_devices):
        devices.append(SingleModelClient(device_num+1, args.dataset_name, args.lr, args.num_samples_per_device, args.num_samples_per_update, args.sampling_method, args.server_ip))
        print('CLIENT ({}/{}) connected to server @ {}:{}'.format(device_num+1, args.num_devices, args.server_ip, PORT))

    iteration = 1
    while True:
        if PRINT_INFO:
            print("Iteration {}".format(iteration))
        # Compute gradients
        gradients = []
        for device_num in range(args.num_devices):
            tmp_grad = devices[device_num].compute_gradient()
            # Malicious agent: noisy gradient update
            if args.enable_malicious_agent == "True":
                if iteration > 40:
                    # Malicious agent assignments starts from 0
                    if device_num < args.num_malicious_agents:
                        for layer_grad in tmp_grad:
                            if args.noise_type == 0:
                                layer_grad += (2 * args.noise_level * torch.rand(layer_grad.shape) - args.noise_level)
                            else:
                                layer_grad += (2 * args.noise_level * random.random() - args.noise_level)      
            gradients.append(tmp_grad)
        if PRINT_INFO:
            print('All devices computed gradients')

        # Send gradients
        for device_num in range(args.num_devices):
            try:
                devices[device_num].client.send(pickle.dumps(gradients[device_num]))
                devices[device_num].client.send('FINISH'.encode(encoding=ENCODING))
            except:
                print('Server shut down. Finished training.')
                break
        if PRINT_INFO:
            print('All devices sent gradients')

        # Recieve model weights and update model
        for device_num in range(args.num_devices):
            update_success = devices[device_num].update_model()
            if not update_success:
                break
        if PRINT_INFO:
            print ('All devices recieved model weights and updated model')

        # Accuracy evaluation
        accuracies = []
        for device_num in range(args.num_devices):
            accuracies.append(devices[device_num].evaluate_model())
        all_accuracies.append(accuracies)
        if PRINT_INFO:
            print('Mean Acc: {}%'.format(np.mean(accuracies)))
            print('All Accs: {}%'.format(accuracies))
        iteration += 1

        if iteration > 100:
            break

    all_accuracies = np.array(all_accuracies)
    with open('accuracies_1agent_1noise_ad.npy', 'wb') as outfile:
        np.save(outfile, all_accuracies)
            
if __name__ == '__main__':
    main()