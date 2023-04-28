# Import packages
import argparse
import numpy as np
import pickle

from socket import socket, AF_INET, SOCK_STREAM
import torch

from data import get_dataset
from net import Net

# Constants/configurations
ENCODING    = 'utf-8' # message encoding
BUFFER_SIZE = 2048 # fixed 500KB buffer size
PORT        = 1234 # fixed application port

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
        aggregated_gradients = []
        while True:
            msg = self.client.recv(BUFFER_SIZE)
            if not msg:
                break
            aggregated_gradients.append(msg)
        aggregated_gradients = pickle.loads(b"".join(aggregated_gradients))
        
        # Update model
        for variable, gradient in zip(self.model.parameters(), aggregated_gradients):
            variable.data.sub_(self.lr * gradient)

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

    def run(self):
        return
        # Compute gradients
        # gradients = self.compute_gradient()

        # Send gradients
        # self.client.send(pickle.dumps(gradients))

        # Recieve aggregated gradients and update model
        # self.update_model()



    def __init__(
        self,
        dataset_name,
        num_samples_per_device,
        num_samples_per_update,
        sampling_method,
        server_ip,
    ):
        self.dataset_name           = dataset_name
        self.num_samples_per_device = num_samples_per_device
        self.num_samples_per_update = num_samples_per_update
        self.sampling_method        = sampling_method
        self.server_ip              = server_ip

        self.num_compute = 0 # number of times calculating gradient

        # Create model
        self.model = Net()

        # Create local dataset
        self.dataset = get_dataset(self.dataset_name, self.num_samples_per_device, self.sampling_method)

        # Connect to server
        self.client = socket(family=AF_INET, type=SOCK_STREAM)
        self.client.connect((self.server_ip, PORT))

        # Compute gradients
        gradients = self.compute_gradient()
        print('Computed gradients')
        
        # Send gradients
        self.client.send(pickle.dumps(gradients))
        print('Sent gradients')

        # Recieve aggregated gradients and update model
        # self.update_model()
        print ('Recieved aggregated gradients and updated model')

        accuracy = self.evaluate_model()
        print(accuracy)

def main():
    print("********** Federated Serving: CLIENTS **********")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default='cifar10', help='Dataset')
    parser.add_argument("--lr", type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--num-devices", type=int, default=1, help='Number of devices.')
    parser.add_argument("--num-rounds", type=int, default=100, help='Number of rounds.')
    parser.add_argument("--num-samples-per-device", type=int, default=5000, help='Number of training samples per device.')
    parser.add_argument("--num-samples-per-update", type=int, default=100, help='Number of training samples per device update.')
    parser.add_argument("--sampling-method", type=str, default='iid', help='Dataset sampling method')
    parser.add_argument("--server-ip", type=str, default='100.90.130.16', help='Server IP address')
    args = parser.parse_args()

    for client_num in range(1, args.num_devices+1):
        client = SingleModelClient(args.dataset_name, args.num_samples_per_device, args.num_samples_per_update, args.sampling_method, args.server_ip)
        print('CLIENT ({}/{}) connected to server @ {}:{}'.format(client_num, args.num_devices, args.server_ip, PORT))
    for client_num in range(1, args.num_devices+1):
        client.run()

if __name__ == '__main__':
    main()