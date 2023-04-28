# Import packages
import argparse
import pickle

import random
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
from threading import Thread

from net import Net

# Constants/configurations
ENCODING    = 'utf-8' # message encoding
BUFFER_SIZE = 2048 # fixed 2KB buffer size
PORT        = 1234 # base application port for leader server

SERVER_IP      = '100.90.130.16' # REPLACE ME with output of ipconfig getifaddr en0
MAX_CLIENTS    = 100

class SingleModelServer:
    def aggregate_gradients(self):
        sum_of_gradients = None
        for sock in self.active_sockets:
            device_gradients = []
            while True:
                msg = sock.recv(BUFFER_SIZE)
                if not msg:
                    break
                device_gradients.append(msg)
            device_gradients = pickle.loads(b"".join(device_gradients))
            # Change to only add if you are a participating device
            if sum_of_gradients is None:
                sum_of_gradients = device_gradients # first device gradient
            else:
                sum_of_gradients = [x.data+device_gradients[i].data for i,x in enumerate(sum_of_gradients)]
        for i, gradient in enumerate(sum_of_gradients):
            sum_of_gradients[i] = gradient / self.num_devices # change to participating devices
        
        return sum_of_gradients

    def __init__(
        self, 
        lr,
        num_devices,
        num_rounds,
        perc_devices_per_round
    ):
        self.lr                     = lr
        self.num_devices            = num_devices
        self.num_rounds             = num_rounds
        self.perc_devices_per_round = perc_devices_per_round # not implemented yet!
        
        # Running stats
        self.accs           = []
        self.mean_accs      = []
        self.total_computes = []

        # Create model
        self.model = Net()

        # Create server socket
        self.server = socket(AF_INET, SOCK_STREAM)
        self.server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1) # allow for multiple clients

        # Remember to run 'ipconfig getifaddr en0' and update SERVER_IP
        self.server.bind((SERVER_IP, PORT))
        self.server.listen(MAX_CLIENTS) # accept up to MAX_CLIENTS active connections

        self.active_sockets = [] # running list of active client sockets

        # Connect to all client devices
        for device_num in range(1, self.num_devices+1):
            sock, client_addr = self.server.accept()
            self.active_sockets.append(sock) # update active sockets list
            print ('SERVER: Device {}/{} connected @ {}:{}'.format(device_num, self.num_devices, client_addr[0], client_addr[1]))

        # Recieve and aggregate gradients
        # participating_devices_sockets = [x for x in self.active_sockets if random.uniform(0, 1) <= self.perc_devices_per_round]
        aggregated_gradients = self.aggregate_gradients()
        print('Recieved and aggregated gradients')

        # Send aggregated gradients
        for sock in self.active_sockets:
            sock.send(pickle.dumps(aggregated_gradients))
        print('Sent aggregated gradients')

        print('Finished initializing server!')

        ### ***** ###

    '''
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
    '''


def main():
    print("********** Federated Serving: SERVER **********")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--num-devices", type=int, default=1, help='Number of devices.')
    parser.add_argument("--num-rounds", type=int, default=100, help='Number of rounds.')
    parser.add_argument("--perc-devices-per-round", type=float, default=1.0, help='Target percentage of devices participating in each round.')
    args = parser.parse_args()

    SingleModelServer(args.lr, args.num_devices, args.num_rounds, args.perc_devices_per_round)
 
    # from plotting import full_scan
    # full_scan(args.num_devices, args.num_rounds, args.num_points_per_device, args.num_sampled_points_per_update)

if __name__ == '__main__':
    main()