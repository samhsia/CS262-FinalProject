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

SERVER_IP      = 'localhost' # REPLACE ME with output of ipconfig getifaddr en0
MAX_CLIENTS    = 100

class SingleModelServer:
    def aggregate_gradients(self):
        sum_of_gradients = None
        for sock in self.active_sockets:
            device_gradients = []
            while True:
                msg = sock.recv(BUFFER_SIZE)
                if not msg:
                    print('ERROR: client disconnected')
                    break
                if len(msg) != 2048:
                    if msg[-6:].decode(encoding=ENCODING) == 'FINISH':
                        device_gradients.append(msg[:-6])
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
    
    def run(self):
        for round in range(1, self.num_rounds+1):
            # Recieve and aggregate gradients
            # participating_devices_sockets = [x for x in self.active_sockets if random.uniform(0, 1) <= self.perc_devices_per_round]
            aggregated_gradients = self.aggregate_gradients()
            print('Recieved and aggregated gradients')

            # Send aggregated gradients
            for sock in self.active_sockets:
                sock.send(pickle.dumps(aggregated_gradients))
                sock.send('FINISH'.encode(encoding=ENCODING))
            print('Sent aggregated gradients')

            print('Round {}/{} finished'.format(round, self.num_rounds))

    def __init__(
        self, 
        num_devices,
        num_rounds,
        perc_devices_per_round
    ):
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

        print('Finished initializing server!')

def main():
    print("********** Federated Serving: SERVER **********")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-devices", type=int, default=1, help='Number of devices.')
    parser.add_argument("--num-rounds", type=int, default=100, help='Number of rounds.')
    parser.add_argument("--perc-devices-per-round", type=float, default=1.0, help='Target percentage of devices participating in each round.')
    args = parser.parse_args()

    server = SingleModelServer(args.num_devices, args.num_rounds, args.perc_devices_per_round)
    server.run()

    # from plotting import full_scan
    # full_scan(args.num_devices, args.num_rounds, args.num_points_per_device, args.num_sampled_points_per_update)

if __name__ == '__main__':
    main()