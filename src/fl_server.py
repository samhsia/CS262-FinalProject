# Import packages
import argparse
import numpy as np
import pickle

from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR

from net import Net_CIFAR, Net_MNIST
import sys
from time import time
import torch

# Constants/configurations
ENCODING    = 'utf-8' # message encoding
BUFFER_SIZE = 2048 # fixed 2KB buffer size
PORT        = 1234 # base application port for leader server

SERVER_IP   = 'localhost' # REPLACE ME with output of ipconfig getifaddr en0
MAX_CLIENTS = 100
PRINT_INFO  = True

# Variables used for profiling
t_overhead    = [] # latency overhead of range detector
profile_max = [] # maximum gradient value recieved from client

class SingleModelServer:
    def aggregate_gradients(self, round):
        sum_of_gradients = None

        for sock_idx, sock in enumerate(self.active_sockets):
            device_gradients = []
            anomaly_deteced  = False # by default no anomaly detected

            while True:
                msg = sock.recv(BUFFER_SIZE)
                if not msg:
                    print('ERROR: client disconnected')
                    break
                if len(msg) != 2048:
                    try:
                        if msg[-6:].decode(encoding=ENCODING) == 'FINISH':
                            device_gradients.append(msg[:-6])
                            break
                    except:
                        device_gradients.append(msg)
                        continue
                device_gradients.append(msg)
            device_gradients = pickle.loads(b"".join(device_gradients))
            
            # Anomaly detection
            if self.enable_anomaly_detection == "True":
                # If device is already identified as malicious, skip the gradients from this device
                if sock_idx in self.list_of_malicious_agents:
                    continue
                
                # Enable anomaly detection past round 40
                if round > 40:
                    t_overhead_start = time()
                    for count, layer in enumerate(device_gradients):
                        profile_max.append(torch.max(torch.abs(layer)))
                        if torch.max(torch.abs(layer)) > self.normal_max[count]:
                            anomaly_deteced = True
                    t_overhead_end = time()
                    t_overhead.append(t_overhead_end - t_overhead_start)

                # Between rounds 30 and 40, record the upper bounds of each layer
                elif round > 30:
                    for count, layer in enumerate(device_gradients):
                        max = torch.max(torch.abs(layer)) * 1.1
                        if max > self.normal_max[count]:
                            self.normal_max[count] = max

            # This socket is not deemed a malicious agent
            if not anomaly_deteced:
                # Change to only add if you are a participating device
                if sum_of_gradients is None:
                    sum_of_gradients = device_gradients # first device gradient
                else:
                    sum_of_gradients = [x.data+device_gradients[i].data for i,x in enumerate(sum_of_gradients)]
            
            # This socket is a newly discovered malicious agent
            else:
                self.count_of_anomaly[sock_idx] += 1
                if self.count_of_anomaly[sock_idx] > 1:
                    self.list_of_malicious_agents.append(sock_idx)
                    self.num_malicious_agents += 1

        if PRINT_INFO:
            print("self.num_malicious_agents: {}".format(self.num_malicious_agents))
            print("self.list_of_malicious_agents: {}".format(self.list_of_malicious_agents))
            print("self.count_of_anomaly: {}".format(self.count_of_anomaly))
        
        if sum_of_gradients is None:
            return None # all clients marked as malicious
        for i, gradient in enumerate(sum_of_gradients):
            sum_of_gradients[i] = gradient / (self.num_devices - self.num_malicious_agents) # change to participating devices
        
        return sum_of_gradients
    
    def run(self):
        for round in range(1, self.num_rounds+1):
            # Recieve and aggregate gradients
            aggregated_gradients = self.aggregate_gradients(round)
            
            if aggregated_gradients:
                if PRINT_INFO:
                    print('Recieved and aggregated gradients')

                # Update model
                for variable, gradient in zip(self.model.parameters(), aggregated_gradients):
                    variable.data.sub_(self.lr * gradient)
                if PRINT_INFO:
                    print('Updated model')

                # Send model weights
                model_weights = [x.data for x in self.model.parameters()]
                for sock in self.active_sockets:
                    sock.send(pickle.dumps(model_weights))
                    sock.send('FINISH'.encode(encoding=ENCODING))
                if PRINT_INFO:
                    print('Sent model weights')
                    print('Round {}/{} finished'.format(round, self.num_rounds))
            else:
                return

    def __init__(
        self, 
        lr,
        num_devices,
        num_rounds,
        perc_devices_per_round,
        enable_anomaly_detection
    ):
        self.lr                     = lr
        self.num_devices            = num_devices
        self.num_rounds             = num_rounds
        self.perc_devices_per_round = perc_devices_per_round # not implemented yet!
        
        
        # Noise injection and anomaly detection
        self.enable_anomaly_detection = enable_anomaly_detection
        number_of_layers              = 4
        number_of_param_layers        = 2 * number_of_layers
        self.normal_max               = [0] * number_of_param_layers # per-layer range detector
        self.num_malicious_agents     = 0
        self.list_of_malicious_agents = []
        self.count_of_anomaly         = [0] * self.num_devices 
        
        # Running stats
        self.accs           = []
        self.mean_accs      = []
        self.total_computes = []
        
        # Create model
        self.model = Net_MNIST()

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
    parser.add_argument("--lr", type=float, default=0.01, help='Learning rate')
    parser.add_argument("--num-devices", type=int, default=10, help='Number of devices.')
    parser.add_argument("--num-rounds", type=int, default=100, help='Number of rounds.')
    parser.add_argument("--perc-devices-per-round", type=float, default=1.0, help='Target percentage of devices participating in each round.')
    parser.add_argument("--enable-anomaly-detection", type=str, default='False', help='Enable anomaly detection to recover from noisy gradient update of malicious agent.')
    args = parser.parse_args()

    server = SingleModelServer(args.lr, args.num_devices, args.num_rounds, args.perc_devices_per_round, args.enable_anomaly_detection)
    
    t_start = time()
    server.run()
    t_end = time()
    print('Total experiment time: {} s'.format(t_end - t_start))

    # Overhead of anomaly detection
    if args.enable_anomaly_detection == 'True':
        print('Range Detector Overhead: {} s'.format(np.sum(t_overhead)))

if __name__ == '__main__':
    main()