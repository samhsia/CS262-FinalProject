# Import packages
import argparse
import pickle

import random
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
from threading import Thread

from net import Net_CIFAR, Net_MNIST
import torch
import random

# Constants/configurations
ENCODING    = 'utf-8' # message encoding
BUFFER_SIZE = 2048 # fixed 2KB buffer size
PORT        = 1234 # base application port for leader server

SERVER_IP      = 'localhost' # REPLACE ME with output of ipconfig getifaddr en0
MAX_CLIENTS    = 100

profile_max = []
class SingleModelServer:
    def aggregate_gradients(self, round):
        sum_of_gradients = None
        self.num_malicious_agent = 0
        print("self.list_of_malicious_agent: ", self.list_of_malicious_agent)
        print("self.count_of_anomaly: ", self.count_of_anomaly)

        for sock_idx, sock in enumerate(self.active_sockets):
            Anomaly = False
            device_gradients = []
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
                if sock_idx in self.list_of_malicious_agent:
                    continue
                if round > 40:
                    for layer in device_gradients:
                        profile_max.append(torch.max(torch.abs(layer)))
                        # print(torch.max(torch.abs(layer)))
                        if torch.max(torch.abs(layer)) > self.normal_max:
                            # print(sock_idx, torch.max(torch.abs(layer)) )
                            Anomaly = True
                
            if not Anomaly:
                # Change to only add if you are a participating device
                if sum_of_gradients is None:
                    sum_of_gradients = device_gradients # first device gradient
                else:
                    sum_of_gradients = [x.data+device_gradients[i].data for i,x in enumerate(sum_of_gradients)]
            else:
                self.count_of_anomaly[sock_idx] += 1
                if self.count_of_anomaly[sock_idx] > 1:
                    self.list_of_malicious_agent.append(sock_idx)
                self.num_malicious_agent += 1
            
                
        for i, gradient in enumerate(sum_of_gradients):
            # *** sum_of_gradients[i] = gradient + NOISE
            sum_of_gradients[i] = gradient / (self.num_devices - self.num_malicious_agent) # change to participating devices
        
        return sum_of_gradients
    
    def run(self):
        for round in range(1, self.num_rounds+1):
            # Recieve and aggregate gradients
            aggregated_gradients = self.aggregate_gradients(round)
            print('Recieved and aggregated gradients')

            # Update model
            for variable, gradient in zip(self.model.parameters(), aggregated_gradients):
                variable.data.sub_(self.lr * gradient)
                # Add noise to the final server model
                # variable.data += random.random()
                # print(random.randint(0,9))
            print('Updated model')

            # Send model weights
            model_weights = [x.data for x in self.model.parameters()]
            for sock in self.active_sockets:
                sock.send(pickle.dumps(model_weights))
                sock.send('FINISH'.encode(encoding=ENCODING))
            print('Sent model weights')

            print('Round {}/{} finished'.format(round, self.num_rounds))

    def __init__(
        self, 
        lr,
        num_devices,
        num_rounds,
        perc_devices_per_round,
        enable_anomaly_detection
    ):
        self.lr                         = lr
        self.num_devices                = num_devices
        self.num_rounds                 = num_rounds
        self.perc_devices_per_round     = perc_devices_per_round # not implemented yet!
        
        # Noise injection and anomaly detection
        self.normal_max                 = 1 # normal maximum parameter value
        self.num_malicious_agent        = 0
        self.count_of_anomaly           = [0] * self.num_devices 
        self.list_of_malicious_agent    = [] 
        self.enable_anomaly_detection   = enable_anomaly_detection
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
    parser.add_argument("--enable-anomaly-detection", type=str, default=False, help='Enable anomaly detection to recover from noisy gradient update of malicious agent.')
    args = parser.parse_args()

    server = SingleModelServer(args.lr, args.num_devices, args.num_rounds, args.perc_devices_per_round, args.enable_anomaly_detection)
    server.run()

if __name__ == '__main__':
    main()