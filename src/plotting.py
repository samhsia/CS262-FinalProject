# Import packages
import matplotlib.pyplot as plt

from dataset import *

from singlemodelserver import *

from UserEdgeDevice import *

def run_and_plot_methods(servers, fname):
    plt.cla()
    for server in servers:
        name = server.__class__.__name__
        server.run()
        #plt.plot(server.total_computes, server.mean_accs, label=name)
        all_compute = []
        accs = []
        for acc, c in zip(server.accs, server.total_computes):
            for aa in acc:
                all_compute.append(c)
                accs.append(aa)
        plt.scatter(all_compute, accs, label=name)

    #plt.ylim(0, 100)
    plt.xlabel("Total Compute Over All Edge Devices")
    plt.ylabel("Mean Device Accuracy")
    plt.legend(loc="best")
    plt.savefig("%s.png" % fname)

def full_scan(n_devices, n_rounds, n_points_per_device, n_sampled_points_per_update):
    dataset = get_dataset()
    dataset_sampling_methods = [sample_dataset_iid]
    server_methods = [SingleModelServer]
    server_kwargs = {"n_rounds" : n_rounds,
                     "lr" : 1e-3}

    for meth in dataset_sampling_methods:
        devices = []
        for i in range(n_devices):
            sampled_dataset = meth(dataset, n_points_per_device)
            devices.append(UserEdgeDevice(sampled_dataset, sample=n_sampled_points_per_update))

        servers = [server_method(devices, **server_kwargs) for server_method in server_methods]
        run_and_plot_methods(servers, meth.__name__)