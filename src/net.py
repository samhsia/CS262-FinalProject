# Import packages
import torch.nn as nn
import torch.nn.functional as F

class Net_CIFAR(nn.Module):
    def __init__(self):
        super(Net_CIFAR, self).__init__()
        self.conv1   = nn.Conv2d(3, 6, 5)
        self.conv2   = nn.Conv2d(6, 16, 5)
        self.fc1     = nn.Linear(16*5*5, 120)
        self.fc2     = nn.Linear(120, 84)
        self.fc3     = nn.Linear(84, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
class Net_MNIST(nn.Module):
    def create_mlp(self):
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[2], self.output_size),
            nn.LogSoftmax(dim=1)
        )

    def __init__(self):
        super(Net_MNIST, self).__init__()
        self.input_size   = 784
        self.hidden_sizes = [512, 256, 128]
        self.output_size  = 10
        self.mlp          = self.create_mlp()
        self.loss_fn      = nn.NLLLoss()

    def forward(self, x):
        return self.mlp(x)