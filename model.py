import torch
import torch.nn as nn
import yaml
import torch.nn.functional as F

def prepare_model(configs):
    # todo: Add the code to prepare the model here.

    with open(configs, 'r') as file:
        configs = yaml.safe_load(file)

    class CIFAR_Model(nn.Module):
        def __init__(self, configs):
            super(CIFAR_Model, self).__init__()

            in_channels = configs['model']['in_channels']
            num_layers = configs['model']['num_layers']
            num_classes = configs['model']['num_classes']


            self.conv1 = nn.Conv2d(in_channels, in_channels*2, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = CIFAR_Model(configs)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    # This is the main function to test the model's components
    print("Testing model components")
    model = prepare_model('configs/config.yaml')
    print(model)
    print(model.conv1)
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")