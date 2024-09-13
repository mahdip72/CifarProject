import torch
import torch.nn as nn
import yaml
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # skip connection layer
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # use 1x1 convolution to match the dimensions
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the skip connection
        out = F.relu(out)
        return out


class CIFAR_Model(nn.Module):
    def __init__(self, configs):
        super(CIFAR_Model, self).__init__()

        # updating to resnet model

        # not sure how to choose the proper amount of out_channels and layers
        in_channels = configs.model.in_channels
        out_channels = configs.model.out_channels
        num_layers = configs.model.num_layers
        num_classes = configs.model.num_classes

        # initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # create residual layers
        self.res_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                res_layer = self._make_layer(out_channels, out_channels, num_blocks=2, stride=1)
            else:
                res_layer = self._make_layer(out_channels, out_channels*2, num_blocks=2, stride=2)
                out_channels *= 2
            self.res_layers.append(res_layer)

        # global average pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # final fully connected layer
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        # initial convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))
        # residual layers
        for res_layer in self.res_layers:
            x = res_layer(x)
        # global average pooling
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # final fully connected layer
        x = self.fc(x)
        return x

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

def prepare_model(configs):
    # todo: Add the code to prepare the model here.

    model = CIFAR_Model(configs)
    return model



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    # This is the main function to test the model's components
    print("Testing model components")

    from box import Box
    config_file_path = 'configs/config.yaml'
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    configs = Box(config_data)

    model = prepare_model(configs)
    print(model)

    dummy_input = torch.randn(1, 3, 32, 32)

    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")

    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")