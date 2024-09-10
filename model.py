import torch
import torch.nn as nn
import yaml
import torch.nn.functional as F

def prepare_model(configs):
    # todo: Add the code to prepare the model here.

    class CIFAR_Model(nn.Module):
        def __init__(self, configs):
            super(CIFAR_Model, self).__init__()

            # not sure how to choose the proper amount of out_channels and layers
            in_channels = configs.model.in_channels
            out_channels = configs.model.out_channels
            num_layers = configs.model.num_layers
            num_classes = configs.model.num_classes
            num_pool_layers = configs.model.num_pool_layers

            self.conv_layers = nn.ModuleList()
            self.bn_layers = nn.ModuleList()
            self.pool = nn.MaxPool2d( 2, 2)

            for i in range(num_layers):
                if i == 0:
                    out_channels = out_channels
                else:
                    out_channels = out_channels * 2  # Double the channels with each layer

                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
                self.conv_layers.append(conv_layer)
                self.bn_layers.append(nn.BatchNorm2d(out_channels))  # Batch norm after each conv layer
                in_channels = out_channels

            feature_size = 32 // (2 ** num_pool_layers)
            flattened_size = out_channels * feature_size * feature_size

            # todo: not sure how to choose the dimensions for the fully connected layers, also not sure how many fully connected layers to use
            self.fc1 = nn.Linear(flattened_size, flattened_size // 16 )
            self.bn_fc1 = nn.BatchNorm1d(flattened_size // 16)
            self.fc2 = nn.Linear(flattened_size // 16, flattened_size // 32)
            self.bn_fc2 = nn.BatchNorm1d(flattened_size // 32)
            self.fc3 = nn.Linear(flattened_size // 32, num_classes)

        def forward(self, x):
            for i, conv_layer in enumerate(self.conv_layers):
                x = F.relu(self.bn_layers[i](conv_layer(x)))
                # pool every other layer
                if i % 2 == 0:
                    x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.bn_fc1(self.fc1(x)))
            x = F.relu(self.bn_fc2(self.fc2(x)))
            x = self.fc3(x)
            return x

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