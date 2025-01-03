import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout)  # Dropout after activation


        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
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
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the skip connection
        out = F.relu(out)
        return out


class CIFARModel(nn.Module):
    def __init__(self, configs):
        super(CIFARModel, self).__init__()

        in_channels = configs.model.in_channels
        out_channels = configs.model.out_channels
        num_layers = configs.model.num_layers
        num_classes = configs.model.num_classes
        num_blocks = configs.model.num_blocks
        growth_rate = configs.model.growth_rate
        conv_dropout = configs.model.conv_dropout
        fc_dropout = configs.model.fc_dropout

        # initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # create residual layers
        self.res_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # first layer: same input and output channels
                res_layer = self._make_layer(out_channels, out_channels,
                                             num_blocks=num_blocks, stride=1, dropout=conv_dropout)
            elif 1 <= i <= 2:
                # middle layers: increase output channels by growth rate
                res_layer = self._make_layer(out_channels, int(out_channels * growth_rate),
                                             num_blocks=num_blocks, stride=2, dropout=conv_dropout)
                out_channels = int(out_channels * growth_rate)
            else:
                # final layers: increase the output channels by even larger amount
                res_layer = self._make_layer(out_channels, int(out_channels * 2),
                                             num_blocks=num_blocks, stride=2, dropout=conv_dropout)
                out_channels = int(out_channels * 2)

            self.res_layers.append(res_layer)

        # global average pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # final fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=fc_dropout),  # Dropout before final classification
            nn.Linear(out_channels // 2, num_classes)
        )

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

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)


def prepare_model(configs):
    model = CIFARModel(configs)
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_layerwise_parameters(model):

    print("\nLayer-wise Parameter Breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    return total_params


if __name__ == '__main__':
    # This is the main function to test the model's components
    print("Testing model components")

    from box import Box
    import yaml

    config_file_path = 'configs/config.yaml'
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    test_configs = Box(config_data)

    test_model = prepare_model(test_configs)
    count_layerwise_parameters(test_model)
    print(test_model)

    dummy_input = torch.randn(1, 3, 32, 32)

    try:
        output = test_model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")

