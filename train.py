import argparse
import yaml
import torch
import numpy as np
from utils import load_configs, prepare_saving_dir
from dataset import prepare_dataloaders
from model import prepare_model


def get_optimizer(model, configs):
    optimizer_config = configs.optimizer
    if optimizer_config.name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.lr,
            betas=(optimizer_config.beta_1, optimizer_config.beta_2),
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay
        )
    return optimizer


def train_model(model, trainloader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0


def validate_model(model, testloader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100*correct/total
    print(f'Accuracy on test set: {accuracy}%')
    return accuracy

def main(dict_config, config_file_path):

    configs = load_configs(dict_config)

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    trainloader, testloader = prepare_dataloaders(configs)

    model = prepare_model(configs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = get_optimizer(model, configs)
    num_epochs = configs.train_settings.num_epochs

    train_model(model, trainloader, optimizer, num_epochs, device)
    accuracy = validate_model(model, testloader, device)

    # should I write the accuracy to the result directory?

    # Todo: Add the rest of the code here.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a deep model on Cifar10 dataset.")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
