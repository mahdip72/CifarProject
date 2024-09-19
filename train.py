import argparse
import yaml
import torch
import numpy as np
from utils import load_configs, prepare_saving_dir, write_accuracy, plot_loss
from dataset import prepare_dataloaders
from model import prepare_model
import random


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

# not very clean, but added a plotting functionality to look for overfitting
def train_model(model, trainloader, optimizer, num_epochs, device,
                plot_losses=False, testloader=None):
    if plot_losses:
        train_losses = []
        val_losses = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if plot_losses:
                total_running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        if plot_losses and testloader is not None:
            train_losses.append(total_running_loss / len(trainloader))
            val_losses.append(quick_valid_loss(model, testloader, device))

    if plot_losses and testloader is not None:
        return train_losses, val_losses

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


def quick_valid_loss(model, testloader, device, subset_size=100):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    criterion = torch.nn.functional.cross_entropy

    # Randomly sample a subset of the testloader
    sampled_batches = random.sample(list(testloader), subset_size)

    with torch.no_grad():
        for inputs, labels in sampled_batches:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / subset_size
    return avg_val_loss

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

    train_losses, test_losses = train_model(
        model, trainloader, optimizer, num_epochs, device,
        plot_losses=True, testloader=testloader
    )
    accuracy = validate_model(model, testloader, device)

    # should I write the accuracy to the result directory?
    write_accuracy(result_path, accuracy)
    plot_loss(train_losses, test_losses, result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a deep model on Cifar10 dataset.")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
