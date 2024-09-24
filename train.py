import argparse
import yaml
import torch
import numpy as np
from utils import load_configs, prepare_saving_dir, prepare_tensorboard, get_optimizer
from dataset import prepare_dataloaders
from model import prepare_model
import tqdm

def training_loop(model, trainloader, optimizer, epoch, device, train_writer=None):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch + 1}',
                                         leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(trainloader)
    if train_writer:
        train_writer.add_scalar('Loss', avg_train_loss, epoch)
        lr = optimizer.param_groups[0]['lr']
        train_writer.add_scalar('Learning_Rate', lr, epoch)

    return avg_train_loss


def validation_loop(model, testloader, epoch, device, valid_writer=None):
    model.eval()
    total = 0
    correct = 0
    valid_loss = 0.0

    criterion = torch.nn.functional.cross_entropy

    for i, (inputs, labels) in tqdm.tqdm(enumerate(testloader), total=len(testloader), desc=f'Validation Epoch {epoch + 1}',
                                         leave=False):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_valid_loss = valid_loss / len(testloader)
    accuracy = 100*correct/total
    if valid_writer:
        valid_writer.add_scalar('Loss', avg_valid_loss, epoch)
        valid_writer.add_scalar('Accuracy', accuracy, epoch)

    print(f'Accuracy on epoch {epoch}: {accuracy}%')
    return valid_loss

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

    # Initialize train and valid TensorBoards
    train_writer, valid_writer = prepare_tensorboard(result_path)

    num_epochs = configs.train_settings.num_epochs

    for epoch in range(num_epochs):
        training_loop(model, trainloader, optimizer, epoch, device, train_writer)
        validation_loop(model, testloader, epoch, device, valid_writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a deep model on Cifar10 dataset.")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
