import argparse
import yaml
import torch
import numpy as np
from utils import load_configs, prepare_saving_dir, prepare_tensorboard, get_optimizer, get_scheduler, save_checkpoint
from dataset import prepare_dataloaders
from model import prepare_model
import tqdm
import torchmetrics
from torch.amp import GradScaler, autocast



def training_loop(model, trainloader, optimizer, epoch, device, scaler, scheduler, train_writer=None, grad_clip_norm=1, **kwargs):
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    f1_score = torchmetrics.F1Score(num_classes=10, average='macro', task="multiclass")

    accuracy.to(device)
    f1_score.to(device)
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch + 1}',
                                         leave=False, disable=not kwargs['configs'].tqdm_progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # mixed precision using autocast
        with autocast(device_type=device.type):
            predicts = model(inputs)
            loss = torch.nn.functional.cross_entropy(predicts, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        predicts = torch.argmax(predicts, dim=1)

        accuracy.update(predicts.detach(), labels.detach())
        f1_score.update(predicts.detach(), labels.detach())

        if train_writer:
            train_writer.add_scalar('Gradient_Norm', grad_norm, epoch * len(trainloader) + i)

    avg_train_loss = running_loss / len(trainloader)

    epoch_acc = accuracy.compute().cpu().item()
    epoch_f1 = f1_score.compute().cpu().item()

    accuracy.reset()
    f1_score.reset()

    if train_writer:
        train_writer.add_scalar('Loss', avg_train_loss, epoch)
        train_writer.add_scalar('Accuracy', epoch_acc, epoch)
        train_writer.add_scalar('F1_Score', epoch_f1, epoch)
        lr = optimizer.param_groups[0]['lr']
        train_writer.add_scalar('Learning_Rate', lr, epoch)

    print(f'Accuracy on epoch {epoch}: {100*epoch_acc : .2f}%')

    return avg_train_loss


def validation_loop(model, testloader, epoch, device, scaler, valid_writer=None, **kwargs):
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    f1_score = torchmetrics.F1Score(num_classes=10, average='macro', task="multiclass")

    accuracy.to(device)
    f1_score.to(device)

    model.eval()
    valid_loss = 0.0

    criterion = torch.nn.functional.cross_entropy

    for i, (inputs, labels) in tqdm.tqdm(enumerate(testloader), total=len(testloader), desc=f'Validation Epoch {epoch + 1}',
                                         leave=False, disable=not kwargs['configs'].tqdm_progress_bar):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type=device.type):
                predicts = model(inputs)
                loss = criterion(predicts, labels)
                valid_loss += loss.item()

            predicts = torch.argmax(predicts, dim=1)
            accuracy.update(predicts.detach(), labels.detach())
            f1_score.update(predicts.detach(), labels.detach())

    avg_valid_loss = valid_loss / len(testloader)

    valid_accuracy = accuracy.compute().cpu().item()
    valid_f1_score = f1_score.compute().cpu().item()

    accuracy.reset()
    f1_score.reset()

    if valid_writer:
        valid_writer.add_scalar('Loss', avg_valid_loss, epoch)
        valid_writer.add_scalar('Accuracy', valid_accuracy, epoch)
        valid_writer.add_scalar('F1_Score', valid_f1_score, epoch)

    print(f'Validation Accuracy on epoch {epoch}: {100*valid_accuracy: .2f}%')

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
    print(f"Using device: {device}")
    model.to(device)
    optimizer = get_optimizer(model, configs)

    scheduler = get_scheduler(optimizer, configs)

    # Initialize train and valid TensorBoards
    train_writer, valid_writer = prepare_tensorboard(result_path)
    num_epochs = configs.train_settings.num_epochs
    grad_clip_norm = configs.train_settings.grad_clip_norm
    checkpoint_every = configs.checkpoints_every


    scaler = GradScaler()

    for epoch in range(num_epochs):

        # Training and validation loops
        training_loop(model, trainloader, optimizer, epoch, device, scaler, scheduler,
                      train_writer=train_writer, grad_clip_norm=grad_clip_norm, configs=configs)
        validation_loop(model, testloader, epoch, device, scaler, valid_writer, configs=configs)

        scheduler.step()

        # For changing the scheduler at the middle of training
        if configs.scheduler.name == 'cosine_annealing_sequential' and (epoch + 1 == num_epochs // 2) :
            new_start_lr = configs.scheduler.eta_min_first
            print(f"Resetting learning rate to {new_start_lr} at epoch {epoch + 1}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_start_lr

        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, checkpoint_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a deep model on Cifar10 dataset.")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
