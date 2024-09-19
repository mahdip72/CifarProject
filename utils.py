from box import Box
from pathlib import Path
import datetime
import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter


def get_optimizer(model, configs):
    optimizer_config = configs.optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config.lr,
        betas=(optimizer_config.beta_1, optimizer_config.beta_2),
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay
    )
    return optimizer


def prepare_tensorboard(result_path):
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    return train_writer, val_writer


def prepare_saving_dir(configs, config_file_path):
    """
    Prepare a directory for saving a training results.

    Args:
        configs: A python box object containing the configuration options.
        config_file_path: Directory of configuration file.

    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Create the result directory and the checkpoint subdirectory.
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)
    # Return the path to the result directory.
    return result_path, checkpoint_path


def load_configs(config):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Convert the necessary values to floats.
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)

    return tree_config

def write_accuracy(result_path, accuracy):
    """
    Write the accuracy to a file in the result directory.

    Args:
        result_path: The path to the result directory.
        accuracy: The accuracy value to write to the file.
    """
    with open(os.path.join(result_path, 'accuracy.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy}%\n")

def plot_loss(train_losses, val_losses, result_path):
    """
    Plot the training and validation loss and save the plot to the result directory.

    Args:
        train_losses: A list of training losses.
        val_losses: A list of validation losses.
        result_path: The path to the result directory.
    """
    import matplotlib.pyplot as plt

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(result_path, 'loss_plot.png'))
    plt.show()