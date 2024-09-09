import torch
import torchvision
import torchvision.transforms as transforms
import yaml

def prepare_dataloaders(configs):
    # todo: Add the code to prepare the dataloaders here.

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10-specific normalization
    ])

    train_batch_size = configs.train_settings.batch_size
    valid_batch_size = configs.valid_settings.batch_size
    shuffle = configs.train_settings.shuffle
    num_workers = configs.train_settings.num_workers

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=valid_batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


if __name__ == '__main__':
    # This is the main function to test the dataloader
    print("Testing dataloader")

    from box import Box
    config_file_path = 'configs/config.yaml'
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    configs = Box(config_data)

    trainloader, testloader = prepare_dataloaders(configs)
    print(len(trainloader))
    print(len(testloader))
    print(trainloader.dataset)
    print(testloader.dataset)
    print(trainloader.dataset.data.shape)
    # print(trainloader.dataset.targets)

    # # Load the dataset without normalization
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    #
    # # Calculate mean and std
    # loader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=2)
    # data = next(iter(loader))
    # images, labels = data
    #
    # # Calculate mean over (batch, height, width)
    # mean = images.mean(dim=[0, 2, 3])
    # std = images.std(dim=[0, 2, 3])
    #
    # print(f"Mean: {mean}")
    # print(f"Standard Deviation: {std}")
