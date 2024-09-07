import torch
import torchvision
import torchvision.transforms as transforms
import yaml

def prepare_dataloaders(configs):
    # todo: Add the code to prepare the dataloaders here.

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # todo: replace this line
    with open(configs, 'r') as file:
        configs = yaml.safe_load(file)

    batch_size = configs['train_settings']['batch_size']
    num_workers = configs['train_settings']['num_workers']

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


if __name__ == '__main__':
    # This is the main function to test the dataloader
    print("Testing dataloader")
    trainloader, testloader = prepare_dataloaders('configs/config.yaml')
    print(len(trainloader))
    print(len(testloader))
    print(trainloader.dataset)
    print(testloader.dataset)
    print(trainloader.dataset.data.shape)
    # print(trainloader.dataset.targets)
