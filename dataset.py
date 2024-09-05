import torch
import torchvision
import torchvision.transforms as transforms

def prepare_dataloaders(configs):
    # todo: Add the code to prepare the dataloaders here.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    pass


if __name__ == '__main__':
    # This is the main function to test the dataloader
    print("Testing dataloader")