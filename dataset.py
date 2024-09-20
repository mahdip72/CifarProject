import torch
import torchvision
import torchvision.transforms as transforms


def prepare_dataloaders(configs):

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 normalization
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10-specific normalization
    ])

    train_batch_size = configs.train_settings.batch_size
    valid_batch_size = configs.valid_settings.batch_size
    shuffle = configs.train_settings.shuffle
    train_num_workers = configs.train_settings.num_workers
    valid_num_workers = configs.valid_settings.num_workers


    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=shuffle,
        num_workers=train_num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=valid_batch_size, shuffle=False,
        num_workers=valid_num_workers
    )

    return trainloader, testloader


def denormalize(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def visualize_random_batch(trainloader, num_images=10):

    import matplotlib.pyplot as plt
    import numpy as np
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    cifar10_classes = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }

    images = images[:num_images]
    labels = labels[:num_images]

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)


    images = torch.stack([denormalize(img, mean, std) for img in images])
    images_np = images.numpy()

    fig, axes = plt.subplots(1, len(images_np), figsize=(num_images * 2, 2))
    for i in range(len(images_np)):
        img = np.transpose(images_np[i], (1, 2, 0))
        axes[i].imshow(img)
        label_name = cifar10_classes[labels[i].item()]
        axes[i].set_title(f"Label: {labels[i].item()}, {label_name}")
        axes[i].axis('off')
    plt.show()

    print("finished visualizing random batch")


if __name__ == '__main__':
    # This is the main function to test the dataloader
    print("Testing dataloader")

    from box import Box
    import yaml

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

    visualize_random_batch(trainloader)

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
