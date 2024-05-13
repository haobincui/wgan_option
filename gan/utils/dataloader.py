from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size):
    # Transformations for the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load the dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader
