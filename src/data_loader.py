import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Load and preprocess the FER-2013 dataset.
    Args:
        data_dir (str): Path to the dataset directory containing 'train', 'val', and 'test' folders.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses to use for data loading.
    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
