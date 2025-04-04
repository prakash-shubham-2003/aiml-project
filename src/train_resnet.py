import os, torch
import torch.optim as optim
import torch.nn as nn
from data_loader import load_data
from torch.utils.data import ConcatDataset, DataLoader
from model import ResNetEmotion
from utils import save_model, get_device
from tqdm import tqdm

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    device = get_device()

    batch_size = 32
    num_workers = 4
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'preprocessed_datasets', 'FER_2013')

    train_loader, val_loader, _ = load_data(data_dir, batch_size, num_workers=num_workers)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    full_train_dataset = ConcatDataset([train_dataset, val_dataset])
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = ResNetEmotion(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with tqdm(total=len(full_train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for inputs, labels in full_train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        avg_loss = running_loss / len(full_train_loader)
        train_accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Save the model only once at the end
    save_model(model, "models", "resnet18_emotion", epoch=num_epochs)

    print("Training complete!")

if __name__ == "__main__":
    main()
