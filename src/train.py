import os
import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import load_data
from model import model1, model2, model3
from tqdm import tqdm  

def train(model, save_path):
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = 'data/preprocessed_datasets/FER_2013'
    train_loader, val_loader, _ = load_data(data_dir, batch_size, num_workers=num_workers)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=(running_loss / (progress_bar.n + 1)))

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct/total:.2f}%')

    torch.save(model.state_dict(), os.path.join(os.getcwd(), save_path))

if __name__ == '__main__':
    model = model1()
    train(model=model, save_path=os.path.join('models', 'model1.pth'))
    model = model2()
    train(model=model, save_path=os.path.join('models', 'model2.pth'))
    model = model3()
    train(model=model, save_path=os.path.join('models', 'model3.pth'))