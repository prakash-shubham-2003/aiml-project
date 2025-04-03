import os
import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import load_data
from model import model1, model2, model3
from tqdm import tqdm
from utils import save_model, get_device

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_HYPERPARAMETERS = {
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'num_workers': 4,
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'input_size': (48, 48),
    'num_classes': 7,
    'dataset': 'FER_2013'
}

def train(model, model_name):
    if os.path.exists(os.path.join(PROJECT_ROOT, 'models', model_name + '.pth')):
        print(f"Model {model_name} already exists. Skipping training.")
        return
    
    hyperparameters = BASE_HYPERPARAMETERS.copy()
    hyperparameters.update({
        'model_type': model.__class__.__name__
    })
    
    device = get_device()
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'preprocessed_datasets', hyperparameters['dataset'])
    train_loader, val_loader, _ = load_data(
        data_dir, 
        hyperparameters['batch_size'], 
        num_workers=hyperparameters['num_workers']
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

    print(f"Starting training for {model_name}...")
    print(f"Using device: {device}")
    print("\nHyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")
    print("\n" + "="*50)

    for epoch in range(hyperparameters['num_epochs']):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{hyperparameters["num_epochs"]}]', leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=(running_loss / (progress_bar.n + 1)))

        train_loss = running_loss/len(train_loader)
        print(f'Epoch [{epoch+1}/{hyperparameters["num_epochs"]}], Training Loss: {train_loss:.4f}')

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

        val_loss = val_loss/len(val_loader)
        val_accuracy = 100 * correct/total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print("="*50)
    
    save_model(
        model=model,
        path=os.path.join(PROJECT_ROOT, 'models'),
        model_name=model_name,
        hyperparameters=hyperparameters
    )

if __name__ == '__main__':
    print("\nTraining Model 1")
    model = model1()
    train(model=model, model_name='model1')
    
    print("\nTraining Model 2")
    model = model2()
    train(model=model, model_name='model2')
    
    print("\nTraining Model 3")
    model = model3()
    BASE_HYPERPARAMETERS['num_epochs'] = 15
    train(model=model, model_name='model3')