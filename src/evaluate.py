import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from data_loader import load_data
from model import model1, model2, model3

def evaluate(model, model_path):
    batch_size = 32
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = 'data/preprocessed_datasets/FER_2013'
    _, _, test_loader = load_data(data_dir, batch_size, num_workers=num_workers)

    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), model_path)))
    model.eval()
    print("Model loaded successfully.")

    test_labels = []
    test_preds = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(test_labels, test_preds)
    average_loss = test_loss / len(test_loader)

    print(f'Test Loss: {average_loss:.4f}')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    for model in ['model1.pth', 'model2.pth', 'model3.pth']:
        model_path = os.path.join('models', model)
        print(f"Evaluating model: {model_path}")
        if model == 'model1.pth':
            model_instance = model1()
        elif model == 'model2.pth':
            model_instance = model2()
        elif model == 'model3.pth':
            model_instance = model3()
        else:
            raise ValueError(f"Unknown model: {model}")
        evaluate(model=model_instance, model_path=model_path)
        print("-*-" * 20)