import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from data_loader import load_data
from utils import load_model, get_device

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def evaluate(model_path, model_type):
    """
    Evaluate a model on the test set.
    
    Args:
        model_path (str): Path to the saved model weights
        model_type (str): Type of model ('model1', 'model2', or 'model3')
    """
    batch_size = 32
    num_workers = 4
    device = get_device()
    
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'preprocessed_datasets', 'FER_2013')
    _, _, test_loader = load_data(data_dir, batch_size, num_workers=num_workers)
    
    model = load_model(model_path, model_type, device)
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
    return average_loss, accuracy

if __name__ == '__main__':
    models_to_evaluate = [
        # (os.path.join(PROJECT_ROOT, 'models', 'model1.pth'), 'model1'),
        # (os.path.join(PROJECT_ROOT, 'models', 'model2.pth'), 'model2'),
        # (os.path.join(PROJECT_ROOT, 'models', 'model3.pth'), 'model3'),
        # (os.path.join(PROJECT_ROOT, 'models', 'resnet18_emotion_epoch15.pth'), 'ResNetEmotion'),
        (os.path.join(PROJECT_ROOT, 'models', 'resnet18_emotion_epoch20.pth'), 'ResNetEmotion')
    ]
    
    for model_path, model_type in models_to_evaluate:
        print(f"\nEvaluating {model_type} at {model_path}")
        print("-*-" * 20)
        try:
            evaluate(model_path, model_type)
        except Exception as e:
            print(f"Error evaluating {model_type}: {str(e)}")
        print("-*-" * 20)