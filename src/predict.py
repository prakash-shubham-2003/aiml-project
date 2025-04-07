import os
import torch
from torchvision import transforms, datasets
from PIL import Image
from utils import load_model, get_device

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def get_class_labels(data_dir):
    dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'))
    return dataset.classes

def preprocess_image(image_path, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)

def predict(image_path, model_path=None, model_type='model3'):
    """
    Predict the emotion of the given image.
    
    Args:
        image_path (str): Path to the image.
        model_path (str): Path to the model file (default: model3.pth).
        model_type (str): Type of model (default: 'model3').
    
    Returns:
        str: Predicted emotion label.
    """
    device = get_device()
    
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, 'models', 'model3.pth')

    model = load_model(model_path, model_type, device)
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'preprocessed_datasets', 'FER_2013')
    class_labels = get_class_labels(data_dir)
    image_tensor = preprocess_image(image_path, device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return class_labels[predicted_class]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Predict the emotion from an image using a trained model.")
    parser.add_argument('image_path', type=str, help="Path to the image file.")
    parser.add_argument('--model_path', type=str, default=None, help="Path to the model file (default: model3.pth).")
    parser.add_argument('--model_type', type=str, default='model3', help="Model type (default: model3).")

    args = parser.parse_args()
    emotion = predict(args.image_path, args.model_path, args.model_type)
    print(f"Predicted Emotion: {emotion}")