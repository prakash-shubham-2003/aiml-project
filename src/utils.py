import torch
from torchvision import transforms
from PIL import Image
import os
from model import *

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path, model_type='model3', device='cpu'):
    """Load a PyTorch model from a saved state dictionary.
    
    Args:
        model_path (str): Path to the saved model file
        model_type (str): Type of model to load ('model1', 'model2', or 'model3')
        device (str): Device to load the model onto ('cpu' or 'cuda')
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
        
    Raises:
        FileNotFoundError: If model_path doesn't exist
        ValueError: If model_type is invalid
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    models_dir = 'models'
    available_models = [f.split('_')[0] for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    try:
        model_class = globals()[model_type]
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of {available_models}")
    
    model = model_class()
    state_dict = torch.load(model_path, map_location=device)
    
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, size=(48, 48), device='cpu'):
    """
    Preprocess an arbitrary image for prediction.
    
    Args:
        image_path (str): Path to the input image.
        size (tuple): Target size for resizing the image (default: (48, 48)).
        device (str): Device to load the image tensor onto ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(device)
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")
    
def save_model(model, path, model_name, hyperparameters=None, epoch=None):
    os.makedirs(path, exist_ok=True)
    filename = f"{model_name}_epoch{epoch}.pth" if epoch is not None else f"{model_name}.pth"
    save_path = os.path.join(path, filename)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'hyperparameters': hyperparameters or {}
    }
    
    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def inspect_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    saved_dict = torch.load(model_path, map_location='cpu')
    state_dict = saved_dict.get('model_state_dict', saved_dict)
    
    info = {
        'total_parameters': sum(p.numel() for p in state_dict.values()),
        'layer_names': list(state_dict.keys()),
        'layer_shapes': {k: v.shape for k, v in state_dict.items()},
        'file_size': os.path.getsize(model_path) / (1024 * 1024),
        'hyperparameters': saved_dict.get('hyperparameters', {})
    }
    
    return info
