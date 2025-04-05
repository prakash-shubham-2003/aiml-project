# Facial Emotion Detection

This project focuses on building and evaluating deep learning models for emotion recognition using the FER-2013 dataset. The models are trained to classify facial expressions into one of seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features
- Training multiple models (e.g., model1, model2, model3, ResNetEmotion) with customizable hyperparameters.
- Evaluation of trained models on a test dataset with metrics like accuracy and loss.
- Prediction of emotions from input images using trained models.

## How to Predict Emotions

Follow these steps to predict emotions from an image:

1. **Prepare the Environment**
   - Ensure you have all the required dependencies installed. You can install them using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Prepare the Model**
   - Train the model using the provided training scripts (`train.py` or `train_resnet.py`) or use pre-trained models available in the `models` directory.

3. **Run the Prediction Script**
   - Use the `predict.py` script to predict the emotion of an image. Run the following command:
     ```bash
     python src/predict.py <image_path> --model_path <model_path> --model_type <model_type>
     ```
     - `<image_path>`: Path to the image file.
     - `<model_path>`: Path to the trained model file (default: `models/model3.pth`).
     - `<model_type>`: Type of model to use (e.g., `model1`, `model2`, `model3`, `ResNetEmotion`).

4. **Example Command**
   ```bash
   python src/predict.py data/sample_image.jpg --model_path models/model3.pth --model_type model3
   ```

   The script will output the predicted emotion label, e.g., `Predicted Emotion: Happy`.

## Directory Structure
- `src/`: Contains the source code for training, evaluation, and prediction.
- `models/`: Directory to store trained model files.
- `data/`: Directory for datasets and preprocessed data.
- `results/`: Directory for evaluation logs and results.