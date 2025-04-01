# Emotion Recognition with Convolutional Neural Networks

This project implements a series of convolutional neural network (CNN) models to perform emotion recognition on the FER-2013 dataset. The models are trained to classify images into one of seven emotion categories.

## Project Structure

- `src/model.py`: Contains the definitions of three CNN models (`model1`, `model2`, `model3`) and additional building blocks (`ConvolutionBlock`, `DenseBlock`) used in `model3`.
- `src/train.py`: Script for training the models. It loads the data, trains each model, and saves the trained model weights.
- `src/evaluate.py`: Script for evaluating the trained models on a test dataset. It loads the model weights and computes the test accuracy and loss.
- `src/data_loader.py`: Contains the `load_data` function, which loads and preprocesses the FER-2013 dataset.

## Models

- **model1**: A simple CNN with one convolutional layer followed by two fully connected layers.
- **model2**: A deeper CNN with three convolutional layers and two fully connected layers.
- **model3**: A modular CNN using custom `ConvolutionBlock` and `DenseBlock` classes for more flexibility and depth.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- tqdm

## Usage

### Training

To train the models, run the `train.py` script. This will train each model and save the weights to the `models` directory.

### Evaluation

To evaluate the models, run the `evaluate.py` script. This will load the saved model weights and evaluate them on the test dataset.

## Data

The FER-2013 dataset should be organized in the following structure:

Each subdirectory (`train`, `val`, `test`) should contain subdirectories for each emotion class, with images inside.