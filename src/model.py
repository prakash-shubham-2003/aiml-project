import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 48 * 48, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 16 * 48 * 48)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)  

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=0.25)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class model3(nn.Module):
    def __init__(self, input_size=48, input_channels=1, num_classes=7):
        super(model3, self).__init__()
        self.conv1 = ConvolutionBlock(input_channels, 32, kernel_size=3)
        self.conv2 = ConvolutionBlock(32, 64, kernel_size=5)
        self.conv3 = ConvolutionBlock(64, 128, kernel_size=3)
        self.flatten = nn.Flatten()
        
        feature_size = input_size // 8
        self.fc1_input_features = 128 * feature_size * feature_size
        
        self.fc1 = DenseBlock(self.fc1_input_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetEmotion, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)