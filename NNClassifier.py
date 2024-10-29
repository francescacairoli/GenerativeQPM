import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
# Definiamo la classe del modello (come già fatto)
class TrajectoryClassifier1D(nn.Module):
    def __init__(self, n_classes):
        super(TrajectoryClassifier1D, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, n_classes)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x)
        
        return x
