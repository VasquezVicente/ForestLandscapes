import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

#path
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"

#select device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Parameters
num_epochs=4
batch_size=4
learning_rate= 0.001

#define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  #224 for now
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

class CrownDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)  # 
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)  

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 1]))
        image = Image.open(img_name).convert('RGB')  
        labels = float(self.annotations.iloc[idx, 2])  
        if self.transform:
            image = self.transform(image)
        return image, labels

dataset = CrownDataset(csv_file=r"timeseries/dataset_training/train_cnn.csv", root_dir=os.path.join(data_path,"train_dataset"), transform=transform)

train_size = int(0.9 * len(dataset))  # 90% for training
test_size = len(dataset) - train_size  # 10% for testing

# Split dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# this is the convolutional neural network architecture
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 

        self.fc1 = nn.Linear(512 * 14 * 14, 4096) 
        self.fc2 = nn.Linear(4096, 1024) 
        self.fc3 = nn.Linear(1024, 1) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 512 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return x
    
model= CNNRegressor().to(device)

criterion = nn.MSELoss()  # For regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).float()
        outputs= model(images)
        #print(outputs)
        loss = criterion(outputs.view(-1), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #track progress
        if (i+1)% 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finish Training')


true_values = []
predicted_values = []

with torch.no_grad():  # Disable gradient computation during inference
    model.eval()  # Set model to evaluation mode
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Collect true and predicted values
        true_values.extend(labels.cpu().numpy())  # Move labels to CPU and convert to numpy
        predicted_values.extend(outputs.cpu().numpy())  # Move outputs to CPU and convert to numpy

# Plotting the true vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(true_values, predicted_values, color='blue', alpha=0.5, label='Predictions')
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal line')  # Line for perfect prediction
plt.xlabel('True Values (Leafing)')
plt.ylabel('Predicted Values (Leafing)')
plt.title('True vs Predicted Leafing Values')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig('plots/cnn_second.png')  # Save to the 'plots' directory
plt.show()