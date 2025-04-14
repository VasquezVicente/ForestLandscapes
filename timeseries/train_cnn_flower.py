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
from sklearn.metrics import mean_squared_error
#path
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"

#select device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Parameters
num_epochs=6
batch_size=4
learning_rate= 0.0001

#define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  #224 for now
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1], is this the issue?
])

class CrownDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)  
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
        return image, labels, img_name

dataset = CrownDataset(csv_file=r"timeseries/dataset_training/train_cnn_flower.csv", root_dir=os.path.join(data_path,"flower_data"), transform=transform)

train_size = int(0.9 * len(dataset))  # 90% for training
test_size = len(dataset) - train_size  # 10% for testing

# Split dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# this is the convolutional neural network architecture
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 

        self.fc1 = nn.Linear(512 * 14 * 14, 1024)  
        self.fc2 = nn.Linear(1024, 256)  
        self.fc3 = nn.Linear(256, 3)  # Change to 4 for classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here (logits output)
        return x

model= CNNClassifier().to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).long()  # important for CrossEntropyLoss

        outputs = model(images)  # shape should be [batch_size, num_classes]
        loss = criterion(outputs, labels)  # do NOT reshape outputs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
print('Finish Training')


true_values = []
predicted_values = []
image_names = []

with torch.no_grad():
    model.eval()
    for images, labels, image_name in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        image_n = [os.path.basename(path) for path in image_name]

        outputs = model(images)  # logits: [batch_size, num_classes]
        _, predicted = torch.max(outputs, 1)  # Get predicted class index

        true_values.extend(labels.cpu().numpy())  # still fine as floats
        predicted_values.extend(predicted.cpu().numpy())  # now class indices
        image_names.extend(image_n)

rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
print(f"RMSE: {rmse:.4f}")

data = {
    'polygon_id': image_names,
    'leafing_predicted':  [float(arr[0]) for arr in predicted_values],
    'leafing': true_values
}

predictions_df = pd.DataFrame(data)
predictions_df['polygon_id'] = predictions_df['polygon_id'].str.split('.', expand=True)[0]
predictions_df.to_csv(r'timeseries/dataset_results/cnn_run2.csv')


# Plotting the true vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(true_values, predicted_values, color='blue', alpha=0.5, label='Predictions')
plt.plot([0, 6], [0, 6], color='red', linestyle='--', label='Ideal line')  
plt.xlabel('True Values (Leafing)')
plt.ylabel('Predicted Values (Leafing)')
plt.title('True vs Predicted Leafing Values')
plt.legend()
plt.grid(True)
plt.show()

torch.save(model.state_dict(), 'timeseries/models/flower_model.pth')


