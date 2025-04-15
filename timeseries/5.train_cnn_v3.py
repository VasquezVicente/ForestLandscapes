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
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

from sklearn.metrics import mean_squared_error

# Path
data_path = r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
num_epochs = 6
batch_size = 6
learning_rate = 0.0001

# Basic transform for original data
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Augmentation transform for underrepresented labels
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class CrownDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_filter=None):
        self.annotations = pd.read_csv(csv_file)
        if label_filter:
            self.annotations = self.annotations[self.annotations.iloc[:, 2].apply(label_filter)]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 1]))
        image = Image.open(img_name).convert('RGB')
        label = float(self.annotations.iloc[idx, 2])
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

# Load full dataset (original)
full_dataset = CrownDataset(
    csv_file=r"timeseries/dataset_training/train_cnn.csv",
    root_dir=os.path.join(data_path, "train_dataset"),
    transform=base_transform
)

# Train-test split
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset_raw, test_dataset = random_split(full_dataset, [train_size, test_size])

# Load samples with label < 100 for augmentation
underrepresented_dataset = CrownDataset(
    csv_file=r"timeseries/dataset_training/train_cnn.csv",
    root_dir=os.path.join(data_path, "train_dataset"),
    transform=augmentation_transform,
    label_filter=lambda x: x < 100
)

# Create augmented samples (e.g. 2 augmentations per sample)
num_augmentations = 2
augmented_data = []
counter = 0

# Iterate over the underrepresented dataset
for i in range(len(underrepresented_dataset)):
    for _ in range(num_augmentations):
        image, label, _ = underrepresented_dataset[i]
        augmented_data.append((image, label))
        counter += 1
        if counter % 100 == 0:  # Print progress every 100 augmentations
            print(f"Processed {counter} augmentations...")

# Prepare original train dataset
train_data_original = [(img, lbl) for img, lbl, _ in train_dataset_raw]

# Combine original + augmented
train_data_combined = train_data_original + augmented_data

# Convert to Tensors
X_train = torch.stack([img for img, _ in train_data_combined])
y_train = torch.tensor([lbl for _, lbl in train_data_combined], dtype=torch.float32)

# Final TensorDataset and DataLoader
train_dataset_aug = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


original_labels = [label for _, label, _ in train_dataset_raw]
plt.figure(figsize=(8, 4))
plt.hist(original_labels, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Label Distribution Before Augmentation")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# After augmentation: original + synthetic
augmented_labels = [label for _, label in train_data_combined]  # train_data_combined includes both
plt.figure(figsize=(8, 4))
plt.hist(augmented_labels, bins=50, alpha=0.7, color='salmon', edgecolor='black')
plt.title("Label Distribution After Augmentation")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# this is the convolutional neural network architecture
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 

        self.fc1 = nn.Linear(512 * 14 * 14, 1024) 
        self.fc2 = nn.Linear(1024, 256) 
        self.fc3 = nn.Linear(256, 1) 

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
image_names= []

with torch.no_grad():  
    model.eval()  
    for images, labels, image_name in test_loader:
        images = images.to(device)
        labels = labels.to(device)  #labeles correspond to leafing 
        image_n = [os.path.basename(path) for path in image_name]
        # Forward pass
        outputs = model(images)  #outputs correspond to predicted_leafing
        
        # Collect true and predicted values
        true_values.extend(labels.cpu().numpy())  
        predicted_values.extend(outputs.cpu().numpy())  
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
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal line')  
plt.xlabel('True Values (Leafing)')
plt.ylabel('Predicted Values (Leafing)')
plt.title('True vs Predicted Leafing Values')
plt.legend()
plt.grid(True)
plt.show()

torch.save(model.state_dict(), 'timeseries/models/cnn_model.pth')


