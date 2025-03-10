#train neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from PIL import Image
import shapely
import rasterio
from rasterio.mask import mask

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.affinity import affine_transform
from matplotlib.backends.backend_pdf import PdfPages





#PATHS
data_path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries"
path_crowns=os.path.join(data_path,r"geodataframes\BCI_50ha_crownmap_timeseries.shp")
orthomosaic_path=os.path.join(data_path,"orthomosaic_aligned_local")
orthomosaic_list=os.listdir(orthomosaic_path)

#training dataset
training_dataset=gpd.read_file(r"timeseries/training_dataset.shp")
print(training_dataset.columns)

class LeafCoverageDataset(Dataset):
    def __init__(self, dataframe, orthomosaic_path, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing your training data.
            orthomosaic_path (str): Base path where orthomosaic files are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.orthomosaic_path = orthomosaic_path
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get the row corresponding to this index
        row = self.dataframe.iloc[idx]
        
        # Build the file path (assumes file naming as in your example)
        path_orthomosaic = os.path.join(self.orthomosaic_path, f"BCI_50ha_{row['date']}_local.tif")
        
        # Load and mask the image
        with rasterio.open(path_orthomosaic) as src:
            out_image, _ = mask(src, [row.geometry], crop=True)
        out_image = np.transpose(out_image, (1, 2, 0))
        
        pil_image = Image.fromarray(out_image.astype('uint8'))
        
        if self.transform:
            pil_image = self.transform(pil_image)
        else:
            # Default transform: convert to tensor
            pil_image = transforms.ToTensor()(pil_image)
        
        leafing_score = torch.tensor(row['leafing'], dtype=torch.float32)
        return pil_image, leafing_score

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = LeafCoverageDataset(training_dataset, orthomosaic_path, transform=transform)

# Create a DataLoader to iterate over your dataset in batches
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for images, targets in dataloader:
    print(images.shape, targets.shape)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




# Define the CNN architecture for regression
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        # Assuming input images are RGB with dimensions (3, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate the size after three rounds of pooling:
        # 224 -> 112 -> 56 -> 28 (each pooling halves the dimensions)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Output: (32, 112, 112)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Output: (64, 56, 56)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # Output: (128, 28, 28)
        
        # Flatten the feature maps into a vector
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Use sigmoid to output a value between 0 and 1.
        return torch.sigmoid(x)

# Instantiate the model
model = CNNRegressor()

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Using mean squared error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Display the model architecture
print(model)