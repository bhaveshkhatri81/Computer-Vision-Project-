print("Installing dependencies...")
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from itertools import cycle
import torchvision.utils as vutils
import torchvision.transforms as transforms
# import wandb
print("Done")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the transformation
transform = Compose([
    Resize((128, 128)),
    ToTensor(),  # Convert the images to PyTorch tensors
    Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])  # Normalize the images to the range [-1, 1]
])

class CustomDatasetNoSketch(Dataset):
    def __init__(self, img_dir, csv_file):
        self.labels_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        # Extract classes from the one-hot vectors
        self.classes = [str(i) for i in range(len(self.labels_frame.iloc[0, 1:]))]  # Assuming classes are represented as one-hot vectors

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame.iloc[idx % len(self.labels_frame), 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        labels = torch.tensor(self.labels_frame.iloc[idx, 1:8].values.astype(int))
        image = transform(image)
        return image, labels

data_path = "/scratch/data/m23cse017/train/"
result_path = "/scratch/data/m23cse017/results/"

# Create the datasets
train_dataset = CustomDatasetNoSketch(csv_file=data_path + 'Train_labels_paired.csv',img_dir=data_path + 'Train_data_paired')
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# Define the CNN classifier
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjusting input size to match output of conv layers
        self.fc2 = nn.Linear(128, 7)  # 7 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 32 * 32)  # Flattening the output before passing to fc1
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the classifier and optimizer
classifier = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Train the classifier
num_epochs = 50
for epoch in range(num_epochs):
    classifier.train()
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        #print(images.shape)
        outputs = classifier(images)
        #print(outputs.shape)
        #print(labels.shape)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the trained classifier model
torch.save(classifier.state_dict(), os.path.join(result_path,f"output/classifier_final.pth"))
