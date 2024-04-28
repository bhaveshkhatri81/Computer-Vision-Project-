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
from torchvision.utils import save_image
from itertools import cycle
import torchvision.utils as vutils
import wandb
print("Done")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def denorm(x):
 out = (x + 1) / 2
 return out.clamp(0, 1)


# Define the generator (U-Net)
class cGen(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(cGen, self).__init__()
    self.down1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
    self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2))
    self.down3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2))    
    self.down4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2))    
    self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.ReLU())
    self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU())
    self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(64), nn.ReLU())
    self.up4 = nn.Sequential(nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1), nn.Tanh())

  def forward(self, sketches, labels):
    # Expand labels to have the same height and width as sketches
    labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, sketches.shape[2], sketches.shape[3])
    # Concatenate sketches and labels along the channel dimension
    x = torch.cat([sketches, labels], dim=1)
    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    d4 = self.down4(d3)
    u1 = self.up1(d4)
    u2 = self.up2(torch.cat([u1, d3], dim=1))
    u3 = self.up3(torch.cat([u2, d2], dim=1))
    u4 = self.up4(torch.cat([u3, d1], dim=1))
    return u4

# Define the discriminator (PatchGAN)
class cDiscr(nn.Module):
  def __init__(self, input_channels):
    super(cDiscr, self).__init__()
    self.layer1 = nn.Sequential(nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
    self.layer2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
    self.output = nn.Sequential(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1), nn.Sigmoid())

  def forward(self, image, labels):
    # Expand labels to have the same number of dimensions as image
    labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, image.shape[2], image.shape[3])
    # Combine image and labels here
    x = torch.cat([image, labels], dim=1)
    x = self.layer1(x)
    x = self.layer2(x)
    out = self.output(x)
    return out

class CustomDataset(Dataset):
  def __init__(self, img_dir, sketch_dir, csv_file):
    self.labels_frame = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.sketch_dir = sketch_dir

  def __len__(self):
    return len(self.labels_frame)

  def __getitem__(self, idx):
    img_name = os.path.join(self.img_dir, self.labels_frame.iloc[idx % len(self.labels_frame), 0] + '.jpg')
    image = Image.open(img_name).convert('RGB')
    sketch_name = os.path.join(self.sketch_dir, self.labels_frame.iloc[idx % len(self.labels_frame), 0] + '_segmentation.png')
    sketch = Image.open(sketch_name).convert('L')
    labels = torch.tensor(self.labels_frame.iloc[idx, 1:8].values.astype(int))
    image = transform_rgb(image)
    sketch = transform_gray(sketch)
    return image, sketch, labels

# Define the transformation
transform_rgb = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # for RGB images
])

transform_gray = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])  # for grayscale images
])

data_path_train = "/scratch/data/m23cse017/train/"
data_path_test = "/scratch/data/m23cse017/test/"
result_path = "/scratch/data/m23cse017/results/"

train_dataset = CustomDataset(csv_file=data_path_train + 'Train_labels_paired.csv',img_dir=data_path_train + 'Train_data_paired',sketch_dir=data_path_train + 'Paired_train_sketches')
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_dataset = CustomDataset(csv_file=data_path_test + 'Test_labels.csv',img_dir=data_path_test + 'Test_data',sketch_dir=data_path_test + 'Paired_test_sketch')
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

# Create the models
num_classes = 7
sketch_channels = 1  # grayscale sketches
image_channels = 3  # RGB images
image_size = 128
batch_size = 64
G = cGen(sketch_channels + num_classes, image_channels).to(device)
D = cDiscr(image_channels + num_classes).to(device)

# Define the loss function
criterion = nn.BCELoss().to(device)
criterion_L1 = nn.L1Loss().to(device)

# Define the optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Number of training epochs
num_epochs = 501

# Hyperparameter for the weight of the L1 loss
lambda_l1 = 200

# Lists to store discriminator and generator losses
d_losses = []
g_losses = []

# Training loop
for epoch in range(num_epochs):
  for i, (images, sketches, labels) in enumerate(train_dataloader):

    # Convert tensors to appropriate device
    sketches = sketches.to(device)
    labels = labels.to(device)
    images = images.to(device)

    # Generate fake images
    fake_images = G(sketches, labels)

    # Train the discriminator
    real_preds = D(images, labels)
    fake_preds = D(fake_images.detach(), labels)
    
    real_labels = torch.rand(real_preds.size()).to(device) * 0.5 + 0.7
    fake_labels = torch.rand(fake_preds.size()).to(device) * 0.3

    d_loss_real = criterion(real_preds, real_labels)
    d_loss_fake = criterion(fake_preds, fake_labels)
    d_loss = (d_loss_real + d_loss_fake) / 2

    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Train the generator more frequently
    for _ in range(2):
        fake_images = G(sketches, labels)
        fake_preds = D(fake_images, labels)
        g_loss_adv = criterion(fake_preds, torch.ones_like(fake_preds))  # Adversarial loss
        g_loss_l1 = criterion_L1(fake_images, images)  # L1 loss
        g_loss = g_loss_adv + lambda_l1 * g_loss_l1  # Total generator loss

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        #Save discriminator and generator losses
    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

    # Print some loss stats
    if i % 50 == 0:
      print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

  if epoch % 50 == 0:
    # Save the generator model
    torch.save({'epoch': epoch,'model_state_dict': G.state_dict(),'optimizer_state_dict': optimizer_G.state_dict(),}, os.path.join(result_path, f"output/generator3_model_{epoch}.pth"))
    # Save the discriminator model
    torch.save({'epoch': epoch,'model_state_dict': D.state_dict(),'optimizer_state_dict': optimizer_D.state_dict(),}, os.path.join(result_path, f"output/discriminator3_model_{epoch}.pth"))


  # After every 50 epochs, print the generated images
  if epoch % 50 == 0:
    with torch.no_grad():
      # Generate a batch of fake images
      fake_images = G(sketches, labels).detach().cpu()
      # Save the images and labels
      for j, (image, label) in enumerate(zip(fake_images, labels)):
        save_image(denorm(image), os.path.join(result_path, f"generated_images/trainGen_image3_{epoch}_{j}.png"))
        with open(os.path.join(result_path, f"generated_images/trainGen_label3_{epoch}_{j}.txt"), "w") as f:
          f.write(str(label.tolist()))  # Save the one-hot encoded label as a list
  
for i, (images, sketches, labels) in enumerate(test_dataloader):
  # Convert tensors to appropriate device
  sketches = sketches.to(device)
  labels = labels.to(device)
  images = images.to(device)    
  with torch.no_grad():
    # Generate a batch of fake images
    fake_images = G(sketches, labels).detach().cpu()
    # Save the images and labels
    for j, (image, label) in enumerate(zip(fake_images, labels)):
      save_image(denorm(image), os.path.join(result_path, f"generated_images/testGen_image3_{epoch}_{j}.png"))
      with open(os.path.join(result_path, f"generated_images/testGen_label3_{epoch}_{j}.txt"), "w") as f:
        f.write(str(label.tolist()))  # Save the one-hot encoded label as a list

plot_name = "loss_plot_500.png"
plot_path = os.path.join(result_path, plot_name)

# Plot discriminator and generator losses
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('GAN Training Loss')
plt.legend()
plt.grid(True)

plt.savefig(plot_path)
