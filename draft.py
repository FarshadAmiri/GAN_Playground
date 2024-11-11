import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
z_dim = 100  # Latent space dimension (noise)
lr = 0.0002  # Learning rate
epochs = 50
sample_interval = 5  # Interval to sample and save images

# Create a simple Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)
        self.tanh = nn.Tanh()  # Normalize output to (-1, 1)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return self.tanh(x).view(-1, 1, 28, 28)  # Reshape into image format (1x28x28)

# Create a simple Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = img.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return self.sigmoid(x)
    

# Initialize the generator and discriminator
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# Loss function and optimizers
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Prepare the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)


# Function to save generated images
def save_generated_images(epoch, generator, device):
    # Create random noise to generate fake images, move to GPU
    z = torch.randn(64, z_dim, device=device)
    gen_imgs = generator(z)

    # Create a grid of images
    grid_img = make_grid(gen_imgs, nrow=8, normalize=True)

    # Convert to numpy for plotting, move to CPU for visualization
    grid_img = grid_img.cpu().detach().numpy()  # Move tensor to CPU and convert to numpy

    # Transpose dimensions for proper image orientation (C, H, W -> H, W, C)
    grid_img = np.transpose(grid_img, (1, 2, 0))

    # Plot the images
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img)
    plt.axis('off')
    plt.show()


# Training the GAN
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        # Real images
        real_imgs = imgs.cuda()

        # Generate fake images with the same batch size as real images
        z = torch.randn(real_imgs.size(0), z_dim).cuda()  # Use real_imgs.size(0) for batch size
        fake_imgs = generator(z)

        # Labels for real and fake images
        real_labels = torch.ones(real_imgs.size(0), 1).cuda()
        fake_labels = torch.zeros(real_imgs.size(0), 1).cuda()

        # Train Discriminator
        optimizer_d.zero_grad()

        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        g_loss = criterion(discriminator(fake_imgs), real_labels)  # Generator wants to fool the discriminator
        g_loss.backward()
        optimizer_g.step()

        # Print loss and save images at intervals
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(train_loader)} \
                  | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Save generated images every sample_interval
    if epoch % sample_interval == 0:
        save_generated_images(epoch, generator)

print("Training finished!")
