// This template file converts Jupyter Notebook code cells to plain Python code.
// For each code cell in the notebook, it extracts the source code and applies
// the 'ipython2python' filter to convert IPython syntax to standard Python.
// Only code cells are processed; other cell types are ignored.


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

get_ipython().run_line_magic('matplotlib', 'inline')





random_seed = 42
lr = 3e-5
batch_size = 32





# Dataset Configuration - Change this to use different datasets
DATASET_NAME = "CIFAR10"  # Options: "MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "CelebA"

def get_dataset_config(dataset_name):
    """Configure dataset-specific parameters"""
    if dataset_name == "MNIST":
        return {
            'dataset': datasets.MNIST,
            'train_transform': transforms.Compose([transforms.ToTensor()]),
            'test_transform': transforms.Compose([transforms.ToTensor()]),
            'root': './data'
        }
    elif dataset_name == "FashionMNIST":
        return {
            'dataset': datasets.FashionMNIST,
            'train_transform': transforms.Compose([transforms.ToTensor()]),
            'train_transform': transforms.Compose([transforms.ToTensor()]),
            'root': './data'
        }
    elif dataset_name == "CIFAR10":
        return {
            'dataset': datasets.CIFAR10,
            'train_transform': transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'test_transform': transforms.Compose([transforms.ToTensor(),
                                                #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                  ]),
            'root': './data'
        }
    elif dataset_name == "CIFAR100":
        return {
            'dataset': datasets.CIFAR100,
            'train_transform': transforms.Compose([transforms.ToTensor()]),
            'test_transform': transforms.Compose([transforms.ToTensor()]),
            'root': './data'
        }
    elif dataset_name == "CelebA":
        def celebA_wrapper(root, train, download, transform):
            # CelebA expects split: "train", "valid", "test"
            split = "train" if train else "test"
            return datasets.CelebA(root=root, split=split, download=download, transform=transform)
        return {
            'dataset': celebA_wrapper,
            'train_transform': transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]),
            'test_transform': transforms.Compose([transforms.ToTensor()]),
            'root': './data'
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

dataset_config = get_dataset_config(DATASET_NAME)
print(f"Using dataset: {DATASET_NAME}")





from torch.utils.data import Subset

subset_size = 1000
train_indices = np.random.choice(len(dataset_config['dataset'](root=dataset_config['root'], train=True)), subset_size, replace=False)
test_indices = np.random.choice(len(dataset_config['dataset'](root=dataset_config['root'], train=False)), subset_size, replace=False)

train_dataset_full = dataset_config['dataset'](
    root=dataset_config['root'], 
    train=True, 
    download=True, 
    transform=dataset_config['train_transform']
)
test_dataset_full = dataset_config['dataset'](
    root=dataset_config['root'], 
    train=False, 
    download=True, 
    transform=dataset_config['test_transform']
)

train_dataset = Subset(train_dataset_full, train_indices)
test_dataset = Subset(test_dataset_full, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Determine number of classes from dataset
num_classes = len(train_dataset_full.classes)
print(f"Number of classes: {num_classes}")

# Get sample to determine input dimensions
sample_input = train_dataset_full[0][0]
input_channels = sample_input.shape[0]
input_height = sample_input.shape[1]
input_width = sample_input.shape[2]
print(f"Input shape: {input_channels}x{input_height}x{input_width}")





def show_images(dataset, title, num_images=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        image, label = dataset[i]
        plt.subplot(1, num_images, i + 1)
        
        # Handle both grayscale and color images
        if image.shape[0] == 1:  # Grayscale image
            plt.imshow(image.squeeze(), cmap='gray')
        else:  # Color image (RGB)
            # Convert from CHW to HWC format for matplotlib
            img_display = image.permute(1, 2, 0)
            # Denormalize if needed (for CIFAR datasets)
            if img_display.min() < 0:
                img_display = (img_display + 1) / 2  # Convert from [-1,1] to [0,1]
            plt.imshow(img_display.clamp(0, 1))
        
        plt.title(f'Label: {label}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

show_images(train_dataset, 'Train Dataset Samples')
show_images(test_dataset, 'Test Dataset Samples')

print(f'Train dataset size: {len(train_dataset)}')
print(f'Train dataset classes: {train_dataset_full.classes}')
# Each item in the dataset is a tuple: (image, label)
# image: torch.Tensor of shape [channels, height, width]
# label: int (class index)
print(f'Size of each data sample in train_dataset: {train_dataset[0][0].shape}')
print(f'Type of image: {type(train_dataset[0][0])}, Type of label: {type(train_dataset[0][1])}')





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEBottleneck(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, in_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.fc_decode(z)
        return x_recon, mu, logvar


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_channels=3, input_height=32, input_width=32, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        # Encoder using strided convs (no MaxPool)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # Bx32x16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Bx64x8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Bx128x4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # Flatten → Bottleneck → Unflatten
        self.flatten = nn.Flatten()
        self.bottleneck = VAEBottleneck(in_dim=128 * 4 * 4, latent_dim=latent_dim)
        self.unflatten = nn.Unflatten(1, (128, 4, 4))

        # Decoder: reverse the encoder structure
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x8x8                                                                                                                                                                                                                                                                                                                                                                                                                                   
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 16x32x32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, input_channels, kernel_size=3, padding=1),
            # nn.Tanh()  # Match normalized output in [-1, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x_flat = self.flatten(x)
        z_flat, mu, logvar = self.bottleneck(x_flat)
        x_unflat = self.unflatten(z_flat)
        out = self.decoder(x_unflat)
        return out, mu, logvar

    def get_latents(self, x):
        x = self.encoder(x)
        x_flat = self.flatten(x)
        z_flat, mu, logvar = self.bottleneck(x_flat)
        return z_flat





model = VariationalAutoEncoder(
    input_channels=input_channels,
    input_height=input_height,
    input_width=input_width,
    latent_dim=256
).to(device)

print(model)





import torch.nn.functional as F

def vae_loss_fn(recon_x, x, mu, logvar, kl_weight=1.0, use_mse=False):
    """
    VAE loss = reconstruction loss + weighted KL divergence

    Args:
        recon_x: reconstructed output [B, C, H, W]
        x: original input image [B, C, H, W]
        mu: mean from encoder [B, latent_dim]
        logvar: log variance from encoder [B, latent_dim]
        kl_weight: weighting factor for KL loss (for annealing)
        use_mse: if True, use MSE; else use BCE

    Returns:
        total loss, recon loss, kl divergence
    """

    if use_mse:
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')  # smooth for RGB
    else:
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')

    # KL divergence between q(z|x) and p(z) = N(0, I)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss + kl_weight * kl_div
    return loss, kl_div, recon_loss





# Train parameters
train_losses_half_epoch = []
test_losses_half_epoch = []
num_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=lr)





# Create these global variables *before* your training loop starts
# These will hold the plot objects for dynamic updates
loss_fig, loss_ax = None, None
train_line, test_line = None, None
image_fig, image_axes = None, None
image_plots = []


def plot_losses(train, test):
    global loss_fig, loss_ax, train_line, test_line

    if loss_fig is None:
        # Initialize the plot for the first time
        loss_fig, loss_ax = plt.subplots(figsize=(10, 6))
        train_line, = loss_ax.plot([], [], label="Train Loss", marker='o', alpha=0.6)
        test_line, = loss_ax.plot([], [], label="Test Loss", marker='x')
        loss_ax.set_xlabel('Epochs')
        loss_ax.set_ylabel('Loss')
        loss_ax.set_title('Training & Test Losses')
        loss_ax.legend()
        loss_ax.grid(True)
        loss_fig.tight_layout()
        plt.show(block=False) # Important: show without blocking, once
    
    # Update the data for the existing lines
    # Adjust `steps` calculation if your logging frequency changes
    steps = np.arange(len(train)) * 0.5 # since you're logging every 0.5 epoch
    train_line.set_data(steps, train)
    test_line.set_data(steps, test)

    # Autoscale the y-axis (optional, but good for dynamic plots)
    loss_ax.relim()
    loss_ax.autoscale_view()

    # Redraw the canvas
    loss_fig.canvas.draw()
    loss_fig.canvas.flush_events() # Crucial for real-time update in notebooks
    plt.pause(0.001) # Small pause to allow GUI events to process


def display_image(img, ax, title_suffix):
    """Display an image after denormalizing from [-1, 1] to [0, 1] if needed"""
    # If your VAE's last layer is Sigmoid, images are already [0,1].
    # If using tanh and Normalize transform with (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), uncomment:
    # img = (img + 1) / 2
    
    img_display = img.permute(1, 2, 0).cpu().detach().numpy() # Move to CPU, detach, to numpy
    
    # Ensure values are within [0, 1] for display with imshow
    img_display = np.clip(img_display, 0, 1) 
    
    ax.set_title(title_suffix)
    ax.axis('off')
    return ax.imshow(img_display)


def sample_images_from_dataset(dataset, num_samples=5, random_seed=42):
    global image_fig, image_axes, image_plots

    model.eval()
    with torch.no_grad():
        np.random.seed(random_seed) # Ensure reproducibility of sample selection
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        sample_images = [dataset[i][0] for i in indices]
        sample_labels = [dataset[i][1] for i in indices]

        batch = torch.stack(sample_images).to(device)
        reconstructed, _, _ = model(batch)

        if image_fig is None:
            # Initialize the plot for the first time
            image_fig, image_axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
            # Handle cases where image_axes might be a 1D array if num_samples is 1
            if num_samples == 1:
                image_axes = np.array([image_axes]).reshape(2, 1) # Ensure 2D for consistency

            image_plots = [[None for _ in range(num_samples)] for _ in range(2)]

            # Initial drawing
            for i in range(num_samples):
                # Original image
                img_orig = sample_images[i]
                image_plots[0][i] = display_image(img_orig, image_axes[0, i], f'Orig: {sample_labels[i]}')

                # Reconstructed image (placeholder for now, will be updated)
                img_recon = reconstructed[i].cpu()
                image_plots[1][i] = display_image(img_recon, image_axes[1, i], 'Recon')
            
            image_fig.suptitle('Original (top) vs Reconstructed (bottom)')
            plt.show(block=False) # Show without blocking for updates
        else:
            # Update data for existing plots
            for i in range(num_samples):
                # Update original image (only needed if new samples are drawn, or labels change)
                # For fixed samples, you technically only need to set_title if labels can change.
                # If images are always the same, this part can be skipped after first draw.
                img_orig = sample_images[i]
                image_plots[0][i].set_data(img_orig.permute(1, 2, 0).clamp(0, 1).numpy()) # Ensure numpy
                image_axes[0, i].set_title(f'Orig: {sample_labels[i]}')

                # Update reconstructed image
                img_recon = reconstructed[i].cpu()
                image_plots[1][i].set_data(img_recon.permute(1, 2, 0).clamp(0, 1).numpy()) # Ensure numpy

        # Redraw the canvas
        image_fig.canvas.draw()
        image_fig.canvas.flush_events()
        plt.pause(0.001) # Small pause to allow GUI events to process





def kl_anneal(epoch, total_epochs, scale=5):
    return float(1 / (1 + np.exp(-scale * (epoch / total_epochs - 0.5))))

def test_eval(kl_weight):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output, mu, logvar = model(data)
            loss, _, _ = vae_loss_fn(output, data, mu, logvar, kl_weight)
            total_loss += loss.item() * data.size(0)
    return total_loss / len(test_loader.dataset)

for epoch in range(num_epochs):
    model.train()
    kl_weight = kl_anneal(epoch, num_epochs)
    running_loss = 0
    count = 0

    kl_div, recon_loss = 0, 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        output, mu, logvar = model(data)
        loss, kl_div, recon_loss = vae_loss_fn(output, data, mu, logvar, kl_weight)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

        # At half epoch
        if batch_idx == len(train_loader) // 2:
            avg_train_loss = running_loss / count
            train_losses_half_epoch.append(avg_train_loss)
            test_loss = test_eval(kl_weight)
            test_losses_half_epoch.append(test_loss)
    
            print(f"Epoch {epoch + 0.5:.1f} | Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"KL div loss = {kl_div:.4f}, Recon loss = {recon_loss:.4f}")
            plot_losses(train_losses_half_epoch, test_losses_half_epoch)
            sample_images_from_dataset(test_dataset, num_samples=5, random_seed=random_seed)

    # At end of epoch
    avg_train_loss = running_loss / count
    train_losses_half_epoch.append(avg_train_loss)
    test_loss = test_eval(kl_weight)
    test_losses_half_epoch.append(test_loss)

    print(f"Epoch {epoch + 1:.1f} | Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
    print(f"KL div loss = {kl_div:.4f}, Recon loss = {recon_loss:.4f}")
    plot_losses(train_losses_half_epoch, test_losses_half_epoch)
    sample_images_from_dataset(test_dataset, num_samples=5, random_seed=random_seed)





from sklearn.manifold import TSNE
import umap

# Collect latent vectors and labels from test set
model.eval()
latents = []
labels = []
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        # Pass through encoder and bottleneck only
        encoded = model.encoder(data)
        encoded_flat = model.flatten(encoded)
        
        mu = model.bottleneck.fc_mu(encoded_flat)
        logvar = model.bottleneck.fc_logvar(encoded_flat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + std * eps

        latents.append(mu.cpu().numpy())
        labels.append(target.numpy())
latents = np.concatenate(latents, axis=0)
labels = np.concatenate(labels, axis=0)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=random_seed)
latents_2d = tsne.fit_transform(latents)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(num_classes), label='Label')
plt.title('VAE Latent Space Visualization (t-SNE)')
plt.xlabel('t-SNE Dim 1')
plt.ylabel('t-SNE Dim 2')
plt.grid(True)
plt.show()





sample_images_from_dataset(test_dataset, num_samples=5)





# Noisy reconstruction.
# This wouldn't work as the VAE hasn't been trained to handle noise.
# Check the other notebook for a proper denoising VAE implementation.


indices = np.random.choice(len(test_dataset), size=5, replace=False)
sample_images = [test_dataset[i][0] for i in indices]
sample_labels = [test_dataset[i][1] for i in indices]

with torch.no_grad():
    batch = torch.stack(sample_images).to(device)
    reconstructed, _, _ = model(batch)

    plt.figure(figsize=(10, 4))
    for i in range(5):
        # Noisy image
        ax1 = plt.subplot(2, 5, i + 1)
        display_image(batch[i].cpu(), ax1, f'Noisy\nLabel: {sample_labels[i]}')
        
        # Reconstructed image
        ax2 = plt.subplot(2, 5, i + 6)
        display_image(reconstructed[i].cpu(), ax2, 'Recon')
        
    plt.suptitle('Noisy (top) vs Reconstructed (bottom)')
    plt.show()





label1, label2 = 0, 0

while label1 == label2:
    indices = np.random.choice(len(test_dataset), size=2, replace=False)
    sample_images = [test_dataset[i][0] for i in indices]
    sample_labels = [test_dataset[i][1] for i in indices]
    label_names = [test_dataset_full.classes[l] for l in sample_labels]
    img1, img2 = sample_images
    label1, label2 = label_names

print(img1.shape, img2.shape)

# Convert images to latent vectors z1,z2
z = model.get_latents(torch.stack([img1, img2]).to(device))
print(z.shape)
z1 = z[0].cpu()
z2 = z[1].cpu()

with torch.no_grad():
    alphas = np.linspace(0, 1, 10)
    interpolated_latents = [a * z2 + (1 - a) * z1 for a in alphas]
    with torch.no_grad():
        recons = model.unflatten(torch.stack(interpolated_latents).to(device))
        recons = model.decoder(recons)
        recons = recons.cpu()

    plt.figure(figsize=(18, 3))
    # Draw img1
    ax1 = plt.subplot(1, len(alphas) + 2, 1)
    display_image(img1.cpu(), ax1, f'Image 1\nLabel: {label1}')

    # Draw reconstructions
    for i, (a, recon) in enumerate(zip(alphas, recons)):
        ax = plt.subplot(1, len(alphas) + 2, i + 2)
        display_image(recon, ax, f'α={a:.1f}')

    # Draw img2
    ax2 = plt.subplot(1, len(alphas) + 2, len(alphas) + 2)
    display_image(img2.cpu(), ax2, f'Image 2\nLabel: {label2}')

    plt.suptitle('Interpolation: img1 → img2')
    plt.show()





from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, ax = plt.subplots(figsize=(3, 3))
# Initialize with proper display format
if img1.shape[0] == 1:  # Grayscale
    im = ax.imshow(img1.squeeze().cpu(), cmap='gray', animated=True)
else:  # Color
    img_display = img1.permute(1, 2, 0).cpu()
    if img_display.min() < 0:
        img_display = (img_display + 1) / 2
    im = ax.imshow(img_display.clamp(0, 1), animated=True)
ax.axis('off')

def animate(i):
    alpha = i / 59  # 60 frames for smoothness
    interpolated = (1 - alpha) * z1 + alpha * z2
    with torch.no_grad():
        recon = model.unflatten(interpolated.unsqueeze(0).to(device))
        recon = model.decoder(recon).cpu()
        # Don't squeeze here to preserve dimensions
        recon = recon[0]  # Remove batch dimension instead
    
    # Display reconstruction properly
    if recon.shape[0] == 1:  # Grayscale
        im.set_array(recon.squeeze())
    else:  # Color - ensure we have 3D tensor before permute
        if len(recon.shape) == 3:  # Should be (C, H, W)
            img_display = recon.permute(1, 2, 0)
            if img_display.min() < 0:
                img_display = (img_display + 1) / 2
            im.set_array(img_display.clamp(0, 1))
        else:
            print(f"Unexpected tensor shape: {recon.shape}")
    
    ax.set_title(f'α={alpha:.2f}')
    return [im]

ani = FuncAnimation(fig, animate, frames=60, interval=50, blit=True)
plt.close(fig)
HTML(ani.to_jshtml())



