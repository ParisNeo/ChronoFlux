import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from omegaconf import OmegaConf
from chronoflux.model import AgeEmbedding, Generator, Discriminator

# Load configuration from YAML file
config = OmegaConf.load("configs/train_config.yaml")


# 4. Custom Dataset
class FaceAgingDataset(Dataset):
    def __init__(self, image_folder, transform):
        self.image_folder = image_folder
        # Filter only .jpg files
        self.image_paths = [
            os.path.join(image_folder, img) 
            for img in os.listdir(image_folder) 
            if img.lower().endswith('.jpg')  # Ensure only .jpg files are loaded
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        # Extract age from filename (<age>_<gender>_<race>_<date>.jpg)
        age = float(os.path.basename(image_path).split("_")[0]) / 100.0  # Normalized age (0-1)
        return self.transform(image), torch.tensor(age, dtype=torch.float32)
    
# 5. Training Configuration
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((config.resolution, config.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = FaceAgingDataset(config.dataset_path, transform)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

optimizer_G = optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
optimizer_D = optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

adversarial_loss = nn.BCELoss()
age_loss = nn.MSELoss()

# 6. Training Loop
for epoch in range(config.epochs):
    for i, (real_images, real_ages) in enumerate(dataloader):
        real_images, real_ages = real_images.to(device), real_ages.to(device)

        # --------------------
        # Train Discriminator
        # --------------------
        optimizer_D.zero_grad()

        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.zeros(real_images.size(0), 1, device=device)

        real_validity, real_age_pred = D(real_images, real_ages)
        d_real_loss = adversarial_loss(real_validity, valid) + age_loss(real_age_pred, real_ages)

        # Generate fake images
        fake_images = G(real_images, torch.rand_like(real_ages))

        fake_validity, fake_age_pred = D(fake_images.detach(), real_ages)
        d_fake_loss = adversarial_loss(fake_validity, fake)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # --------------------
        # Train Generator
        # --------------------
        optimizer_G.zero_grad()

        fake_validity, fake_age_pred = D(fake_images, real_ages)
        g_loss = adversarial_loss(fake_validity, valid) + age_loss(fake_age_pred, real_ages)

        g_loss.backward()
        optimizer_G.step()

        if i % config.log_interval == 0:
            print(f"Epoch {epoch}/{config.epochs} | Batch {i} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Save a sample generation
    if epoch % config.checkpoint_interval == 0:
        with torch.no_grad():
            sample_ages = torch.linspace(0, 1, steps=5).to(device)
            sample_images = real_images[:1].repeat(5, 1, 1, 1)  # Use the same face
            generated = G(sample_images, sample_ages)
            grid = torchvision.utils.make_grid(generated, normalize=True)
            torchvision.utils.save_image(grid, f"{config.output_dir}/generated_epoch_{epoch}.png")

# 7. Save Model
torch.save(G.state_dict(), f"{config.output_dir}/generator.pth")
torch.save(D.state_dict(), f"{config.output_dir}/discriminator.pth")