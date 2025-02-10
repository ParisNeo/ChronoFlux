import torch
import torch.nn as nn

# 1. Age Embedding
class AgeEmbedding(nn.Module):
    def __init__(self, embed_size=128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, age):
        return self.embed(age.unsqueeze(-1))

# 2. Generator Model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.age_embed = AgeEmbedding(128)

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # Upsampling avec les canaux corrigés
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, 4, 2, 1),  # 256 (middle) + 128 (age_emb)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1),  # 128 (up1) + 128 (d2)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.final = nn.ConvTranspose2d(64 + 64, 3, 4, 2, 1)  # 64 (up2) + 64 (d1)

    def forward(self, x, age):
        age_emb = self.age_embed(age)
        d1 = self.down1(x)       # [B, 64, 64, 64]
        d2 = self.down2(d1)      # [B, 128, 32, 32]
        m = self.middle(d2)      # [B, 256, 16, 16]

        # Concaténation avec l'embedding d'âge
        age_emb = age_emb.view(-1, 128, 1, 1).expand(-1, -1, m.shape[2], m.shape[3])
        m = torch.cat([m, age_emb], dim=1)  # [B, 256+128=384, 16, 16]

        u1 = self.up1(m)         # [B, 128, 32, 32]
        u1 = torch.cat([u1, d2], dim=1)      # [B, 128+128=256, 32, 32]
        u2 = self.up2(u1)        # [B, 64, 64, 64]
        u2 = torch.cat([u2, d1], dim=1)      # [B, 64+64=128, 64, 64]
        return torch.tanh(self.final(u2))    # [B, 3, 128, 128]

# 3. Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))

        self.age_embed = AgeEmbedding(256)
        self.discriminator = nn.Linear(256 + 256, 1)
        self.age_predictor = nn.Linear(256 + 256, 1)

    def forward(self, x, age):
        features = self.conv(x)
        features = features.mean(dim=[2, 3])  # Global average pooling

        age_emb = self.age_embed(age)
        combined = torch.cat([features, age_emb], dim=1)

        validity = torch.sigmoid(self.discriminator(combined))
        age_pred = torch.sigmoid(self.age_predictor(combined))
        return validity, age_pred
