import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, image_size=64, latent_dim=128, nc=3):
        """
        image_size: 输入图像的宽高 (假设宽高相同)
        latent_dim: 潜在空间维度
        nc:         图像通道数 (CelebA为RGB，因此nc=3)
        """
        super(VAE, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.nc = nc

        # 编码器网络：卷积层或全连接层均可，这里以简单卷积示例
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # (B, 32, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # (B, 64, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # (B, 128, 8, 8)
            nn.ReLU(True),
            nn.Flatten()                 # (B, 128*8*8) = (B, 8192)
        )

        # 获取flatten后通道数
        self.enc_out_dim = 128 * (image_size // 8) * (image_size // 8)

        # 均值与对数方差
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # 解码器：反卷积或线性+reshape，这里以反卷积示例
        self.decoder_input = nn.Linear(latent_dim, self.enc_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),
            nn.Sigmoid()  # 输出在[0,1]之间
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 128, self.image_size // 8, self.image_size // 8)
        result = self.decoder(result)
        return result

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar