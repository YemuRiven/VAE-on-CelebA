"""
Usage:
Training:
python main.py \
  --data_path ./data/CelebA/Img \
  --epochs 10 \
  --batch_size 64 \
  --lr 0.001 \
  --image_size 64 \
  --latent_dim 128 \
  --out_dir ./outputs
"""



import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.vae import VAE
from utils.dataset import get_celeba_dataloader

def loss_function(recon_x, x, mu, logvar):
    # 重构损失 (MSE或BCE均可，这里用MSE为例)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL散度损失
    # KL(N(mu, sigma), N(0,1)) = sum( 1/2 * (mu^2 + sigma^2 - log sigma^2 - 1) )
    # logvar = log(sigma^2)
    kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)

    return recon_loss + kl_loss, recon_loss, kl_loss

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for x, _ in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = loss_function(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    return total_loss / len(dataloader.dataset), \
           total_recon / len(dataloader.dataset), \
           total_kl / len(dataloader.dataset)

def sample_images(model, epoch, device, sample_num=16, out_dir="outputs"):
    model.eval()
    with torch.no_grad():
        z = torch.randn(sample_num, model.latent_dim).to(device)
        samples = model.decode(z).cpu()

    # 可视化并保存
    grid_size = int(sample_num**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5,5))
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            axes[i][j].imshow(samples[idx].permute(1,2,0).numpy())
            axes[i][j].axis('off')
            idx += 1

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}_samples.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/CelebA',
                        help='Path to CelebA dataset.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--image_size', type=int, default=64, help='Target image size.')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers.')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Output directory.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    dataloader = get_celeba_dataloader(args.data_path, args.image_size, args.batch_size, args.num_workers)

    # 初始化模型
    model = VAE(image_size=args.image_size, latent_dim=args.latent_dim, nc=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        train_loss, recon_loss, kl_loss = train(model, dataloader, optimizer, device)
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {train_loss:.4f}, Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}")

        # 生成样本
        sample_images(model, epoch, device, sample_num=16, out_dir=args.out_dir)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(args.out_dir, "vae_celebA.pth"))
    print("Training complete! Model saved.")

if __name__ == '__main__':
    main()