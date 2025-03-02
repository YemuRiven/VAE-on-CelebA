import torch
from models.vae import VAE  # 确保导入类名与实际定义的类名一致
from torchvision.utils import save_image, make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设已经定义好和训练时相同结构的 VAE 类
model2 = VAE().to(device)
model2.load_state_dict(torch.load('outputs/vae_celebA.pth'))  # 加载已训练好的权重

model2.eval()
with torch.no_grad():
    # 从标准正态分布中随机采样 latent_dim=128 的向量
    z = torch.randn(16, 128).to(device)

    # 通过解码器生成新头像
    samples = model2.decode(z)

    # 可视化并保存生成结果
    # 将样本拼接成网格 (4x4)，并输出到本地文件 new_faces.png
    grid = make_grid(samples, nrow=4, padding=2, normalize=True)
    save_image(grid, 'new_faces.png')
    print("New faces generated and saved to new_faces.png")
