# VAE on CelebA

项目地址：https://github.com/YemuRiven/VAE-on-CelebA

本项目使用 PyTorch 对 [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 数据集进行训练，构建一个简单的 Variational Autoencoder (VAE)，并生成新的头像图像。

## 特性

- 使用自定义的 VAE 模型 (PyTorch)
- 支持对 CelebA 数据进行裁剪/缩放等预处理
- 训练后可直接从先验分布采样生成新的人脸图像

## 环境安装

1. 克隆本仓库：
    ```bash
    git clone https://github.com/YemuRiven/VAE-on-CelebA.git
    cd vae-celeba
    ```

2. 安装依赖 (Conda 或 pip 方式均可)：
    ```bash
    conda env create -f environment.yml
    conda activate vae-celeba
    ```

## 数据准备

1. 从 [CelebA 官方地址](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 下载数据集，创建并放置到 `data/` 文件夹下，结构为：
    ```
   data/CelebA
      ├── Anno        (标注信息)
      ├── Eval        (验证/测试信息)
      └── Img
         └── img_align_celeba
            ├── 000001.jpg
            ├── 000002.jpg
            ├── ...
    ```

2. 在 `main.py` 中配置数据集路径。

## 运行项目

```bash
python main.py \
  --data_path ./data/CelebA/Img \
  --epochs 10 \
  --batch_size 64 \
  --lr 0.001 \
  --image_size 64 \
  --latent_dim 128 \
  --out_dir ./outputs
