import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

def get_celeba_dataloader(data_path, image_size=64, batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # CelebA原始大小178x218
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    # 这里假设 data_path 指向上级目录，如 data/img_align_celeba
    # ImageFolder 需要再往里一层，但实际要根据你放置数据的结构自行调整

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return dataloader