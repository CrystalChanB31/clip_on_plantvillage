import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
NW = 32
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
def load_data(data_dir, batch_size=384):
    """
    加载训练、验证和测试数据
    """
    data_dir = Path(data_dir)
    # 数据增强和预处理
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),  # 调整大小
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=CLIP_MEAN,std=CLIP_STD)  # 标准化
    ])

    # 使用 ImageFolder 加载数据集
    train_data = datasets.ImageFolder(root=data_dir / 'train', transform=transform)
    val_data = datasets.ImageFolder(root=data_dir / 'val', transform=transform)
    test_data = datasets.ImageFolder(root=data_dir / 'test', transform=transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=NW,pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,num_workers=NW,pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,num_workers=NW,pin_memory=True)

    return train_loader, val_loader, test_loader

# 检查加载的数据集
if __name__ == "__main__":
    data_dir = "./Plantvillage_224"  # 你的数据集路径
    train_loader, val_loader, test_loader = load_data(data_dir)

    # 打印一些batch数据检查加载是否正确
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")
