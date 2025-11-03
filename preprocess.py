import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 1. 定义你的原始数据集路径
source_dir = Path("./Plantvillage")

# 2. 定义你想要保存新数据集的路径
target_dir = Path("./Plantvillage_224")

# 3. 定义我们想要的统一尺寸
new_size = (224, 224)

# 确保 PIL 使用高质量的缩放算法
resample_filter = Image.Resampling.BILINEAR

def preprocess_images():
    # 遍历 train, val, test 文件夹
    for split in ["train", "val", "test"]:
        split_path = source_dir / split
        target_split_path = target_dir / split
        
        if not split_path.is_dir():
            print(f"Skipping {split_path}, not a directory.")
            continue

        # 获取所有类别文件夹 (e.g., "Tomato___Bacterial_spot")
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        print(f"Found {len(class_dirs)} classes in {split}...")

        # 使用 tqdm 显示总进度
        for class_dir in tqdm(class_dirs, desc=f"Processing {split} set"):
            # 在新目录中创建对应的类别文件夹
            target_class_path = target_split_path / class_dir.name
            target_class_path.mkdir(parents=True, exist_ok=True)
            
            # 遍历这个类别中的所有图片
            # (假设是 .jpg, .JPG, .jpeg, .png)
            image_files = list(class_dir.glob("*.jpg")) + \
                          list(class_dir.glob("*.JPG")) + \
                          list(class_dir.glob("*.jpeg")) + \
                          list(class_dir.glob("*.png"))

            for image_path in image_files:
                try:
                    with Image.open(image_path) as img:
                        # 1. 转换为 "RGB" (防止有些是 P 模式或 RGBA)
                        # 2. 缩放
                        # 3. 保存
                        img_rgb = img.convert("RGB")
                        img_resized = img_rgb.resize(new_size, resample_filter)
                        # 定义新图片的保存路径
                        new_image_path = target_class_path / image_path.name
                        img_resized.save(new_image_path, "JPEG",quality=95)
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    print("--- Pre-processing Complete!(V2) ---")
    print(f"All images resized and saved to {target_dir}")

if __name__ == "__main__":
    preprocess_images()
