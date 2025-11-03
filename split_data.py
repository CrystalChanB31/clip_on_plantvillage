# Plantvillage/split_data.py
import os, shutil, random, sys
from pathlib import Path

# ===== 配置区 =====
SRC_DIR = Path("./dataset/color")   # 你的源数据：color 文件夹路径
DEST_DIR = Path("./Plantvillage")                # 目标根目录：会生成 train/val/test
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.2, 0.1
SEED = 42
CLEAR_DEST = False   # 若你多次尝试，想先清空再重新拷贝，改为 True（小心！会删除目标目录）

# ===== 工具函数 =====
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(d: Path):
    return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def ensure_dirs(*dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def copy_many(paths, target_dir: Path):
    ensure_dirs(target_dir)
    for p in paths:
        shutil.copy2(p, target_dir / p.name)

def split_indices(n, tr=TRAIN_RATIO, vr=VAL_RATIO, te=TEST_RATIO):
    """对长度为 n 的数组索引，返回 (train_idx, val_idx, test_idx)"""
    idx = list(range(n))
    random.shuffle(idx)

    if n == 0:
        return [], [], []
    if n == 1:
        return idx, [], []            # 1张：全放train
    if n == 2:
        return idx[:1], idx[1:], []   # 2张：1/1/0
    if n == 3:
        return idx[:2], idx[2:], []   # 3张：2/1/0
    if n == 4:
        return idx[:3], idx[3:], []   # 4张：3/1/0

    # n >= 5 用比例
    n_train = max(1, int(round(tr * n)))
    n_val   = max(1, int(round(vr * n)))
    # 确保不超
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test  = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        # 再次纠偏
        n_val = min(n_val, n - n_train)

    tr_idx = idx[:n_train]
    va_idx = idx[n_train:n_train+n_val]
    te_idx = idx[n_train+n_val:]
    return tr_idx, va_idx, te_idx

def main():
    random.seed(SEED)

    if not SRC_DIR.exists():
        print(f"[ERR] 源目录不存在：{SRC_DIR.resolve()}")
        sys.exit(1)

    if CLEAR_DEST and DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    ensure_dirs(DEST_DIR / "train", DEST_DIR / "val", DEST_DIR / "test")

    class_dirs = [p for p in SRC_DIR.iterdir() if p.is_dir()]
    if not class_dirs:
        print(f"[ERR] 在 {SRC_DIR} 下未找到类别文件夹。请确认路径是否正确（应为 color/ 下的各类别目录）。")
        sys.exit(1)

    total_train = total_val = total_test = 0
    skipped = 0

    for cls_dir in sorted(class_dirs):
        imgs = list_images(cls_dir)
        if len(imgs) == 0:
            print(f"[WARN] 类别 {cls_dir.name} 无图片，跳过。")
            skipped += 1
            continue

        tr_idx, va_idx, te_idx = split_indices(len(imgs))
        tr_imgs = [imgs[i] for i in tr_idx]
        va_imgs = [imgs[i] for i in va_idx]
        te_imgs = [imgs[i] for i in te_idx]

        copy_many(tr_imgs, DEST_DIR / "train" / cls_dir.name)
        copy_many(va_imgs, DEST_DIR / "val"   / cls_dir.name)
        copy_many(te_imgs, DEST_DIR / "test"  / cls_dir.name)

        total_train += len(tr_imgs)
        total_val   += len(va_imgs)
        total_test  += len(te_imgs)

        print(f"[OK] {cls_dir.name}: {len(imgs)} => train {len(tr_imgs)}, val {len(va_imgs)}, test {len(te_imgs)}")

    print("\n====== 汇总 ======")
    print(f"类别总数：{len(class_dirs)}（跳过空类 {skipped}）")
    print(f"Train: {total_train} | Val: {total_val} | Test: {total_test}")
    print(f"输出目录：{DEST_DIR.resolve()}")

if __name__ == "__main__":
    main()

