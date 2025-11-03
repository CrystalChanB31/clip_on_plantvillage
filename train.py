#train8
import matplotlib
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from tqdm import tqdm
from model import PlantDiseaseModel  # 导入 *修改后* 的模型
from data_loader import load_data
import clip
import matplotlib.pyplot as plt
import platform
system = platform.system()
if system == "Windows":
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
elif system == "Darwin":
    matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
else:
    matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# 加载数据集
data_dir = "./Plantvillage_224"
train_loader, val_loader, test_loader = load_data(data_dir)

# --- [修改] ---
num_classes = 38
model = PlantDiseaseModel(in_channels_img=512, out_channels_img=256, num_classes=num_classes).to(device)
# --- [修改结束] ---

# 强制模型为 float32
model = model.float()

# 设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 加载 CLIP 模型 (这部分保留，用于在 *训练脚本* 中提取特征)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# 训练函数
def train(model, train_loader, val_loader, num_epochs):
    best_accuracy = 0.0
    best_model_path = "best_model.pth"
    history_loss = []
    history_val_accuracy = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            labels = labels.long()

            # 1. 获取图像特征
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
            image_features = image_features.float()
            
            # 2. 获取模型输出
            outputs = model(image_features)
            
            # 3. 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 计算并打印 Loss
        epoch_loss = running_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")

        # 验证模型并获取准确率
        val_accuracy = validate(model, val_loader)
        
        # 记录这轮的 loss 和 accuracy
        history_loss.append(epoch_loss)
        history_val_accuracy.append(val_accuracy)
        
        # 检查并保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"*** 新的最佳模型已保存，准确率: {best_accuracy * 100:.2f}% ***")

    # 训练循环结束后，返回历史数据
    return history_loss, history_val_accuracy

# 验证函数
def validate(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    # 使用 tqdm 包装 val_loader
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            labels = labels.long()

            # 1. 获取图像特征
            image_features = clip_model.encode_image(images)
            image_features = image_features.float()
            
            # 2. 获取模型输出
            outputs = model(image_features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print("混淆矩阵 (Validation):")
    print(cm)
    
    return accuracy  # <-- 返回计算出的准确率

# 测试函数
def test(model, test_loader):
    print("\n--- 启动测试阶段 ---")
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_labels = []
    
    # 从 test_loader 中获取类别名称，用于报告
    try:
        class_names = test_loader.dataset.classes
    except:
        class_names = [str(i) for i in range(num_classes)] # 备用方案

    with torch.no_grad():
        # 使用 tqdm 显示进度条
        for images, labels in tqdm(test_loader, desc="Testing"): 
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            labels = labels.long()

            # 1. 获取图像特征 (clip_model 是全局变量)
            image_features = clip_model.encode_image(images)
            image_features = image_features.float()
            
            # 2. 获取模型输出
            outputs = model(image_features)
            
            # 3. 获取预测
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n--- 测试结果 ---")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    print("\n混淆矩阵 (Test):")
    print(cm)
    
    # 打印分类报告 (包含精确率, 召回率, F1-score)
    print("\n分类报告 (Test):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
def plot_charts(history_loss, history_val_accuracy, num_epochs):
    """
    绘制 Loss 和 Validation Accuracy 曲线图
    """
    print("正在生成图表...")
    
    # 准备 x 轴 (Epochs)
    epochs = range(1, num_epochs + 1)
    
    # --- 绘制 Loss 曲线 ---
    plt.figure(figsize=(12, 5)) # 创建一个图窗
    
    plt.subplot(1, 2, 1) # 1行2列的第1个子图
    plt.plot(epochs, history_loss, 'bo-', label='Training Loss') # 蓝色, 圆点, 实线
    plt.title('Training Loss (训练损失)')
    plt.xlabel('Epochs (轮次)')
    plt.ylabel('Loss (损失值)')
    plt.legend()
    plt.grid(True) # 显示网格

    # --- 绘制 Accuracy 曲线 ---
    plt.subplot(1, 2, 2) # 1行2列的第2个子图
    plt.plot(epochs, history_val_accuracy, 'go-', label='Validation Accuracy') # 绿色, 圆点, 实线
    plt.title('Validation Accuracy (验证准确率)')
    plt.xlabel('Epochs (轮次)')
    plt.ylabel('Accuracy (准确率)')
    plt.ylim(0.5, 1.0) # 设置 Y 轴范围在 0.5 到 1.0 之间，更易观察
    plt.legend()
    plt.grid(True)

    # 保存图表
    plt.suptitle('Model Training History (模型训练历史)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，为总标题留出空间
    
    output_filename = 'training_history_plot.png'
    plt.savefig(output_filename)
    print(f"图表已保存为: {output_filename}")
# 开始训练
if __name__ == "__main__":
    best_model_path = "best_model.pth"
    # 在这里统一定义 epoch 数量
    num_epochs_to_train = 20 

    # 1. 训练模型，并接收返回的历史数据
    history_loss, history_val_accuracy = train(model, train_loader, val_loader, num_epochs=num_epochs_to_train)
    
    print("\n--- 训练完成 ---")

    # 2. [新增] 调用绘图函数
    plot_charts(history_loss, history_val_accuracy, num_epochs_to_train)

    # 3. 加载最佳模型权重用于测试
    print("正在加载最佳模型权重用于测试...")
    model.load_state_dict(torch.load(best_model_path))

    # 4. 使用最佳模型进行测试
    test(model, test_loader)