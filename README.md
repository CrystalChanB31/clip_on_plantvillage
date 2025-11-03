# clip_on_plantvillage
CLIP模型在PlantVillage植物病害识别任务中的应用探究
CLIP模型在PlantVillage植物病害识别任务中的应用探究
1.环境准备
1.1 数据集
PlantVillage Dataset
显卡：Nvidia Geforce RTX5090 @ 32GB * 1
1.2 软件环境配置
Linux：Ubuntu 24.04LTS（WSL2）
Anaconda：最新版本
CUDA：13.0
Python version info: 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]
PyTorch version info: 2.10.0.dev20251026+cu130
1.3 requirements.txt
2.数据处理
2.1 先进行数据集的划分（测试集，训练集和验证集）
数据分类方法：
下载的数据集中分为 color , grayscale , segmented 三个文件夹，这里以 color 文件夹为例：
训练集比率：70%
验证集比率：20%
测试集比率：10%
3.使用教程
0.文件目录结构：
（工作）根目录
-dataset
--color
-data_loader.py
-split_data.py
-model.py
-train.py

1.先运行pip install -r requirements.txt 安装依赖

2.运行split_data.py划分数据集

3.运行train.py训练

4.训练结果
在Epoch为20时，有最高准确率为93.18%

模型在测试集上实现了93.49%的准确率。

precision recall f1-score support

accuracy 0.9349 5435
macro avg 0.9030 0.8849 0.8910 5435
weighted avg 0.9322 0.9349 0.9320 5435

训练损失和验证准确率与Epoch关系如下：


具体训练结果可以看这里：

results下载
