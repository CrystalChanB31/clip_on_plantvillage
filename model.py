import torch
import torch.nn as nn

class PlantDiseaseModel(nn.Module):
    def __init__(self, in_channels_img=512, out_channels_img=256, num_classes=38):
        """
        一个标准的图像分类模型，它接收来自 CLIP 的 512 维特征。
        """
        super(PlantDiseaseModel, self).__init__()
        
        # 1. 图像特征处理层
        self.image_fc = nn.Linear(in_channels_img, out_channels_img)
        
        # 2. 最终分类层
        self.fc = nn.Linear(out_channels_img, num_classes)
    
    def forward(self, image_features):
        """
        定义模型的前向传播。
        输入 'image_features' 是 CLIP 已经提取好的 [batch_size, 512] 特征。
        """
        # 1. 通过图像层
        # [B, 512] -> [B, 256]
        x = torch.relu(self.image_fc(image_features.view(image_features.size(0), -1)))
        
        # 2. 通过最终分类层
        # [B, 256] -> [B, num_classes]
        output = self.fc(x)
        
        return output
