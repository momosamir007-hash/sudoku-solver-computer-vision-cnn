import torch
import torch.nn as nn
from torchvision import models

class ConvNet(nn.Module):
    """Advanced CNN for digit recognition (28x28 input) with BatchNorm & Dropout."""
    def __init__(self, num_classes=10):
        super().__init__()
        # الطبقة الأولى مع إضافة Batch Normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # الطبقة الثانية مع إضافة Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # الطبقات المتصلة بالكامل (Fully Connected) مع إضافة Dropout
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # تكبير عدد العصبونات قليلاً لزيادة الاستيعاب
        self.dropout = nn.Dropout(0.5) # إسقاط 50% من الاتصالات لمنع الحفظ الأعمى
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # تمرير البيانات: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        x = x.flatten(start_dim=1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) # تطبيق الـ Dropout قبل الطبقة الأخيرة
        x = self.fc2(x)
        return x


class ResNet152(nn.Module):
    """ResNet152 with custom head for digit recognition."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-5]:
            param.requires_grad = False

        # Custom classification head
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)
