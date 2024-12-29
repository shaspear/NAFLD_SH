import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 数据预处理
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 数据加载
data_dir = 'path_to_your_dataset'
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=image_transforms['train'])
valid_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'valid'), transform=image_transforms['valid'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# 定义ResNet50模型
class MultiTaskResNet50(nn.Module):
    def __init__(self, task1_classes, task2_classes):
        super(MultiTaskResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity()  # 移除最后的全连接层
        self.task1_fc = nn.Linear(self.resnet50.fc.in_features, task1_classes)
        self.task2_fc = nn.Linear(self.resnet50.fc.in_features, task2_classes)

    def forward(self, x):
        features = self.resnet50(x)
        task1_output = self.task1_fc(features)
        task2_output = self.task2_fc(features)
        return task1_output, task2_output

# 实例化模型
task1_classes = 2  # 第一个任务的类别数
task2_classes = 5  # 第二个任务的类别数
task3_classes = 3  # 第3个任务的类别数
task4_classes = 3  # 第4个任务的类别数
model = MultiTaskResNet50(task1_classes, task2_classes)

# 定义损失函数和优化器
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
criterion3 = nn.CrossEntropyLoss()
criterion4 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, train_loader, valid_loader, criterion1, criterion2, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, (labels1, labels2) in train_loader:
            inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = model(inputs)
            loss1 = criterion1(outputs1, labels1)
            loss2 = criterion2(outputs2, labels2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

        # 验证模型
        model.eval()
        correct1, correct2 = 0, 0
        with torch.no_grad():
            for inputs, (labels1, labels2) in valid_loader:
                inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device)
                outputs1, outputs2 = model(inputs)
                _, predicted1 = torch.max(outputs1, 1)
                _, predicted2 = torch.max(outputs2, 1)
                correct1 += (predicted1 == labels1).sum().item()
                correct2 += (predicted2 == labels2).sum().item()
        accuracy1 = correct1 / len(valid_dataset)
        accuracy2 = correct2 / len(valid_dataset)
        print(f'Validation Accuracy Task 1: {accuracy1:.4f}, Task 2: {accuracy2:.4f}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 开始训练
train_model(model, train_loader, valid_loader, criterion1, criterion2, optimizer, num_epochs=25)