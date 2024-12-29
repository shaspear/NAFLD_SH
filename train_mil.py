import copy
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, DataLoader
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm

from local_tools.MIL_Dataset import MILDataset

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# 定义ResNet50模型
class ResNetMIL(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMIL, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()  # 移除最后的全连接层
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.resnet50(x)
        return self.fc(features.mean(dim=0)).unsqueeze(dim=0)  # 取bag内所有图片特征的平均值

def load_partial_state_dict(model, state_dict):
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


# 实例化模型
num_classes = 2  # 假设有两个类别
model = ResNetMIL(num_classes)
# 加载预训练参数
pretrained_model_path = './results/mil_resnet50_epoch0.pth'
if os.path.exists(pretrained_model_path):
    pretrained_dict = torch.load(pretrained_model_path)
    load_partial_state_dict(model, pretrained_dict)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据加载
data_dir = r'E:\myproject\nafld\data\outputs\bags'
# train_dataset = MILDataset(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
# valid_dataset = MILDataset(os.path.join(data_dir, 'val'), transform=data_transforms['val'])

dataset = MILDataset(data_dir, transform=data_transforms['train'])
# 计算分割点
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size

# 分割数据集
train_dataset, valid_dataset = Subset(dataset, range(train_size)), Subset(dataset, range(train_size, train_size + val_size))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
# 训练模型
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=25):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for bags, labels, l2, l3, l4 in tqdm(train_loader):
            # bags = torch.cat(bags)
            # b, c, h, w = bags.shape
            # bags = bags.reshape((b,3,h,-1))
            bags = torch.squeeze(bags)
            bags, labels = bags.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(bags)
            #outputs = torch.unsqueeze(outputs, 0)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            #running_corrects += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.double() / len(train_loader)
        print(f'\nEpoch {epoch+1}, Training Loss: {epoch_loss}, acc: {epoch_acc}')
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        # 验证模型
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for bags, labels, l2, l3, l4 in tqdm(valid_loader):
                # bags = torch.cat(bags)
                # b, c, h, w = bags.shape
                # bags = bags.reshape((1, 3, h, -1))
                bags = torch.squeeze(bags)
                bags, labels = bags.to(device), labels.to(device)
                outputs = model(bags)
                #outputs = torch.unsqueeze(outputs, 0)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                val_loss += loss.item()
                correct += (predicted == labels).sum().item()
        epoch_loss = val_loss / len(valid_loader)
        epoch_acc = correct / len(valid_loader)
        #accuracy = correct / len(valid_dataset)
        print(f'\nValidation loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        # 深拷贝模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # 保存模型
            torch.save(best_model_wts, f'results'
                                       f''
                                       f'/mil_resnet50_epoch{epoch}.pth')
    return best_model_wts, history
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 开始训练
best_wts, history = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=30)

# 保存history为CSV文件
with open('history.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
    for i, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(history['train_loss'], history['train_acc'], history['val_loss'], history['val_acc'])):
        writer.writerow([i+1, train_loss, train_acc, val_loss, val_acc])

import matplotlib.pyplot as plt
#绘制损失曲线
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Loss')
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制准确度曲线
plt.figure(figsize=(10, 5))
plt.title('Validation Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()