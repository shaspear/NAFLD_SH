import copy
import csv

import torchvision.models as models
#from torch import nn


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
#from pretrainedmodels import se_resnet50
import os
import torch.hub
from local_tools.my_dataset import LBDataset
import matplotlib.pyplot as plt
# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
BASEDIR_HUMAN_LP = "E:\\myproject\\DL_BASED_NAFLD\\CNN"
BASEDIR_MOUSE_LP = "E:\\models\\Deep_learning_for_liver_NAS_and_fibrosis_scoring-master\\model\\liver"
IMG_DIR = 'fibrosis\\train'
tmp_dir = os.path.join(BASEDIR_HUMAN_LP, IMG_DIR)
# 数据加载
data_dir = os.path.join(BASEDIR_HUMAN_LP, 'fibrosis')
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                   for x in ['train', 'val']}
image_datasets = {x: LBDataset(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']
}
# dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=1)
#               for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].label_name
# 模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = se_resnet50(num_classes=10)  # 假设有10个类别



# 训练模型
def train_model(num_epochs=25, pretrained=True):
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc':[]
    }
    if pretrained:
        se_resnet = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=True)
        se_resnet.fc = nn.Linear(se_resnet.fc.in_features, 6)
        for param in se_resnet.parameters():
            param.requires_grad = False
        for param in se_resnet.fc.parameters():
            param.requires_grad = True
    else:
        se_resnet = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=False)
        se_resnet.fc = nn.Linear(se_resnet.fc.in_features, 6)
    model = se_resnet.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 跟踪历史，仅在训练阶段
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段才反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                history['train_loss'].append(epoch_loss)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            else:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().detach().numpy())
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                # 深拷贝模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # 保存模型
                    torch.save(best_model_wts, f'results'
                                               f''
                                               f'/se_resnet50_epoch{epoch}.pth')
            print()
    return best_model_wts, history

# 调用训练函数
best_wts, history = train_model(num_epochs=100, pretrained=False)
# 保存history为CSV文件
with open('history.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val Acc'])
    for i, (train_loss, val_loss, val_acc) in enumerate(zip(history['train_loss'], history['val_loss'], history['val_acc'])):
        writer.writerow([i+1, train_loss, val_loss, val_acc])

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

