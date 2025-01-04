import copy
import csv
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, DataLoader
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm

from local_tools.MIL_Dataset_b import MILDataset
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(128),
        #transforms.CenterCrop(112),
        transforms.Resize(112),
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
        #return self.fc(features.max(dim=0)).unsqueeze(dim=0)  # 取bag内所有图片特征的平均值

# 定义Se-ResNet50模型
class SeResNetMIL(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMIL, self).__init__()
        self.se_resnet = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=False)
        in_features = self.se_resnet.fc.in_features
        self.se_resnet.fc = nn.Identity()  # 移除最后的全连接层
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.se_resnet(x)
        return self.fc(features.mean(dim=0)).unsqueeze(dim=0)  # 取bag内所有图片特征的平均值

def load_partial_state_dict(model, state_dict):
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

# 定义ResNet50模型
class MultiTaskResNet50(nn.Module):
    def __init__(self, task1_classes=3, task2_classes=6, task3_classes=4, task4_classes=4):
        super(MultiTaskResNet50, self).__init__()
        #self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=False)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()  # 移除最后的全连接层
        self.task1_fc = nn.Linear(in_features, task1_classes)
        self.task2_fc = nn.Linear(in_features, task2_classes)
        self.task3_fc = nn.Linear(in_features, task3_classes)
        self.task4_fc = nn.Linear(in_features, task4_classes)

    def forward(self, x):
        features = self.resnet50(x)
        task1_output = self.task1_fc(features.mean(dim=0)).unsqueeze(dim=0)
        task2_output = self.task2_fc(features.mean(dim=0)).unsqueeze(dim=0)
        task3_output = self.task3_fc(features.mean(dim=0)).unsqueeze(dim=0)
        task4_output = self.task4_fc(features.mean(dim=0)).unsqueeze(dim=0)
        return task1_output, task2_output, task3_output, task4_output

# 实例化模型
task1_classes = 3  # 第一个任务的类别数
task2_classes = 6 # 第二个任务的类别数
task3_classes = 4 # 第3个任务的类别数
task4_classes = 4 # 第4个任务的类别数
model = MultiTaskResNet50()

# 定义损失函数和优化器
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
criterion3 = nn.CrossEntropyLoss()
criterion4 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 实例化模型

# 加载预训练参数
pretrained_model_path = './results/mil_mtl_se_epoch3_V0.3.pth'
if os.path.exists(pretrained_model_path):
    pretrained_dict = torch.load(pretrained_model_path)
    load_partial_state_dict(model, pretrained_dict)

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
def get_now_time():
    from datetime import datetime
    # 获取当前时间
    now = datetime.now()
    # 格式化当前时间
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("当前时间（年-月-日 时:分:秒）:", formatted_now)
    return  now

def get_lastingtime(time1, time2):
    # 定义两个时间点
    # time1 = datetime(2023, 12, 24, 10, 0, 0)  # 2023年12月24日10时0分0秒
    # time2 = datetime(2023, 12, 25, 12, 0, 0)  # 2023年12月25日12时0分0秒

    # 计算时间差
    time_diff = time2 - time1

    # 打印时间差
    print("时间差:", time_diff)
    print("时间差（秒）:", time_diff.total_seconds())
    print("时间差（分钟）:", time_diff.total_seconds() / 60)
    print("时间差（小时）:", time_diff.total_seconds() / 3600)
    print("时间差（天）:", time_diff.days)

def cal_loss(loss1, loss2, loss3, loss4):
    loss = 0.5 * loss1 + 0.8 * loss2 + loss3 + 2 * loss4
    return loss
def train_model(model, train_loader, valid_loader, optimizer, num_epochs=25):
    history = {
        'train_loss': [],
        'val_acc1': [],
        'val_acc2': [],
        'val_acc3': [],
        'val_acc4': [],
        'val_acc': [],
        'val_loss': []
    }
    starting_time = get_now_time()
    print(f'\nTotal {num_epochs} Epochs, Starting time: {starting_time}')
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for bags, l1, l2, l3, l4 in tqdm(train_loader):
            bags = torch.squeeze(bags)
            bags, l1, l2, l3, l4 = bags.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device)
            optimizer.zero_grad()
            op1, op2, op3, op4 = model(bags)
            #outputs = torch.unsqueeze(outputs, 0)
            #_, preds = torch.max(outputs, 1)
            loss1 = criterion1(op1, l1)
            loss2 = criterion2(op2, l2)
            loss3 = criterion3(op3, l3)
            loss4 = criterion4(op4, l4)
            #loss = 0.5*loss1 + 0.8*loss2 + loss3 + 2*loss4
            loss = cal_loss(loss1, loss2, loss3, loss4)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #running_corrects += torch.sum(preds == labels.data)
            #running_corrects += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        #epoch_acc = running_corrects.double() / len(train_loader)
        print(f'\nEpoch {epoch+1}, Training Loss: {epoch_loss}')
        history['train_loss'].append(epoch_loss)
        #history['train_acc'].append(epoch_acc.item())

        # 验证模型
        model.eval()
        val_loss = 0.0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        with torch.no_grad():
            for bags, l1, l2, l3, l4 in tqdm(valid_loader):
                bags = torch.squeeze(bags)
                bags, l1, l2, l3, l4 = bags.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device)
                op1, op2, op3, op4 = model(bags)
                _, predicted1 = torch.max(op1, 1)
                _, predicted2 = torch.max(op2, 1)
                _, predicted3 = torch.max(op3, 1)
                _, predicted4 = torch.max(op4, 1)
                loss1 = criterion1(op1, l1)
                loss2 = criterion2(op2, l2)
                loss3 = criterion3(op3, l3)
                loss4 = criterion4(op4, l4)
                correct1 += (predicted1 == l1).sum().item()
                correct2 += (predicted2 == l2).sum().item()
                correct3 += (predicted3 == l3).sum().item()
                correct4 += (predicted4 == l4).sum().item()
                val_loss += loss1 + loss2 + loss3 + loss4
        accuracy1 = correct1 / len(valid_dataset)
        accuracy2 = correct2 / len(valid_dataset)
        accuracy3 = correct3 / len(valid_dataset)
        accuracy4 = correct4 / len(valid_dataset)
        epoch_acc = 0.3*accuracy1 + 0.4*accuracy2 + 0.8*accuracy3 + accuracy4
        epoch_loss = val_loss.item() / len(valid_loader)
        print(f'\nValidation  Accuracy1: {accuracy1:.4f}, '
              f'Accuracy2: {accuracy2:.4f}',
              f'Accuracy3: {accuracy3:.4f}',
              f'Accuracy4: {accuracy4:.4f}',
              f'Val loss: {epoch_loss:.4f}',
              )
        history['val_acc1'].append(accuracy1)
        history['val_acc2'].append(accuracy2)
        history['val_acc3'].append(accuracy3)
        history['val_acc4'].append(accuracy4)
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        # 深拷贝模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # 保存模型
            torch.save(best_model_wts, f'results/mil_mtl_se_epoch{epoch}_V0.3.pth')
    endingtime = get_now_time()
    get_lastingtime(starting_time, endingtime)
    return best_model_wts, history
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 开始训练
best_wts, history = train_model(model, train_loader, valid_loader, optimizer, num_epochs=50)
from datetime import date
# 保存history为CSV文件
with open(f'history_mil_mtl_{date.today()}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Val Acc1', 'Val Acc2', 'Val Acc3', 'Val Acc4', 'Val loss', 'Val acc'])
    for i, (train_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_loss, val_acc) in enumerate(
            zip(history['train_loss'], history['val_acc1'], history['val_acc2'], history['val_acc3'], history['val_acc4'], history['val_loss'], history['val_acc'])):
        writer.writerow([i+1, train_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_loss, val_acc ])

# import matplotlib.pyplot as plt
# #绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.title('Training and Validation Loss')
# plt.plot(history['train_loss'], label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 绘制准确度曲线
# plt.figure(figsize=(10, 5))
# plt.title('Validation Accuracy')
# plt.plot(history['val_acc1'], label='Validation Accuracy1')
# plt.plot(history['val_acc2'], label='Validation Accuracy2')
# plt.plot(history['val_acc3'], label='Validation Accuracy3')
# plt.plot(history['val_acc4'], label='Validation Accuracy4')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()