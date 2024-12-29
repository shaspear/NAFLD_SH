import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
from local_tools.my_dataset import LBDataset
from torch.utils.data import DataLoader


BASEDIR_HUMAN_LP = "E:\\myproject\\DL_BASED_NAFLD\\CNN"
BASEDIR_MOUSE_LP = "E:\\models\\Deep_learning_for_liver_NAS_and_fibrosis_scoring-master\\model\\liver"

IMG_DIR = 'fibrosis\\train'

tmp_dir = os.path.join(BASEDIR_HUMAN_LP, IMG_DIR)

def tsne_direct(tmpdir):

    transformer = transforms.Compose([
        transforms.Resize((48, 48)),
        #transforms.RandomCrop(48, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])
    lb = LBDataset(tmpdir, transformer)
    print(len(lb))
    trainloader = DataLoader(dataset=lb, batch_size=64, shuffle=True)

    # 直接将图像压缩
    X = np.array([])
    y = np.array([])
    for data_batch, label_batch in iter(trainloader):
        #print(data_batch.shape)
        bs = data_batch.shape[0]
        if len(X.shape) < 2:
            X = data_batch.reshape([bs,-1])
            y = label_batch
        else:
            X = np.concatenate((X, data_batch.reshape([bs,-1])))
            y = np.concatenate((y,label_batch))


    print(X.shape, y.shape)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    for i in np.unique(y):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=f'Class {i}')
    plt.legend()
    plt.savefig(f'./outputs/tsne_{IMG_DIR.split('\\')[0]}.png')
    plt.show()

import torch
import torchvision.models as models


def tsne_features(tmpdir, modelname='ResNet'):
    model = None
    transformer = None
    # 加载预训练的ResNet50模型
    if modelname == 'ResNet':
        model = models.resnet50(pretrained=True)
    elif modelname == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif modelname == 'efficientnet':
        model = models.efficientnet_b4(pretrained=True)
    elif modelname == 'VIT':
        model = models.vit_b_32(pretrained=True)

    if model==None:
        print("Error: Model is not initialized")
    if transformer == None:
        transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomCrop(48, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])
    lb = LBDataset(tmpdir, transformer)
    print(len(lb))
    trainloader = DataLoader(dataset=lb, batch_size=64, shuffle=True)

    # 直接将图像压缩
    X = np.array([])
    y = np.array([])
    for data_batch, label_batch in iter(trainloader):
        #print(data_batch.shape)
        # 假设input_image是一个预处理后的图像张量

        output = model(data_batch)
        if len(X.shape) < 2:
            X = output.detach().numpy()
            y = label_batch.detach().numpy()
        else:
            X = np.concatenate((X, output.detach().numpy()))
            y = np.concatenate((y,label_batch.detach().numpy()))
        print(output.shape)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    for i in np.unique(y):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=f'Class {i}')
    plt.legend()
    plt.savefig(f'./outputs/tsne_{IMG_DIR.split('\\')[0]}_{modelname}.png')
    plt.show()

def tsne_inceptionv3(tmpdir):
    model = None
    transformer = None
    # 加载预训练的ResNet50模型
    model = models.inception_v3(pretrained=True)
    '''for inception-v3'''
    transformer = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    lb = LBDataset(tmpdir, transformer)
    print(len(lb))
    trainloader = DataLoader(dataset=lb, batch_size=64, shuffle=True)

    # 直接将图像压缩
    X = np.array([])
    y = np.array([])
    for data_batch, label_batch in iter(trainloader):
        #print(data_batch.shape)
        # 假设input_image是一个预处理后的图像张量

        output = model(data_batch)
        if len(X.shape) < 2:
            X = output[0].data
            y = label_batch.numpy()
        else:
            X = np.concatenate((X, output[0].data))
            y = np.concatenate((y,label_batch.numpy()))
        print(output[0].data.shape)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    for i in np.unique(y):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=f'Class {i}')
    plt.legend()
    plt.savefig(f'./outputs/tsne_{IMG_DIR.split('\\')[0]}_inceptionv3.png')
    plt.show()

import timm
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
def tsne_vit(tmpdir):

    # 加载预训练的ViT模型
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
    # 设置为评估模式，以便进行推断
    model.eval()
    # 加载图像并进行预处理
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(48, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])
    lb = LBDataset(tmpdir, transformer)
    print(len(lb))
    trainloader = DataLoader(dataset=lb, batch_size=64, shuffle=True)
    # 使用模型进行推断


    # 直接将图像压缩
    X = np.array([])
    y = np.array([])
    for data_batch, label_batch in iter(trainloader):
        # print(data_batch.shape)
        # 假设input_image是一个预处理后的图像张量
        with torch.no_grad():
            output = model(data_batch)
            preds = torch.argmax(output, dim=1)

            if len(X.shape) < 2:
                X = output
                y = label_batch.numpy()
            else:
                X = np.concatenate((X, output))
                y = np.concatenate((y, label_batch.numpy()))
            print(X.shape)
            if X.shape[0]>64*100:
                break

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    for i in np.unique(y):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=f'Class {i}')
    plt.legend()
    plt.savefig(f'./outputs/tsne_{IMG_DIR.split('\\')[0]}_vit.png')
    plt.show()

#tsne_inceptionv3(tmp_dir)
#tsne_features(tmp_dir, 'ResNet')
#tsne_features(tmp_dir, 'vgg19')
#tsne_features(tmp_dir, 'efficientnet')
tsne_vit(tmp_dir)