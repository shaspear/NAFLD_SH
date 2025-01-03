import cv2
from tqdm import tqdm

BASEDIR_HUMAN_LP = "E:/models/NAFLD_LB_PyTorch/CNN"
BASEDIR_MOUSE_LP = "E:\\models\\Deep_learning_for_liver_NAS_and_fibrosis_scoring-master\\model\\liver"

import os
from local_tools.my_dataset import LBDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision.utils as vutils
import torch
from PIL import Image
import torchvision.transforms as transforms

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

'''
查看数据集情况，并用grid的显示显示图片
'''
def view_single_class():
    tmpdir = os.path.join(BASEDIR_HUMAN_LP, 'fibrosis\\train')
    print(tmpdir)
    transformer = transforms.Compose([
        transforms.Resize((48, 48)),
        # transforms.RandomCrop(48, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])
    lb = LBDataset(tmpdir, transformer)
    print(len(lb))
    trainloader = DataLoader(dataset=lb, batch_size=64, shuffle=True)

    # data_batch, label_batch = next(iter(trainloader))
    # img_grid = vutils.make_grid(data_batch, nrow=8, normalize=True, scale_each=True)
    # # 使用transforms.ToPILImage将Tensor转换为PIL图像
    # to_pil_image = transforms.ToPILImage()
    # pil_image = to_pil_image(img_grid)
    #
    # # 使用Image.show显示图像，或者使用Image.save保存图像到文件
    # pil_image.show()  # 显示图像
    # pil_image.save('image.png')  # 保存图像到文件
    batch_counter = 0
    for data_batch, label_batch in iter(trainloader):
        img_grid = vutils.make_grid(data_batch, nrow=8, normalize=True, scale_each=True)
        # 使用transforms.ToPILImage将Tensor转换为PIL图像
        to_pil_image = transforms.ToPILImage()
        pil_image = to_pil_image(img_grid)
        # pil_image.show()
        pil_image.save("./outputs/image_{0}batch.png".format(batch_counter))
        batch_counter += 1


def draw2x2bar(**kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    data = []
    # 创建一个2行2列的子图网格
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # 设置柱状图的宽度
    bar_width = 0.35
    keys = list(kwargs.keys())
    print(keys)
    values = list(kwargs.values())
    print(values)
    for i in range(2):
        for j in range(2):
            ij = i * 2 + j
            k, v = keys[ij], values[ij]
            axs[i, j].bar(v.keys(), v.values(), bar_width)
            axs[i, j].set_title(k)

    # 自动调整子图间距
    plt.tight_layout()

    # 显示图表
    plt.show()
'''
查看多示例中包（bag）的正负标签分布情况
'''
def view_mtl_bags():
    data_dir = r'./outputs/bags'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = MILDataset(data_dir, transform=data_transforms['train'])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    ballooning = {x:0 for x in range(3)}
    inflammation = {x:0 for x in range(6)}
    fibrosis = {x:0 for x in range(4)}
    steatosis = {x:0 for x in range(4)}


    for bags, l1, l2, l3, l4 in tqdm(train_loader):
        bags, l1, l2, l3, l4 = bags.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device)
        ballooning[l1.item()] += 1
        inflammation[l2.item()] += 1
        fibrosis[l3.item()] += 1
        steatosis[l4.item()] += 1
    #
    # print(ballooning)
    # print(inflammation)
    # print(fibrosis)
    # print(steatosis)
    draw2x2bar(ballooning=ballooning, inflammation=inflammation, fibrosis=fibrosis, steatosis=steatosis)

view_mtl_bags()