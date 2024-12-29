import cv2

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

tmpdir = os.path.join(BASEDIR_HUMAN_LP, 'fibrosis\\train')
print(tmpdir)
transformer = transforms.Compose([
    transforms.Resize((48, 48)),
    #transforms.RandomCrop(48, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])
lb = LBDataset(tmpdir, transformer)
print(len(lb))
trainloader = DataLoader(dataset=lb, batch_size=64, shuffle=True)

#data_batch, label_batch = next(iter(trainloader))
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
    #pil_image.show()
    pil_image.save("./outputs/image_{0}batch.png".format(batch_counter))
    batch_counter+=1