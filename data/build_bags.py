BASEDIR_HUMAN_LP = "E:\\myproject\\DL_BASED_NAFLD\\CNN"
BASEDIR_MOUSE_LP = "E:\\models\\Deep_learning_for_liver_NAS_and_fibrosis_scoring-master\\model\\liver"

OUTPUT_DIR = "./outputs/bags/"

import os
from local_tools.my_dataset import LBDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision.utils as vutils
import torch
from PIL import Image
import torchvision.transforms as transforms

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 {directory} 已创建。")
    else:
        print(f"目录 {directory} 已存在。")

import shutil
def copy_file(src, dst):
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # 复制文件
        shutil.copy(src, dst)
        #print(f"文件已成功复制到 {dst}")
    except Exception as e:
        print(f"复制文件时出错：{e}")


def traverse_directory(directory):
    counter = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            #if counter > 10:
            #    break
            if 'ignore' in root:
                continue
            if file.endswith('.png'):
                #print(, file[:-4].split('_'))
                slide, x, y = file[:-4].split('_')
                # 使用示例
                source_file_path = os.path.join(root, file)  # 替换为你的源文件路径
                c1, _, c2 = root.split('\\')[-3:]
                #destination_dir = os.path.join(OUTPUT_DIR, slide)
                destination_file_path = os.path.join(OUTPUT_DIR, slide, file[:-4]+'_'+c1+'_'+c2+'.png')  # 替换为你的目标文件路径
                #ensure_directory_exists(destination_dir)
                copy_file(source_file_path, destination_file_path)

                counter += 1
    print(f'total {counter} files were found')

# 使用示例
directory_path = BASEDIR_HUMAN_LP  # 替换为你的目录路径
traverse_directory(directory_path)