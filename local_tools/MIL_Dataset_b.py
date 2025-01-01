import math

import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
'''
Liver Biopsy bags
ballooning:2
fibrosis:5
inflammation:3
steatosis:5~25->0,30~50->1, 55~75->2
'''
class MILDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"0": 0, "1": 1, '2':2, '3':3, '4':4, 'ignore':5}
        #self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info_balanced(data_dir)
        random.shuffle(self.data_info)
        self.transform = transform
        self.dataset_size = len(self.data_info)

        #print(len(self.data_info))

    def __getitem__(self, index):
        bag, label = self.data_info[index]
        bag_images = []
        for img_path in bag:

            image = Image.open(img_path).convert('RGB')     # 0~255
            if self.transform:
                image = self.transform(image)
            bag_images.append(image)
            #bag_images.extend(image)
        bag_images = torch.stack(bag_images)
        b,c,h,w = bag_images.shape

        return bag_images, label['ballooning'], label['fibrosis'], label['inflammation'], label['steatosis']

    def __len__(self):
        return len(self.data_info)

    '''
    # this function has been discarded
    '''
    @staticmethod
    def sample_images_discarded(image_files, group_size=10):
        # 如果图片数量少于group_size，则直接返回所有图片
        if len(image_files) < group_size:
            return [image_files]

        # 初始化图片组列表
        image_groups = []
        # 不断地从剩余图片中随机采样group_size张图片，直到所有图片都被分配
        while len(image_files) >= group_size:
            lDict = {'ballooning':0, 'inflammation':0, 'steatosis':0, 'fibrosis':0}
            group = random.sample(image_files, group_size)

            for fname in group:
                t, s = fname[:-4].split('_')[-2:]
                s = int(s)
                if s > lDict[t]:
                    lDict[t] = s
            image_groups.append((group, lDict))
            # 从剩余图片中移除已经分配的图片
            image_files = [f for f in image_files if f not in group]

        # 不够1组的情况下，舍弃 by yum
        # 如果最后剩余的图片少于group_size，将它们作为一个组添加
        # if image_files:
        #     image_groups.append(image_files)

        return image_groups

    @staticmethod
    def calc_dict_len(slide_id, d, classname):
        counter = 0

        for k, v in  d.items():
            counter += len(v)
            print(f"{classname} Label:{k} is {len(v)}")
        print(f"slide:{slide_id} has total {counter} {classname}")
        return counter
    @staticmethod
    def parse_data_intodicts(img_names):

        if len(img_names)<1:
            return {}, {}, {}, {}
        slide_id = img_names[0].split('\\')[-2]

        b = {x: [] for x in range(3)}  # ballooning + ignore
        f = {x: [] for x in range(6)}  # fibrosis + ignore
        i = {x: [] for x in range(4)}  # inflammation + ignore
        s = {x: [] for x in range(4)}  # steatosis + ignore
        for fn in img_names:
            # if "ignore" in fn:
            #     print(fn)
            t, l = fn[:-4].split('_')[-2:]

            if t == 'steatosis':
                l = int(l) if l.isdigit() else 3
                if l > 50:
                    s[2].append(fn)
                elif l > 25:
                    s[1].append(fn)
                else:
                    s[0].append(fn)
            elif t == "ballooning":
                l = int(l) if l.isdigit() else 2
                b[l].append(fn)
            elif t == "inflammation":
                l = int(l) if l.isdigit() else 3
                i[l].append(fn)
            else:
                l = int(l) if l.isdigit() else 5
                f[l].append(fn)
        MILDataset.calc_dict_len(slide_id, b, "ballooning")
        MILDataset.calc_dict_len(slide_id, f, "fibrosis")
        MILDataset.calc_dict_len(slide_id, i, "inflammation")
        MILDataset.calc_dict_len(slide_id, s, "steatosis")
        return b, f, i, s

    @staticmethod
    def parse_data_PosNeg(img_names):

        if len(img_names) < 1:
            return [], [], [], [], []
        slide_id = img_names[0].split('\\')[-2]

        b = []  # ballooning
        f = []  # fibrosis
        i = []  # inflammation
        s = []  # steatosis
        neg = []
        for fn in img_names:
            if "ignore" in fn:
                if slide_id not in ['38', '15', '161', '167', '173', '184', '215', '229', '238', '247', '253', '255', '256', '258']:
                    continue
                 #print(fn)
            t, l = fn[:-4].split('_')[-2:]

            if t == 'steatosis':
                l = int(l) if l.isdigit() else 3
                if l > 5:
                    s.append(fn)
                else:
                    neg.append(fn)
            elif t == "ballooning":
                l = int(l) if l.isdigit() else 2
                if l == 1:
                    b.append(fn)
                else:
                    neg.append(fn)
            elif t == "inflammation":
                l = int(l) if l.isdigit() else 3
                if l > 0 and l < 3:
                    i.append(fn)
                else:
                    neg.append(fn)
            else:
                l = int(l) if l.isdigit() else 5
                if l > 0 and l < 5:
                    f.append(fn)
                else:
                    neg.append(fn)
        print(f"Slide:{slide_id} has positive tiles count: b-{len(b)}, f-{len(f)}, i-{len(i)}, s-{len(s)}")
        return b, f, i, s, neg
    @staticmethod
    def get_img_info_balanced(data_dir):
        slides = {}
        image_groups = list()
        group_size = 16
        sample_expand = 1#包膨胀倍数

        for root, dirs, _ in os.walk(data_dir):

            # 遍历类别
            for sub_dir in dirs:
                # if sub_dir != '10':# only read images of folder 1
                #     continue
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                img_names = [os.path.join(root, sub_dir, f) for f in img_names]
                slides[sub_dir] = img_names
                # if len(slides)>10:
                #     break
        for key in slides.keys():
            image_files = slides[key]

            num_groups = math.ceil(len(image_files)*sample_expand/group_size)
            sample_groups = MILDataset.sample_balanced_groups(image_files, num_groups, group_size)
            image_groups.extend(sample_groups)
            # if len(image_groups) > 50: #调试用
            #     break
            # # 遍历图片
            # for i in range(len(img_names)):
            #     img_name = img_names[i]
            #     path_img = os.path.join(root, sub_dir, img_name)
            #     label = fibrosis_label[sub_dir]
            #     data_info.append((path_img, int(label)))
        #print(counter)
        return image_groups
    '''
    Parameter：
    lDict 是以label为键，值为image path的list
    return:
    尽量选出1个阳性样本，如果没有，则返回阴性样本，都没有返回ignore样本
    '''
    @staticmethod
    def get_positive_tile(lDict:dict):
        length = len(lDict)
        for i in range(1, length-1):
            if len(lDict[i]) > 0:
                return lDict[i].pop()
        if len(lDict[0] > 0):
            return lDict[0].pop()
        if len(lDict[length-1])>0:
            return lDict[length-1].pop()
        return None

    @staticmethod
    def get_positive_tile(lDict: list, k: int):
        length = len(lDict)
        if len(lDict) > 0:
            #return lDict.pop()
            return random.sample(lDict, 1)
        return None

    @staticmethod
    def sample_tiles(b:[], f:[], i:[], s:[], k: int):
        group = []

        ffn = MILDataset.get_positive_tile(b, k)
        if ffn:
            group.extend(ffn)
        ffn = MILDataset.get_positive_tile(f, k)
        if ffn:
            group.extend(ffn)
        ffn = MILDataset.get_positive_tile(i, k)
        if ffn:
            group.extend(ffn)
        ffn = MILDataset.get_positive_tile(s, k)
        if ffn:
            group.extend(ffn)
        return group



    @staticmethod
    def sample_balanced_groups(image_files, num_groups=10, group_size=3):
        # 获取所有图片文件名
        # image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        random.shuffle(image_files)  # 打乱顺序以确保随机性
        #b, f, i, s = MILDataset.parse_data_intodicts(image_files)
        b, f, i, s, neg = MILDataset.parse_data_PosNeg(image_files)
        if len(b) == 0:
            num_groups = math.ceil(num_groups*0.8)
        if len(f) == 0:
            num_groups = math.ceil(num_groups*0.8)
        if len(i) == 0:
            num_groups = math.ceil(num_groups*0.8)
        if len(s) == 0:
            num_groups = math.ceil(num_groups*0.8)
        # 初始化图片组列表
        image_groups = []
        seen_groups = set()  # 用于存储已见过的组，以确保不重复
        counter = 0
        counter_f5 = 0 #控制fibrosis=5的数量
        counter_s0 = 0  # 控制steatosis=0的数量
        # 不断地随机采样，直到获得所需数量的不重复组
        while len(image_groups) < num_groups:
            if counter > num_groups:
                break
            if len(image_files) < group_size:
                break
            group = MILDataset.sample_tiles(b, f, i, s, 1)
            #print(f"sample_tiles get {len(group)} positive samples")
            if len(neg)<group_size-len(group):
                print(image_files[0])
                if '\\258\\' in image_files[0]:
                    group = random.sample(image_files, group_size)
            else:
                group.extend(random.sample(neg, group_size-len(group)))
            lDict = {'ballooning': -1, 'inflammation': -1, 'steatosis': -1, 'fibrosis': -1}
            # 将组转换为元组，以便可以加入到集合中进行比较
            group_tuple = tuple(sorted(group))

            # 如果这个组之前没有出现过，则添加到结果中
            if group_tuple not in seen_groups:
                for fname in group_tuple:
                    t, l = fname[:-4].split('_')[-2:]
                    if t == 'steatosis':
                        l = int(l) if l.isdigit() else -1
                        if l > 50:
                            lDict[t] = 2
                        elif l > 25:
                            lDict[t] = 1
                        elif l > 0:
                            lDict[t] = 0
                        else:
                            lDict[t] = 3
                    else:
                        if l.isdigit():
                            l = int(l)
                            if(l > lDict[t]) :
                                lDict[t] = l
                for k,v in lDict.items():
                    if v == -1:
                        if k == 'ballooning':
                            lDict[k] = 2
                        elif k == 'fibrosis':
                            lDict[k] = 5
                        elif k == 'inflammation' or k == 'steatosis':
                            lDict[k] = 3

                if lDict['fibrosis'] == 5 and len(f) > 0:
                    if counter_f5 > num_groups/3:
                        continue
                    else:
                        counter_f5 += 1
                        image_groups.append((group, lDict))
                elif lDict['steatosis'] == 0 and len(s) > 0:
                    if counter_s0 > num_groups:
                        continue
                    else:
                        counter_s0 += 1
                        image_groups.append((group, lDict))
                else:
                    image_groups.append((group, lDict))


                seen_groups.add(group_tuple)
            counter += 1
            # 如果已经获得了所需数量的组，则退出循环
            # if len(image_groups) >= num_groups:
            #     break

        return image_groups

    @staticmethod
    def get_img_info(data_dir):
        slides = {}
        image_groups = list()
        group_size = 16
        sample_expand = 1  # 包膨胀倍数

        for root, dirs, _ in os.walk(data_dir):

            # 遍历类别
            for sub_dir in dirs:
                if sub_dir != '10':  # only read images of folder 1
                    continue
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                img_names = [os.path.join(root, sub_dir, f) for f in img_names]
                slides[sub_dir] = img_names
                # if len(slides)>10:
                #     break
        for key in slides.keys():
            image_files = slides[key]
            num_groups = math.ceil(len(image_files) * sample_expand / group_size)
            sample_groups = MILDataset.sample_unique_groups(image_files, num_groups, group_size)
            image_groups.extend(sample_groups)
            # if len(image_groups) > 50: #调试用
            #     break
            # # 遍历图片
            # for i in range(len(img_names)):
            #     img_name = img_names[i]
            #     path_img = os.path.join(root, sub_dir, img_name)
            #     label = fibrosis_label[sub_dir]
            #     data_info.append((path_img, int(label)))
        # print(counter)
        return image_groups
    @staticmethod
    def sample_unique_groups(image_files, num_groups=10, group_size=3):
        # 获取所有图片文件名
        #image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        random.shuffle(image_files)  # 打乱顺序以确保随机性

        # 初始化图片组列表
        image_groups = []
        seen_groups = set()  # 用于存储已见过的组，以确保不重复
        counter = 0
        # 不断地随机采样，直到获得所需数量的不重复组
        while len(image_groups) < num_groups:
            # 随机选择3张图片
            if counter > num_groups:
                break
            #print(len(image_files), group_size, image_files[0])
            if len(image_files) < group_size:
                break
            group = random.sample(image_files, group_size)
            lDict = {'ballooning': 0, 'inflammation': 0, 'steatosis': 0, 'fibrosis': 0}
            # 将组转换为元组，以便可以加入到集合中进行比较
            group_tuple = tuple(sorted(group))

            # 如果这个组之前没有出现过，则添加到结果中
            if group_tuple not in seen_groups:
                for fname in group_tuple:
                    t, s = fname[:-4].split('_')[-2:]
                    s = int(s)
                    if t ==  'steatosis':
                        if s > 50:
                            lDict[t] = 2
                        elif s> 25:
                            lDict[t] = 1
                        else:
                            lDict[t] = 0
                    elif (s > lDict[t]):
                        lDict[t] = s
                image_groups.append((group, lDict))
                seen_groups.add(group_tuple)
            counter += 1
            # 如果已经获得了所需数量的组，则退出循环
            # if len(image_groups) >= num_groups:
            #     break

        return image_groups




# 自定义数据集类，用于加载MIL数据
class MILDataset_KIMI(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.bags = []
        self.labels = []
        # 假设数据集的目录结构为：data_dir/bag1/img1.jpg, data_dir/bag1/img2.jpg, ..., data_dir/bag2/img1.jpg, ...
        for bag in os.listdir(data_dir):
            bag_path = os.path.join(data_dir, bag)
            self.bags.append([os.path.join(bag_path, img) for img in os.listdir(bag_path)])
            self.labels.append(int(bag) - 1)  # 假设文件夹名称是类别标签

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        bag_images = []
        for img_path in bag:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            bag_images.append(image)
        return bag_images, self.labels[idx]
