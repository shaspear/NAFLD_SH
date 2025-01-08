import itertools
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
            #print(f"{classname} Label:{k} is {len(v)}")
        #print(f"slide:{slide_id} has total {counter} {classname}")
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
        #print(f"Slide:{slide_id} has positive tiles count: b-{len(b)}, f-{len(f)}, i-{len(i)}, s-{len(s)}")
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
                # if sub_dir != '444':# only read images of folder 1
                #     continue
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png') and "fibrosis" not in x, img_names))
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
        if len(lDict) > k:
            #return lDict.pop()
            return random.sample(lDict, k)
        else:
            return lDict
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

    '''
    '''
    @staticmethod
    def get_k_sample(data, k):
        combinations = list(itertools.combinations(data, k))
        cblist = [list(combo) for combo in combinations]
        return cblist
    '''
    从阳性病例中，枚举出所有的可能性
    threshold: 采样重复计数阈值，超过阈值后不再继续采样。可以控制单个slide采样的数量，值越大，采样的数量越多，默认100。
    '''
    @staticmethod
    def sample_enumerate(b:[], f:[], i:[], s:[], bagsize=8, threshold=100):
        group = []
        seen_groups = set()  # 用于存储已见过的组，以确保不重复
        seen_count = 0
        l_b, l_f, l_i, l_s = len(b), len(f), len(i), len(s)
        l_total = len(b + f + i + s)
        flag_b1f1234 = 10
        for k in range(4, math.ceil(bagsize/3)):
            while True:
                if len(group) > flag_b1f1234:
                    break

                tmp_smp = []
                # b list
                if l_b > 0:
                    s_b = math.floor(k * l_b/l_total)
                    tmp_smp.extend(MILDataset.get_positive_tile(b, s_b))
                    for fn in tmp_smp:
                        if "ballooning_1" in fn:
                            flag_b1f1234 = 50
                # f list
                if l_f > 0:
                    s_f = math.ceil(k * l_f / l_total)
                    tmp_smp.extend(MILDataset.get_positive_tile(f, s_f))
                    for fn in tmp_smp:
                        if ("fibrosis_1" in fn) or ("fibrosis_2" in fn) or ("fibrosis_3" in fn) or ("fibrosis_4" in fn):
                            flag_b1f1234 = 200
                # i list
                if l_i > 0:
                    s_i = math.ceil(k * l_i / l_total)
                    tmp_smp.extend(MILDataset.get_positive_tile(i, s_i))
                # s list
                if l_s > 0:
                    s_s = math.ceil(k * l_s / l_total)
                    tmp_smp.extend(MILDataset.get_positive_tile(s, s_s))
                # 将组转换为元组，以便可以加入到集合中进行比较
                group_tuple = tuple(sorted(tmp_smp))
                # 如果这个组之前没有出现过，则添加到结果中
                if group_tuple not in seen_groups:
                    group.append(tmp_smp)
                    seen_groups.add(group_tuple)
                else:
                    seen_count += 1
                    if seen_count > threshold:
                        break

        #print(len(group))
        return group
    '''
    根据采集到的阳性样本数量，采集适量的阴性样本，以平衡数据集的分布
    '''
    @staticmethod
    def sample_neg_groups(image_files, num_groups=10, group_size=3, threshold=50):
        image_groups = []
        seen_groups = set()
        counter = 0
        seen_count = 0
        expand_num = 1
        if len(image_files) < group_size:
            return image_groups
        while counter < num_groups:
            lDict = {'ballooning': -1, 'inflammation': -1, 'steatosis': -1, 'fibrosis': -1}

            groups = []
            group = random.sample(image_files, group_size)
            for fn in group:
                #这部分代码是为了调节数据类别分布不平衡，但之前的统计未计入阴性病例，所以这段代码不需要。
                #暂时放在这里，以备后用。
                if "fibrosis_0" in fn:
                    group = list(filter(lambda
                                            x: 'fibrosis' in x,
                                        group))
                    expand_num = 1
                    break
            if len(image_files) > group_size-len(group):
                while expand_num:
                    tmp = group.copy()
                    tmp.extend(random.sample(image_files, group_size - len(group)))
                    groups.append(tmp)
                    expand_num -= 1
            else:
                break
            for group in groups:
                group_tuple = tuple(sorted(group))
                if group_tuple not in seen_groups:
                    for fname in group_tuple:
                        t, l = fname[:-4].split('_')[-2:]
                        if t == 'steatosis':
                            l = int(l) if l.isdigit() else -1
                            if l > 40:
                                lDict[t] = 2
                            elif l > 10:
                                lDict[t] = 1
                            elif l > 0:
                                lDict[t] = 0
                            else:
                                lDict[t] = 3
                        else:
                            if l.isdigit():
                                l = int(l)
                                if (l > lDict[t]):
                                    lDict[t] = l
                    for k, v in lDict.items():
                        if v == -1:
                            if k == 'ballooning':
                                lDict[k] = 2
                            elif k == 'fibrosis':
                                lDict[k] = 5
                            elif k == 'inflammation' or k == 'steatosis':
                                lDict[k] = 3
                    MILDataset.b_c[lDict['ballooning']] += 1
                    MILDataset.f_c[lDict['fibrosis']] += 1
                    MILDataset.i_c[lDict['inflammation']] += 1
                    MILDataset.s_c[lDict['steatosis']] += 1
                    image_groups.append((group, lDict))
                    seen_groups.add(group_tuple)
                    counter += 1
            else:
                seen_count += 1
                if seen_count > threshold:
                    break
        return image_groups

    b_c = {x: 0 for x in range(3)}
    f_c = {x: 0 for x in range(6)}
    i_c = {x: 0 for x in range(4)}
    s_c = {x: 0 for x in range(4)}
    @staticmethod
    def sample_balanced_groups(image_files, num_groups=10, group_size=3):

        random.shuffle(image_files)  # 打乱顺序以确保随机性
        #b, f, i, s = MILDataset.parse_data_intodicts(image_files)
        b, f, i, s, neg = MILDataset.parse_data_PosNeg(image_files)
        # 初始化图片组列表
        image_groups = []
        seen_groups = set()  # 用于存储已见过的组，以确保不重复
        counter = 0
        sample_pos_tiles = MILDataset.sample_enumerate(b, f, i, s, group_size, 5)
        for spt in sample_pos_tiles:
            group = spt
            groups = []
            expand_num = 1
            for fn in group:
                if "ballooning_1" in fn:
                    expand_num = 20

                if ("fibrosis_1" in fn) or ("fibrosis_2" in fn) or ("fibrosis_3" in fn) or ("fibrosis_4" in fn):
                    expand_num = 20
            if len(neg)>group_size-len(spt):
                while expand_num:
                    group.extend(random.sample(neg, group_size-len(spt)))
                    groups.append(group)
                    expand_num -= 1
            else:
                break

            for group in groups:
                lDict = {'ballooning': -1, 'inflammation': -1, 'steatosis': -1, 'fibrosis': -1}
                # 将组转换为元组，以便可以加入到集合中进行比较
                group_tuple = tuple(sorted(group))

                # 如果这个组之前没有出现过，则添加到结果中
                if group_tuple not in seen_groups:
                    for fname in group_tuple:
                        t, l = fname[:-4].split('_')[-2:]
                        if t == 'steatosis':
                            l = int(l) if l.isdigit() else -1
                            if l > 40:
                                lDict[t] = 2
                            elif l > 10:
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
                    MILDataset.b_c[lDict['ballooning']] += 1
                    MILDataset.f_c[lDict['fibrosis']] += 1
                    MILDataset.i_c[lDict['inflammation']] += 1
                    MILDataset.s_c[lDict['steatosis']] += 1
                    image_groups.append((group, lDict))
                    seen_groups.add(group_tuple)
                counter += 1
        neg_groups = MILDataset.sample_neg_groups(neg, len(sample_pos_tiles), group_size)
        #print(len(neg_groups))
        image_groups.extend(neg_groups)
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
