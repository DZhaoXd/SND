import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2
import pickle
import json

def read_json_file_standard(file_path):
    try:
        with open(file_path, 'r') as file:
            # 读取 JSON 数据
            data = json.load(file)
            print("JSON 数据加载成功！")
            return data
    except FileNotFoundError:
        print("文件未找到：", file_path)
        return None
    except json.JSONDecodeError as e:
        print("JSON 解析错误：", e)
        return None
        

class SDllama70bDataSet(data.Dataset):
    def __init__(self,
        data_root,
        data_list,
        max_iters=None,
        num_classes=19, 
        split="train",
        transform=None,
        ignore_label=255,
        debug=False,):

        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        content = read_json_file_standard(data_list)
        self.img_ids = [i_id['file_name'].split('/')[-1] for i_id in content]
        
        if max_iters is not None:
            self.label_to_file, self.file_to_label = pickle.load(open(osp.join(data_root, "gtav_label_info.p"), "rb"))
            self.img_ids = []
            SUB_EPOCH_SIZE = 3000
            tmp_list = []
            ind = dict()
            for i in range(self.NUM_CLASS):
                ind[i] = 0
            for e in range(int(max_iters/SUB_EPOCH_SIZE)+1):
                cur_class_dist = np.zeros(self.NUM_CLASS)
                for i in range(SUB_EPOCH_SIZE):
                    if cur_class_dist.sum() == 0:
                        dist1 = cur_class_dist.copy()
                    else:
                        dist1 = cur_class_dist/cur_class_dist.sum()
                    w = 1/np.log(1+1e-2 + dist1)
                    w = w/w.sum()
                    c = np.random.choice(self.NUM_CLASS, p=w)

                    if ind[c] > (len(self.label_to_file[c])-1):
                        np.random.shuffle(self.label_to_file[c])
                        ind[c] = ind[c]%(len(self.label_to_file[c])-1)

                    c_file = self.label_to_file[c][ind[c]]
                    tmp_list.append(c_file)
                    ind[c] = ind[c]+1
                    cur_class_dist[self.file_to_label[c_file]] += 1

            self.img_ids = tmp_list

        for name in self.img_ids:
            self.data_list.append(
                {
                    "img": os.path.join(self.data_root, name),
                    "name": name.split('/')[-1],
                }
            )
        
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
        
        print('length of SDllama70bDataSet', len(self.data_list))

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.trainid2name = {
            0:"road",
            1:"sidewalk",
            2:"building",
            3:"wall",
            4:"fence",
            5:"pole",
            6:"light",
            7:"sign",
            8:"vegetation",
            9:"terrain",
            10:"sky",
            11:"person",
            12:"rider",
            13:"car",
            14:"truck",
            15:"bus",
            16:"train",
            17:"motocycle",
            18:"bicycle"
        }
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        #label = np.array(Image.open(datafiles["label"]),dtype=np.float32)
        name = datafiles["name"]

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(image.size, dtype=np.float32)
        """
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)
        """
        
        if self.transform is not None:
            image, label, _ = self.transform(image, label_copy)

        return image, label, name
