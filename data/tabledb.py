"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torch.utils.data as data
from PIL import Image
from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob
import pandas as pd
from detectron2.data import transforms as T_

main_dir = '/scratch/b/bkantarc/jfern090/Projects/Lytica/'
img_dir = 'Dataset/Images/'
ref_dir = 'Reference/'
output_dir = 'Output/'
model_dir = 'Models/'


class TableDB(data.Dataset):
    def __init__(self, split='train', transform=None):
        file_name = ""
        if split == 'train':
            file_name = main_dir + ref_dir + "train_set1_unsup.csv"
        else:
            file_name = main_dir + ref_dir + "ic13_test_unsup.csv"

        self.csv_file = pd.DataFrame(file_name)
        self.transform = transform 
        self.split = split
        self.resize = T_.ResizeShortestEdge([912, 912], 1600)
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        row = self.csv_file.iloc[index]
        path, target, class_name = row[1], row[2], row[3]
        img = Image.open(path)
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}

        return out

    def get_image(self, index):
        path = self.csv_file.iloc[index, 1]
        img = Image.open(path)
        img = self.resize(img) 
        return img


