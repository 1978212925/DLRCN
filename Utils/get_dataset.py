import os
import sys

from torchvision.transforms import transforms

from Utils.data_transform import lung_transform

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import PIL.Image
import pandas as pd
import torch.utils.data
from torch.utils.data.dataset import Dataset
from Utils import data_transform
import cv2
import numpy as np


def get_csv_data(csv_path):
    df = pd.read_csv(csv_path, encoding='GBK', usecols=[0, 1])
    # df.drop([0], inplace=True)
    label = df.to_numpy()
    return label


class LungDataset(Dataset):
    def __init__(self, img_path_list, label_list, fat_img_list, transform, mode_select=0):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform
        self.mode_select = mode_select
        # self.fat_list = fat_list
        self.fat_img_list = fat_img_list
        # self.img_sex_list = img_sex_list

    def __getitem__(self, index):
        # 拿到roi图像路径dataset/roi/img_train/675577.png
        roi_path = self.img_path_list[index]
        merge_path = roi_path.replace('roi', 'merge_region')
        peritumor_path = merge_path.replace("merge_region", "peritumor")
        intratumor_path = merge_path.replace("merge_region", "intratumor")
        # 拿到脂肪图像路径
        fat_intrathoracic_path = self.fat_img_list[index]
        fat_mediastinum_path = fat_intrathoracic_path.replace("fat_intrathoracic", "fat_mediastinum")
        intrathoractic_mediastinum_path = fat_intrathoracic_path.replace("fat_intrathoracic",
                                                                         "intrathoractic_mediastinum")
        label = self.label_list[index]
        # 只使用roi原图像
        if self.mode_select == 0:
            roi = cv2.imread(roi_path)
            roi = PIL.Image.fromarray(roi)
            data = self.transform(roi)
            data = data
        # 只使用瘤周图像
        elif self.mode_select == 1:
            peritumor = cv2.imread(peritumor_path)
            peritumor = PIL.Image.fromarray(peritumor)
            data = self.transform(peritumor)
            data = data
        # 只使用瘤内图像
        elif self.mode_select == 2:
            intratumor = cv2.imread(intratumor_path)
            intratumor = PIL.Image.fromarray(intratumor)
            data = self.transform(intratumor)
            data = data
        # 只使用瘤周瘤内混合图像
        elif self.mode_select == 3:
            merge = cv2.imread(merge_path)
            merge = PIL.Image.fromarray(merge)
            data = self.transform(merge)
            data = data
        # 瘤周瘤内混合三图像合并
        elif self.mode_select == 4:
            merge = cv2.imread(merge_path, flags=cv2.IMREAD_GRAYSCALE)
            peritumor = cv2.imread(peritumor_path, flags=cv2.IMREAD_GRAYSCALE)
            intratumor = cv2.imread(intratumor_path, flags=cv2.IMREAD_GRAYSCALE)
            # cv2.imshow("1", peritumor)
            # cv2.imshow("2", intratumor)
            # cv2.imshow("3", merge)
            # cv2.waitKey(-1)
            merge = np.expand_dims(merge, 2)
            peritumor = np.expand_dims(peritumor, 2)
            intratumor = np.expand_dims(intratumor, 2)

            data = np.concatenate((merge, peritumor, intratumor), 2)
            # cv2.imshow("4", data)
            # cv2.waitKey(-1)
            data = PIL.Image.fromarray(data)
            data = self.transform(data)
            data = data
        # 只使用脂肪图像
        elif self.mode_select == 5:
            fat_intrathoracic = cv2.imread(fat_intrathoracic_path, flags=cv2.IMREAD_COLOR)
            fat_img = fat_intrathoracic
            fat_img = PIL.Image.fromarray(fat_img)
            fat_img = self.transform(fat_img)
            data = fat_img
        else:
            print("mode_select error")
            data = None
            fat_img = None
            data = [data, fat_img]


        return data, label

    def __len__(self):
        return len(self.img_path_list)


def getdataset(csv_path, nodule_path, fat_path, transform, mode_select=0, is_augment=False):
    label_numpy = get_csv_data(csv_path)
    img_list = os.listdir(nodule_path)
    img_content_path = []
    for ls in img_list:
        img_content_path.append(os.path.join(nodule_path, ls))
    img_num = label_numpy[:, 0]
    img_num = img_num.astype(np.int64)
    img_label = label_numpy[:, 1].astype('int')
    img_num, img_label = img_num.tolist(), img_label.tolist()
    new_img_list = []
    new_label_list = []
    new_fat_img_list = []

    for img_name in img_list:
        img_name_int = int(img_name.replace(".png", ""))
        try:
            img_name_index = img_num.index(img_name_int)
            new_label_list.append(img_label[img_name_index])
            new_img_list.append(os.path.join(nodule_path, img_name))
            new_fat_img_list.append(os.path.join(fat_path, img_name))
        except Exception as err:
            pass

    print(len(new_img_list), len(new_label_list), len(new_fat_img_list))
    if is_augment:
        aug_transform = data_transform.aug_transform
        totaldataset1 = LungDataset(new_img_list, new_label_list, new_fat_img_list, transform,
                                    mode_select=mode_select)
        only_0_img_list = []
        only_0_label_list = []
        only_0_fat_list = []
        only_0_fat_img_list = []
        for i in range(len(new_label_list)):
            if new_label_list[i] == 0:
                only_0_img_list.append(new_img_list[i])
                only_0_label_list.append(new_label_list[i])
                # only_0_fat_list.append(new_fat_list[i])
                only_0_fat_img_list.append(new_fat_img_list[i])
        totaldataset2 = LungDataset(only_0_img_list, only_0_label_list, only_0_fat_img_list, aug_transform,
                                    mode_select=mode_select)
        totaldataset = totaldataset1+totaldataset2
    else:
        totaldataset = LungDataset(new_img_list, new_label_list, new_fat_img_list, transform, mode_select=mode_select)
    return totaldataset


if __name__ == '__main__':
    transformer = data_transform.transform_none
    total_dataset = getdataset("../dataset/lung.csv", "../dataset/roi/img_train", "../dataset/fat_intrathoracic",
                               lung_transform['train'], mode_select=5)
    print(total_dataset[30])
    # fat = get_fat_index("../dataset/muscle_fat.csv")
    # print(fat)
