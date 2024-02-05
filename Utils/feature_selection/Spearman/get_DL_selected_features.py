# encoding=utf-8
import copy

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from Utils import get_five_fold_dataset
from Utils.data_transform import lung_transform
from Utils.get_DL_final_conv_featuremap import get_last_layer_feature_map_from_dataloader
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

from Utils.get_dataset import getdataset
import SimpleITK as sitk
from radiomics import featureextractor



def normalize_image(image):
    # 将图像像素值归一化到 0-1 范围
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # 将归一化后的图像乘以 255，以便转换为 0-255 整数范围
    normalized_image = (normalized_image * 255).astype(np.uint8)
    return normalized_image
def GLCM_image(image):
    image = (image/16).astype(np.uint8)
    return image
# 1. H_uniformity (3 features)
def calculate_uniformity(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= np.sum(hist)
    uniformity = np.sum(np.square(hist))
    return uniformity


# 2. H_energy (2 features)
def calculate_energy(image):
    energy = np.sum(np.square(image))
    return energy


# 3. H_skewness
def calculate_skewness(image):
    mean = np.mean(image)
    N = image.size
    skewness = (np.sum((image - mean)**3) / N) / (np.sqrt(np.sum((image - mean)**2) / N))**3
    return skewness


# 4. H_root_mean_square
def calculate_root_mean_square(image):
    root_mean_square = np.sqrt(np.sum(np.square(image)) / image.size)
    return root_mean_square


# 5. H_variance
def calculate_variance(image):
    mean = np.mean(image)
    variance = np.sum(np.square(image - mean)) / (image.size - 1)
    return variance


# 6. H_minimum
def calculate_minimum(image):
    minimum = np.min(image)
    return minimum


# 7. H_maximum (3 features)
def calculate_maximum(image):
    maximum = np.max(image)
    return maximum


# 8. H_range (3 features)
def calculate_range(image):
    minimum = np.min(image)
    maximum = np.max(image)
    data_range = maximum - minimum
    return data_range


# 9. GLCM_dissimilarity (2 features)
def calculate_dissimilarity(image):
    glcm = greycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 16, symmetric=True, normed=True)
    glcm = np.mean(glcm, axis=3)
    glcm = np.expand_dims(glcm, axis=3)
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    return dissimilarity

# 10. GLCM_homogeneity
def calculate_homogeneity(image):
    glcm = greycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 16, symmetric=True, normed=True)
    glcm = np.mean(glcm, axis=3)
    glcm = np.expand_dims(glcm, axis=3)
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    return homogeneity


# 11. GLCM_cluster_tendency
def calculate_cluster_tendency(image):
    glcm = greycomatrix(image.astype(np.uint8), [1], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 16, symmetric=True, normed=True)
    glcm = np.mean(glcm, axis=3)
    glcm = np.expand_dims(glcm, axis=3)
    num_levels = glcm.shape[0]
    cluster_tendency = 0

    for i in range(num_levels):
        for j in range(num_levels):
            px = np.sum(glcm[i, :])
            py = np.sum(glcm[:, j])
            cluster_tendency += (i + j - px - py) ** 2 * glcm[i, j]

    return cluster_tendency[0, 0]

train_val_dataloader = get_five_fold_dataset.get_five_fold_dataloaders(csv_path='../../../dataset/lung.csv',
                                                                       node_path='../../../dataset/roi/img_train',
                                                                       fat_path='../../../dataset/fat_intrathoracic',
                                                                       transform_mode='train', mode_select=4)
fold_model_dict = [
    '../../../runner/Model_Dict/BiVGG/BiVGG_node 第1折 0.9201 0.8737.pth',
    '../../../runner/Model_Dict/BiVGG/BiVGG_node 第2折 0.9082 0.8984.pth',
    '../../../runner/Model_Dict/BiVGG/BiVGG_node 第3折 0.9333 0.8229.pth',
    '../../../runner/Model_Dict/BiVGG/BiVGG_node 第4折 0.8989 0.7855.pth',
    '../../../runner/Model_Dict/BiVGG/BiVGG_node 第5折 0.8799 0.8466.pth',
]
selected_features_map_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 20, 21, 23, 24, 26, 27, 28, 29,
                               30, 31, 32, 35, 38, 41, 49, 50, 51, 52, 53, 62, 67, 68, 71, 73, 75, 80, 87, 103, 110,
                               115, 117, 123, 144, 148, 155, 156, 163, 166, 179, 180, 201, 228, 250, 267, 314, 347, 359,
                               365, 393, 410, 429, 431, 435, 442, 443, 484, 510]

features_name = ['H_uniformity', 'H_energy', 'H_skewness', 'H_root_mean_square', 'H_variance', 'H_minimum', 'H_maximum',
                 'H_range', 'GLCM_dissimilarity', 'GLCM_homogeneity', 'GLCM_cluster_tendency']
# 第n折
for fold, (train_loader, val_loader) in enumerate(train_val_dataloader):
    if True:
        test0_dataset = getdataset("../../../dataset/lung.csv", "../../../dataset/roi/img_test0",
                                       "../../../dataset/fat_intrathoracic",
                                       lung_transform['train'], mode_select=4, is_augment=False)
        test1_dataset = getdataset("../../../dataset/lung.csv", "../../../dataset/roi/img_test1",
                                   "../../../dataset/fat_intrathoracic",
                                   lung_transform['train'], mode_select=4, is_augment=False)
        test2_dataset = getdataset("../../../dataset/lung.csv", "../../../dataset/roi/img_test2",
                                   "../../../dataset/fat_intrathoracic",
                                   lung_transform['train'], mode_select=4, is_augment=False)
        test0_name_list = copy.deepcopy(test0_dataset.img_path_list)
        test1_name_list = copy.deepcopy(test1_dataset.img_path_list)
        test2_name_list = copy.deepcopy(test2_dataset.img_path_list)
        test0_dataloader = torch.utils.data.DataLoader(test0_dataset, batch_size=1)
        test1_dataloader = torch.utils.data.DataLoader(test1_dataset, batch_size=1)
        test2_dataloader = torch.utils.data.DataLoader(test2_dataset, batch_size=1)
        train_name_list = []
        val_name_list = []
        train_name_list_all = train_loader.dataset.img_path_list
        val_name_list_all = val_loader.dataset.img_path_list
        # 用来存储图像id号
        for sam in train_loader.sampler:
            train_name_list.append(train_name_list_all[sam])
        for sam2 in val_loader.sampler:
            val_name_list.append(val_name_list_all[sam2])

        all_dataloader = [train_loader, val_loader, test0_dataloader, test1_dataloader, test2_dataloader]
        all_data_name = [train_name_list, val_name_list, test0_name_list, test1_name_list, test2_name_list]
        all_csv_name = ['train_data', 'val_data', 'test0_data', 'test1_data', 'test2_data']
        for i in range(len(all_dataloader)):
            # 标签，深度学习特征，预测值
            name_list = all_data_name[i]
            for name_index in range(len(name_list)):
                name_list[name_index] = name_list[name_index][31:-4]

            DL_final_conv_features_label = get_last_layer_feature_map_from_dataloader(all_dataloader[i],
                                                                                      fold_model_dict[fold])

            DL_final_conv_features = DL_final_conv_features_label[0]
            label_list = DL_final_conv_features_label[1]
            selected_DL_FCF = DL_final_conv_features[:, selected_features_map_index, :, :]
            # 按照特征进行计算，总共有74个特征图，每个特征图计算11个影像组学特征，最后得到11*74个特征
            all_features_patient = []
            for j in range(selected_DL_FCF.shape[0]):
                all_features = []
                for k in range(len(selected_features_map_index)):
                    selected_DL_FCF_one = selected_DL_FCF[j, k, :, :]
                    selected_DL_FCF_one_normalize = normalize_image(selected_DL_FCF_one)
                    selected_DL_FCF_one_normalize_GLCM = GLCM_image(selected_DL_FCF_one_normalize)
                    # 一个病人的第一个特征图，开始计算影像组学特征
                    uniformity = calculate_uniformity(selected_DL_FCF_one_normalize)
                    energy = calculate_energy(selected_DL_FCF_one)
                    skewness = calculate_skewness(selected_DL_FCF_one)
                    root_mean_square = calculate_root_mean_square(selected_DL_FCF_one)
                    variance = calculate_variance(selected_DL_FCF_one)
                    minimum = calculate_minimum(selected_DL_FCF_one)
                    maximum = calculate_maximum(selected_DL_FCF_one)
                    data_range = calculate_range(selected_DL_FCF_one)
                    dissimilarity = calculate_dissimilarity(selected_DL_FCF_one_normalize_GLCM)
                    homogeneity = calculate_homogeneity(selected_DL_FCF_one_normalize_GLCM)
                    cluster_tendency = calculate_cluster_tendency(selected_DL_FCF_one_normalize_GLCM)
                    all_features.append(uniformity)
                    all_features.append(energy)
                    all_features.append(skewness)
                    all_features.append(root_mean_square)
                    all_features.append(variance)
                    all_features.append(minimum)
                    all_features.append(maximum)
                    all_features.append(data_range)
                    all_features.append(dissimilarity)
                    all_features.append(homogeneity)
                    all_features.append(cluster_tendency)
                all_features_patient.append(all_features)
            id = [str(sublist) for sublist in name_list]
            label = [int(sublist) for sublist in label_list]
            final_data = {'Image_ID': id, 'Label': label}
            for l in range(74):
                for m in range(11):
                    final_data[f'Map_{selected_features_map_index[l]}_{features_name[m]}'] = [feat[(l*11)+m] for feat in all_features_patient]
            df = pd.DataFrame(final_data)
            df.to_csv(f'node_fold_data/fold_{fold + 1}/{all_csv_name[i]}_{fold + 1}.csv', index=False)

