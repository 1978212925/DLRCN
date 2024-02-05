# encoding=utf-8
import numpy as np
from matplotlib import pyplot as plt
plt.rc('font', family='Times New Roman')
from scipy.stats import spearmanr
from Utils import get_five_fold_dataset
from Utils.get_DL_final_conv_featuremap import get_last_layer_feature_map_from_dataloader
import seaborn as sns
# 模拟数据：训练集有302个病人，每个病人有512个特征图，每个特征图大小为7x7
num_patients = 302
num_features = 512
feature_size = (7, 7)
fold_model_dict = [
        '../../../runner/Model_Dict/BiVGG/BiVGG_node 第1折 0.9201 0.8737.pth',
        '../../../runner/Model_Dict/BiVGG/BiVGG_node 第2折 0.9082 0.8984.pth',
        '../../../runner/Model_Dict/BiVGG/BiVGG_node 第3折 0.9333 0.8229.pth',
        '../../../runner/Model_Dict/BiVGG/BiVGG_node 第4折 0.8989 0.7855.pth',
        '../../../runner/Model_Dict/BiVGG/BiVGG_node 第5折 0.8799 0.8466.pth',
                       ]
# 由于五折交叉验证，每一折都需要循环一次，特征筛选是在训练集上完成
# 获取五折交叉验证的所有数据集
all_fold_dataloader = get_five_fold_dataset.get_five_fold_dataloaders(csv_path='../../../dataset/lung.csv',
                                                                      node_path='../../../dataset/roi/img_train',
                                                                      fat_path='../../../dataset/fat_intrathoracic',
                                                                      transform_mode='train', mode_select=4)
all_fold_train_dataloader = [x[0] for x in all_fold_dataloader]
for i in range(len(all_fold_dataloader)):
    DL_final_conv_features_label = get_last_layer_feature_map_from_dataloader(all_fold_train_dataloader[i], fold_model_dict[i])
    # 存储所有热图，最后做平均
    all_correlation_matrices = []
    DL_final_conv_features = DL_final_conv_features_label[0]
    for j in range(DL_final_conv_features.shape[0]):
        DL_final_conv_features_one = DL_final_conv_features[j]

        # 获取特征图扁平化后的矩阵
        flattened_features = DL_final_conv_features_one.reshape(num_features, -1)

        # 计算特征图之间的Spearman相关系数矩阵
        correlation_matrix, _ = spearmanr(flattened_features, axis=1)

        all_correlation_matrices.append(correlation_matrix)
    # 将所有样本得到的相关系数矩阵相加
    summed_correlation_matrix = np.sum(all_correlation_matrices, axis=0)

    # 对总和矩阵进行平均
    average_correlation_matrix = summed_correlation_matrix / num_patients
    # 绘制Spearman相关系数热图
    plt.figure(figsize=(7, 7), dpi=600)
    plt.imshow(average_correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Average Spearman Correlation Matrix in Fold {i+1}')
    plt.xlabel('Feature Map Index')
    plt.ylabel('Feature Map Index')
    plt.savefig(f'./img/fold_{i}.png')

    # 根据相关系数矩阵进行特征图选择
    threshold = 0.6  # 设定相关性阈值
    correlation_mask = (np.abs(average_correlation_matrix) > threshold)  # 生成相关性掩码

    # 通过遍历相关性掩码，删除相关性大于阈值的特征图
    selected_features = list(range(num_features))  # 初始化选定的特征图索引列表
    for k in range(num_features):
        for l in range(k + 1, num_features):
            if correlation_mask[k, l] and (l in selected_features):
                selected_features.remove(l)

    # 输出选择的特征图索引
    print("Selected Features after correlation-based selection:")
    print(selected_features)
