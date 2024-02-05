# encoding=utf-8
import random
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset

def get_five_fold_dataloaders(csv_path="../dataset/lung.csv", node_path="../dataset/roi/img_train",
                             fat_path="../dataset/fat_intrathoracic", transform_mode='train', mode_select=4):
    total_dataset = getdataset(csv_path, node_path, fat_path,
                               lung_transform[transform_mode], mode_select=mode_select, is_augment=False)
    labels = [label for (_, label) in total_dataset]
    label_count = dict(Counter(labels))
    print(f'数据集类别比例是:{label_count}')
    # 创建一个包含数据集索引和标签的列表
    data = []
    for i in range(len(total_dataset)):
        data.append((i, total_dataset[i][1]))

    # 将数据集随机排序
    random.seed(120)
    random.shuffle(data)
    # 创建 StratifiedKFold 对象
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 初始化五个数据集分割器的数据加载器
    data_loaders = []
    for train_index, val_index in skf.split(range(len(total_dataset)), [label for _, label in data]):
        # 获取训练集和验证集的索引
        train_indices = [data[i][0] for i in train_index]
        val_indices = [data[i][0] for i in val_index]

        # 创建训练集和验证集的数据加载器
        train_loader = DataLoader(total_dataset, batch_size=1, sampler=train_indices)
        val_loader = DataLoader(total_dataset, batch_size=1, sampler=val_indices)

        # 将数据加载器添加到列表中
        data_loaders.append((train_loader, val_loader))
    return data_loaders
if __name__ == '__main__':
    # 总共有5折，每折里面有train_loader和val_loader
    data_set = get_five_fold_dataloaders()
    fold1_dataloader = data_set[0]
    fold2_dataloader = data_set[1]
    fold3_dataloader = data_set[2]
    fold4_dataloader = data_set[3]
    train_loader, val_loader = fold1_dataloader
    for data in train_loader:
        img, label = data
        print(img, label)
