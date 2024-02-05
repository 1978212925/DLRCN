import argparse
import os
import random
import sys
import time
from collections import Counter
import torch
import torch.nn as nn
import torchvision.models
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from Runner_utils import train_val_epoch
from Utils.get_dataset import getdataset
from Utils.data_transform import lung_transform
from models.cla_models import resnet50, resnet101, resnet18, resnet34
from models.cla_models.swin_transfomer import swin_small_patch4_window7_224
from models.cla_models.BiVGG import BiVGG
from evaluation_visualization.test import test
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import configs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
def parser_args():
    parser = argparse.ArgumentParser()
    config_path = os.path.join(PROJECT_ROOT, "configs/node_cla.py")
    fine_path = ''
    parser.add_argument('--config_path', type=str, default=config_path)
    parser.add_argument('--finetune_from', type=str, default=fine_path)
    return parser.parse_args()

def model_freeze(model, device, finetune_from, frezze_layer=False):
    if finetune_from != "":
        assert os.path.exists(finetune_from), "weights file: '{}' not exist.".format(finetune_from)
        weights_dict = torch.load(finetune_from, map_location=device)
        # weights_dict = weights_dict['model']
        # 删除有关分类层的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
            if "fc1" in k:
                del weights_dict[k]
            if "class" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
        # 冻结其他权重，只训练分类层
    if frezze_layer:
        for (name, param) in model.named_parameters():
            if (name.find('denseblock4') != -1) or (name.find('class') != -1) or (name.find('denseblock4') != -1):
                param.requires_grad = True
            else:
                param.requires_grad = False


def train(mode_select = 0):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    total_dataset = getdataset("../dataset/lung.csv", "../dataset/roi/img_train", "../dataset/fat_intrathoracic",
                               lung_transform['train'], mode_select=mode_select, is_augment=False)
    labels = [label for (_, label) in total_dataset]
    label_count = dict(Counter(labels))
    print(f'数据集类别比例是:{label_count}')
    # 创建一个包含数据集索引和标签的列表
    data = []
    for i in range(len(total_dataset)):
        data.append((i, total_dataset[i][1]))  # 第三个元素为标签

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
        train_loader = DataLoader(total_dataset, batch_size=cfg.train_batch, sampler=train_indices)
        val_loader = DataLoader(total_dataset, batch_size=cfg.val_batch, sampler=val_indices)

        # 将数据加载器添加到列表中
        data_loaders.append((train_loader, val_loader))
    train_val_epoch.img_pre_visualization(data_loaders[0][0])
    train_val_epoch.img_pre_visualization(data_loaders[0][1])
    # 遍历五个数据加载器，进行训练和验证
    for fold, (train_loader, val_loader) in enumerate(data_loaders):
        best_fold_auc = 0
        best_fold_val_auc = 0
        print(f"五折交叉验证第{fold + 1}折开始")
        if fold+1 > 4:
            # 每一轮重新定义损失函数，优化器，步长衰减，类别权重，模型
            # model = resnet101()
            # model = resnet18()
            # model = torchvision.models.resnet34(pretrained = True)
            # model.fc = nn.Linear(512, 2, bias=True)
            # print(model)
            # model = torchvision.models.resnext50_32x4d()
            # model.fc = nn.Linear(2048, 2)
            # model = torchvision.models.densenet201()
            # model.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)
            # num_features = model.classifier[-1].in_features
            # model.classifier[-1] = nn.Linear(num_features, 2)
            # model = torchvision.models.vgg16()
            # print(model)
            # num_features = model.classifier[-1].in_features
            # model.classifier[-1] = nn.Linear(num_features, 2)
            # model = torchvision.models.densenet121(pretrained=True)
            # model.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
            # model = torchvision.models.densenet161()
            # model.classifier = nn.Linear(in_features=2208, out_features=2, bias=True)
            model = BiVGG()
            model_freeze(model=model, device=device, finetune_from=args.finetune_from, frezze_layer=False)
            pg = [p for p in model.parameters() if p.requires_grad]
            class_weights = cfg.class_weights
            class_weights = torch.FloatTensor(class_weights).to(device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(pg, lr=cfg.lr_start, weight_decay=cfg.weight_decay)
            sche_lr = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.98)
            model.to(device)
            model_name = (type(model).__name__) + "_node"
            print(model_name)

            for epoch in range(cfg.epochs):
                print(f"第{epoch + 1}轮学习开始")
                # 先训练,但是不输出训练集结果,因为训练时候的参数在改变,输出的结果不是最终的准确率
                start = time.time()
                train_acc, train_loss, train_sens, train_spe, train_ppv, train_npv, train_auc = \
                    train_val_epoch.train_cla_epoch(model, loss_fn, optimizer, train_loader, device)
                end = time.time()
                sche_lr.step()
                lr = sche_lr.get_lr()
                print(lr)
                print("train_acc:{:.5f}, train_loss:{:.5f}, train_sens:{:.5f}, train_spe:{:.5f}, train_ppv:{:.5f}, train_npv:{:.5f}, train_auc:{:.5f} "
                      "train_time:{:.0f}m, {:.0f}s".format(
                        train_acc, train_loss, train_sens, train_spe, train_ppv, train_npv, train_auc,  (end - start) // 60, (end - start) % 60))
                start = time.time()
                val_acc, val_loss, val_sens, val_spe, val_ppv, val_npv, val_auc = \
                    train_val_epoch.val_cla_epoch(model, loss_fn, optimizer, val_loader, device)
                end = time.time()
                print( "验证集准确率为:val_acc:{:.5f}, val_loss:{:.5f}, val_sens:{:.5f}, val_spe:{:.5f}, val_ppv:{:.5f}, val_npv:{:.5f}, val_auc:{:.5f} "
                        "train_time:{:.0f}m, {:.0f}s".format(
                        val_acc, val_loss, val_sens, val_spe, val_ppv, val_npv, val_auc, (end - start) // 60, (end - start) % 60))

                start = time.time()
                train_acc, train_loss, train_sens, train_spe, train_ppv, train_npv, train_auc = \
                    train_val_epoch.val_cla_epoch(model, loss_fn, optimizer, train_loader, device)
                end = time.time()
                print(
                    "训练集准确率为:train_acc:{:.5f}, train_loss:{:.5f}, train_sens:{:.5f}, train_spe:{:.5f}, train_ppv:{:.5f}, train_npv:{:.5f}, train_auc:{:.5f} "
                    "train_time:{:.0f}m, {:.0f}s".format(
                        train_acc, train_loss, train_sens, train_spe, train_ppv, train_npv, train_auc,
                        (end - start) // 60, (end - start) % 60))
                if ((best_fold_auc < train_auc) or (best_fold_val_auc < val_auc)) and (val_sens not in (0.0, 1.0)) \
                        and (val_spe not in (0.0, 1.0)) and (train_sens not in (0.0, 1.0)) and (train_spe not in (0.0, 1.0)) \
                        and (train_auc > 0.80) and (val_auc > 0.65):
                    best_fold_auc = train_auc
                    best_fold_val_auc = val_auc
                    print("模型保存中 第{:d}折最好AUC是{:.5f}".format(fold+1, best_fold_auc))
                    torch.save(model.state_dict(),
                               "./Running_Dict/{:s} 第{:d}折 {:.4f} {:.4f}.pth".format(model_name, fold+1, best_fold_auc, best_fold_val_auc))
                print(f"第{epoch + 1}轮学习结束")
            print(f"五折交叉验证第{fold + 1}折结束")
        else:
            pass


if __name__ == '__main__':
    args = parser_args()
    cfg = configs.set_cfg_from_file(args.config_path)
    mode_select = int(input("please select data fuse mode(range is 0-5)："))
    train(mode_select=mode_select)


