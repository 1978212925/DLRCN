# encoding=utf-8
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from models.cla_models.BiVGG import BiVGG
from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset

# 加载已经训练好的模型，这里以ResNet50为例
model = BiVGG()
model.cuda()
print(model)
# 定义一个函数用于获取特征图
def get_last_layer_feature_map(image_tensor):
    global feature_maps

    # 定义一个hook函数，用于获取特征图
    def hook_fn(module, input, output):
        global feature_maps
        feature_maps = output.data

    # 获取模型的最后一层卷积层
    last_conv_layer = model.conv_final  # 最后一层卷积层是VGG16模型的features中的第二层

    # 注册hook函数到最后一层卷积层
    hook_handle = last_conv_layer.register_forward_hook(hook_fn)

    # 将图片输入模型，获取特征图
    model(image_tensor.cuda())

    # 取消hook
    hook_handle.remove()

    return feature_maps

def get_last_layer_feature_map_from_dataloader(dataloader, model_dict_path):
    model.load_state_dict(torch.load(model_dict_path))
    # 设置模型为评估模式，这样可以避免Dropout等层的影响
    model.eval()
    # dataloader的batchsize设置为1
    all_patience_last_layer_feature_map = []
    all_patience_label = []
    for data in tqdm(dataloader):
        imgs, labels = data
        # show_img = torch.squeeze(imgs, dim=0)
        # show_img = torch.permute(show_img, (1, 2, 0))
        # plt.imshow(show_img)
        # plt.show()
        imgs = imgs.cuda()
        last_layer_feature_map = get_last_layer_feature_map(imgs)

        # result = last_layer_feature_map[0, 1, :, :]
        # result = result.cpu()
        # result = result.detach().numpy()
        # plt.imshow(result)
        # plt.show()
        all_patience_last_layer_feature_map.append(last_layer_feature_map.cpu())
        all_patience_label.append(labels)
    all_patience_last_layer_feature_map = np.concatenate(all_patience_last_layer_feature_map)
    all_patience_label = np.concatenate(all_patience_label)
    return all_patience_last_layer_feature_map, all_patience_label


if __name__ == '__main__':
    fold_model_dict = [
        '../runner/Model_Dict/BiVGG/BiVGG_node 第1折 0.9201 0.8737.pth',
        '../runner/Model_Dict/BiVGG/BiVGG_node 第2折 0.9082 0.8984.pth',
        '../runner/Model_Dict/BiVGG/BiVGG_node 第3折 0.9333 0.8229.pth',
        '../runner/Model_Dict/BiVGG/BiVGG_node 第4折 0.8989 0.7855.pth',
        '../runner/Model_Dict/BiVGG/BiVGG_node 第5折 0.8799 0.8466.pth',
    ]
    total_dataset = getdataset("../dataset/lung.csv", "../dataset/roi/img_test0", "../dataset/fat_intrathoracic",
                               lung_transform['train'], mode_select=4, is_augment=False)
    total_dataloader = torch.utils.data.DataLoader(total_dataset, batch_size=1)
    all_patience_last_layer_feature_map, all_patience_label = get_last_layer_feature_map_from_dataloader(total_dataloader,
                                                                                     fold_model_dict[0])
    all_patience_last_layer_feature_map = all_patience_last_layer_feature_map.reshape(all_patience_last_layer_feature_map.shape[0], -1)

    logistic_regressor = LogisticRegression(C=1, penalty='l2', max_iter=1000)
    logistic_regressor.fit(all_patience_last_layer_feature_map, all_patience_label)
    y_pred_train = logistic_regressor.predict_proba(all_patience_last_layer_feature_map)[:, 1]
    print(roc_auc_score(all_patience_label, y_pred_train))
