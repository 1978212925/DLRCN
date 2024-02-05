#encoding=utf-8
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch.nn as nn
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))
dataset_name_list = ['train', 'val', 'test0', 'test1', 'test2']
# # 存储每一折选择的特征系数
# fold_selected_coefs = []
# # 存储每一折选择的特征索引值
# fold_selected_indices = []
# for fold in range(1, 6):
#
#     train_data = pd.read_csv(f'../Spearman/node_fold_data/fold_{fold}/train_data_{fold}.csv')
#     val_data = pd.read_csv(f'../Spearman/node_fold_data/fold_{fold}/val_data_{fold}.csv')
#     test0_data = pd.read_csv(f"../Spearman/node_fold_data/fold_{fold}/test0_data_{fold}.csv")
#     test1_data = pd.read_csv(f"../Spearman/node_fold_data/fold_{fold}/test1_data_{fold}.csv")
#     test2_data = pd.read_csv(f"../Spearman/node_fold_data/fold_{fold}/test2_data_{fold}.csv")
#
#     train_img_ids = np.array(train_data.iloc[:, 0]).astype(int)
#     train_labels = np.array(train_data.iloc[:, 1]).astype(int)
#     train_features = np.array(train_data.iloc[:, 2:]).astype(float)
#
#     val_img_ids = np.array(val_data.iloc[:, 0]).astype(int)
#     val_labels = np.array(val_data.iloc[:, 1]).astype(int)
#     val_features = np.array(val_data.iloc[:, 2:]).astype(float)
#
#     test0_img_ids = np.array(test0_data.iloc[:, 0]).astype(int)
#     test0_labels = np.array(test0_data.iloc[:, 1]).astype(int)
#     test0_features = np.array(test0_data.iloc[:, 2:]).astype(float)
#
#     test1_img_ids = np.array(test1_data.iloc[:, 0]).astype(int)
#     test1_labels = np.array(test1_data.iloc[:, 1]).astype(int)
#     test1_features = np.array(test1_data.iloc[:, 2:]).astype(float)
#
#     test2_img_ids = np.array(test2_data.iloc[:, 0]).astype(int)
#     test2_labels = np.array(test2_data.iloc[:, 1]).astype(int)
#     test2_features = np.array(test2_data.iloc[:, 2:]).astype(float)
#
#     alphas = np.logspace(-10, -1, num=100)  # 可以根据需要选择不同的λ值
#     best_alpha = None
#     best_auc = 0.0
#
#     for alpha in tqdm(alphas):
#         lasso = Lasso(alpha=alpha, max_iter=1000)
#         lasso.fit(train_features, train_labels)
#         val_preds = lasso.predict(val_features)
#         auc = roc_auc_score(val_labels, val_preds)
#
#         if auc > best_auc:
#             best_auc = auc
#             best_alpha = alpha
#             joblib.dump(lasso, './pkl/lasso_node_model.pkl')
#
#     # 读取最好的LASSO模型
#     lasso_model = joblib.load('./pkl/lasso_node_model.pkl')
#     val_preds = lasso_model.predict(val_features)
#     val_auc = roc_auc_score(val_labels, val_preds)
#     # 提取LASSO模型中的系数
#     lasso_coefs = lasso_model.coef_
#     # 计算LASSO有多少个非零系数
#     none_zero = np.count_nonzero(lasso_coefs)
#     lasso_coefs_abs = np.abs(lasso_coefs)
#
#     num_list = list(range(1, 11, 1))
#     best_auc = 0
#     best_num = 1
#     all_top_indices = np.argsort(lasso_coefs_abs)
#     for i in num_list:
#         top_indices = np.argsort(lasso_coefs_abs)[-i:]
#         top_features = val_features[:, top_indices]
#         top_coefs = lasso_coefs[top_indices]
#         lasso_sele_pre = np.dot(top_features, top_coefs.T)
#         selected_auc = roc_auc_score(val_labels, lasso_sele_pre)
#         if best_auc < selected_auc :
#             best_auc = selected_auc
#             best_num = i
#
#     print(best_auc)
#     print(best_num)
#
#     # 计算一下根据选择特征数，所得到的train, val, test1，test2，test0的auc是多少
#     selected_indices = np.argsort(lasso_coefs_abs)[-best_num:]
#     print(selected_indices)
#     selected_coefs = lasso_coefs[selected_indices]
#     fold_selected_coefs.append(selected_coefs)
#     fold_selected_indices.append(selected_indices)
#     train_selected_features = train_features[:, selected_indices]
#     val_selected_features = val_features[:, selected_indices]
#     test0_selected_features = test0_features[:, selected_indices]
#     test1_selected_features = test1_features[:, selected_indices]
#     test2_selected_features = test2_features[:, selected_indices]
#     train_lasso_auc = roc_auc_score(train_labels, np.dot(train_selected_features, selected_coefs.T))
#     val_lasso_auc = roc_auc_score(val_labels, np.dot(val_selected_features, selected_coefs.T))
#     test0_lasso_auc = roc_auc_score(test0_labels, np.dot(test0_selected_features, selected_coefs.T))
#     test1_lasso_auc = roc_auc_score(test1_labels, np.dot(test1_selected_features, selected_coefs.T))
#     test2_lasso_auc = roc_auc_score(test2_labels, np.dot(test2_selected_features, selected_coefs.T))
#     print(
#         f"train val test0，test1，test2auc结果分别为{train_lasso_auc}, {val_lasso_auc}, {test0_lasso_auc}, {test1_lasso_auc}, {test2_lasso_auc}")
# fold_selected_coefs = np.concatenate(fold_selected_coefs)
# fold_selected_indices = np.concatenate(fold_selected_indices)
# # print(fold_selected_coefs)
# # print(fold_selected_indices)
# # take it all 拿到所有选择的特征下标索引值,再单独存储为csv文件
# fold_selected_indices = np.unique(fold_selected_indices)
# fold_selected_indices = np.sort(fold_selected_indices)
# print(fold_selected_indices)
fold_selected_indices = [269, 358, 368, 545, 562, 583, 607, 750,
                           252, 483, 494, 699,
                           90, 198, 539, 566, 571, 583, 698,
                           340, 654,
                           156, 237, 750]
fold_selected_indices = sorted(list(set(fold_selected_indices)))
# fold_selected_indices = [2, 13, 41, 98, 100, 120, 129, 153, 217, 219, 228, 263, 296, 305, 329, 343, 349, 439,
#  472, 483, 494, 503, 516, 527, 552, 563, 593, 596, 626, 657, 668, 761, 802]
all_csv_name = ['selected_train_data', 'selected_val_data', 'selected_test0_data', 'selected_test1_data', 'selected_test2_data']
for fold in range(1, 6):
    for i in range(len(dataset_name_list)):
        data = pd.read_csv(f'../Spearman/node_fold_data/fold_{fold}/{dataset_name_list[i]}_data_{fold}.csv')
        img_ids = data.iloc[:, 0]
        labels = data.iloc[:, 1]
        features = data.iloc[:, 2:]
        selected_features = features.iloc[:, fold_selected_indices]
        final_data = {'Image_ID': img_ids, 'Label': labels}
        df = pd.DataFrame(final_data)
        df = pd.concat([df, selected_features], axis=1)
        df.to_csv(f'./node_fold_data_selected_R_LASSO/fold_{fold}/{all_csv_name[i]}_{fold}.csv', index=False)
