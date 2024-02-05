#encoding=utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))
selected_coefs_list = [-0.20232596,  0.20370864, -0.21636134, -0.03672266, -0.16462728, -0.20389416,
  0.38885453, -0.42797165,  0.47378235, -0.48857085, -0.13180395,  0.17072977,
 -0.26728774, -0.27983843, -0.29469404,  0.29699422,  0.34908429, -0.36753503,
  0.39142821,  0.39148083,  0.40157872, -0.84740017, -0.93005854, -1.30167062,
  0.20265865,  0.20584791, -0.25574244,  0.28370561,  0.32585384,  0.36052646,
 -0.41522595, -0.52733674, -0.7629195,   0.2498445,  -0.26321677,  0.27004809,
  0.31978011, -0.34761315,  0.34915949, -0.39708377, -0.39773461, -0.43337463,
 -0.48124059, -0.30223669,  4.37036321]
selected_coefs_list = np.array(selected_coefs_list)
selected_csv_name_list = ['selected_train_data', 'selected_val_data', 'selected_test0_data', 'selected_test1_data', 'selected_test2_data']
for fold in range(1, 11):
    for dataset_name in selected_csv_name_list:
        selected_data = pd.read_csv(f'../feature_selection/selected_node_fold_data/fold_{fold}/{dataset_name}_{fold}.csv')
        selected_label = np.array(selected_data.iloc[:, 1]).astype(int)
        selected_features = np.array(selected_data.iloc[:, 2:]).astype(np.float)

        selected_pred = sigmoid(np.dot(selected_features, selected_coefs_list.T))
        selected_auc = roc_auc_score(selected_label, selected_pred)

        print(f'{fold}折的{dataset_name}的结果是 auc:{selected_auc}')