import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset
from runner.Runner_utils.evaluation_index import ClassificationMetric
from Utils.machine_learning.evaluate_cal_mean_ci import cal_mean_CI, cal_acc_sens_spec_ppv_npv
# 先提取fold1的train和val,用来判断使用什么参数
fold1_train_data = pd.read_csv('../../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_1/selected_train_data_1.csv')
fold1_train_labels = np.array(fold1_train_data.iloc[:, 1]).astype(int)
fold1_train_features = np.array(fold1_train_data.iloc[:, 2:]).astype(float)

fold1_val_data = pd.read_csv('../../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_1/selected_val_data_1.csv')
fold1_val_labels = np.array(fold1_val_data.iloc[:, 1]).astype(int)
fold1_val_features = np.array(fold1_val_data.iloc[:, 2:]).astype(float)
best_auc = 0
C = [0.001, 0.01, 0.1, 1, 10, 100]
penalty = ['l1', 'l2']
for c in C:
    for pe in penalty:
        if pe == "l1":
            solver = 'liblinear'
        else:
            solver = 'lbfgs'
        Logistic_regressor = LogisticRegression(C=c, penalty=pe, max_iter=1000, solver=solver)
        Logistic_regressor.fit(fold1_train_features, fold1_train_labels)
        y_pred_valid = Logistic_regressor.predict_proba(fold1_val_features)[:, 1]
        y_pred_train = Logistic_regressor.predict_proba(fold1_train_features)[:, 1]
        auc_valid = roc_auc_score(fold1_val_labels, y_pred_valid)
        auc_train = roc_auc_score(fold1_train_labels, y_pred_train)
        print(f"C={c}, penalty={pe}, AUC={auc_valid:.3f}, train_auc = {auc_train:.3f}")

        if auc_valid > best_auc:
            best_auc = auc_valid
            best_params = {'C': c, 'penalty': pe}
            joblib.dump(Logistic_regressor, f'./pkl/LR_node_model.pkl')
dataset_name_list = ['selected_train', 'selected_val', 'selected_test0', 'selected_test1', 'selected_test2']
print(best_params)

# # 确定了最佳的参数之后,计算每一折的验证集的结果,并求平均值
# all_fold_auc_train = []
# all_fold_acc_train = []
# all_fold_sens_train = []
# all_fold_spec_train = []
# all_fold_ppv_train = []
# all_fold_npv_train = []
#
# all_fold_auc_val = []
# all_fold_acc_val = []
# all_fold_sens_val = []
# all_fold_spec_val = []
# all_fold_ppv_val = []
# all_fold_npv_val = []
#
# all_fold_auc_test0 = []
# all_fold_acc_test0 = []
# all_fold_sens_test0 = []
# all_fold_spec_test0 = []
# all_fold_ppv_test0 = []
# all_fold_npv_test0 = []
#
# all_fold_auc_test1 = []
# all_fold_acc_test1 = []
# all_fold_sens_test1 = []
# all_fold_spec_test1= []
# all_fold_ppv_test1 = []
# all_fold_npv_test1 = []
#
# all_fold_auc_test2 = []
# all_fold_acc_test2 = []
# all_fold_sens_test2 = []
# all_fold_spec_test2 = []
# all_fold_ppv_test2 = []
# all_fold_npv_test2 = []
#
# for fold in range(1, 6):
#
#     train_data = pd.read_csv(f'../../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_train_data_{fold}.csv')
#     val_data = pd.read_csv(f'../../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_val_data_{fold}.csv')
#     test0_data = pd.read_csv(f"../../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_test0_data_{fold}.csv")
#     test1_data = pd.read_csv(f"../../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_test1_data_{fold}.csv")
#     test2_data = pd.read_csv(f"../../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_test2_data_{fold}.csv")
#
#     train_img_ids = np.array(train_data.iloc[:, 0]).astype(int)
#     train_labels = np.array(train_data.iloc[:, 1]).astype(int)
#     train_features = np.array(train_data.iloc[:, 2:]).astype(float)
#     logistic_regressor = LogisticRegression(C=100, penalty='l2', max_iter=1000)
#     logistic_regressor.fit(train_features, train_labels)
#     y_pred_train = logistic_regressor.predict_proba(train_features)[:, 1]
#
#     val_img_ids = np.array(val_data.iloc[:, 0]).astype(int)
#     val_labels = np.array(val_data.iloc[:, 1]).astype(int)
#     val_features = np.array(val_data.iloc[:, 2:]).astype(float)
#     y_pred_valid = logistic_regressor.predict_proba(val_features)[:, 1]
#
#     test0_img_ids = np.array(test0_data.iloc[:, 0]).astype(int)
#     test0_labels = np.array(test0_data.iloc[:, 1]).astype(int)
#     test0_features = np.array(test0_data.iloc[:, 2:]).astype(float)
#     y_pred_test0 = logistic_regressor.predict_proba(test0_features)[:, 1]
#
#     test1_img_ids = np.array(test1_data.iloc[:, 0]).astype(int)
#     test1_labels = np.array(test1_data.iloc[:, 1]).astype(int)
#     test1_features = np.array(test1_data.iloc[:, 2:]).astype(float)
#     y_pred_test1 = logistic_regressor.predict_proba(test1_features)[:, 1]
#
#     test2_img_ids = np.array(test2_data.iloc[:, 0]).astype(int)
#     test2_labels = np.array(test2_data.iloc[:, 1]).astype(int)
#     test2_features = np.array(test2_data.iloc[:, 2:]).astype(float)
#     y_pred_test2 = logistic_regressor.predict_proba(test2_features)[:, 1]
#     # train
#     train_fpr, train_tpr, train_thresholds = roc_curve(train_labels, y_pred_train)
#     train_best_threshold = train_thresholds[np.argmax(train_tpr - train_fpr)]
#     train_fold_roc_auc = auc(train_fpr, train_tpr)
#     train_fold_acc, train_fold_sens, train_fold_spec, train_fold_ppv, train_fold_npv = cal_acc_sens_spec_ppv_npv(0.64273, train_labels,
#                                                                                    y_pred_train)
#     all_fold_auc_train.append(train_fold_roc_auc)
#     all_fold_acc_train.append(train_fold_acc)
#     all_fold_sens_train.append(train_fold_sens)
#     all_fold_spec_train.append(train_fold_spec)
#     all_fold_ppv_train.append(train_fold_ppv)
#     all_fold_npv_train.append(train_fold_npv)
#     # val
#     val_fpr, val_tpr, val_thresholds = roc_curve(val_labels, y_pred_valid)
#     val_best_threshold = val_thresholds[np.argmax(val_tpr - val_fpr)]
#     val_fold_roc_auc = auc(val_fpr, val_tpr)
#     val_fold_acc, val_fold_sens, val_fold_spec, val_fold_ppv, val_fold_npv = cal_acc_sens_spec_ppv_npv(0.64273, val_labels,
#                                                                                    y_pred_valid)
#     all_fold_auc_val.append(val_fold_roc_auc)
#     all_fold_acc_val.append(val_fold_acc)
#     all_fold_sens_val.append(val_fold_sens)
#     all_fold_spec_val.append(val_fold_spec)
#     all_fold_ppv_val.append(val_fold_ppv)
#     all_fold_npv_val.append(val_fold_npv)
#     # test0
#     test0_fpr, test0_tpr, test0_thresholds = roc_curve(test0_labels, y_pred_test0)
#     test0_best_threshold = test0_thresholds[np.argmax(test0_tpr - test0_fpr)]
#     test0_fold_roc_auc = auc(test0_fpr, test0_tpr)
#     test0_fold_acc, test0_fold_sens, test0_fold_spec, test0_fold_ppv, test0_fold_npv = cal_acc_sens_spec_ppv_npv(0.64273, test0_labels,
#                                                                                    y_pred_test0)
#     all_fold_auc_test0.append(test0_fold_roc_auc)
#     all_fold_acc_test0.append(test0_fold_acc)
#     all_fold_sens_test0.append(test0_fold_sens)
#     all_fold_spec_test0.append(test0_fold_spec)
#     all_fold_ppv_test0.append(test0_fold_ppv)
#     all_fold_npv_test0.append(test0_fold_npv)
#     # test1
#     test1_fpr, test1_tpr, test1_thresholds = roc_curve(test1_labels, y_pred_test1)
#     test1_best_threshold = test1_thresholds[np.argmax(test1_tpr - test1_fpr)]
#     test1_fold_roc_auc = auc(test1_fpr, test1_tpr)
#     test1_fold_acc, test1_fold_sens, test1_fold_spec, test1_fold_ppv, test1_fold_npv = cal_acc_sens_spec_ppv_npv(0.64273, test1_labels,
#                                                                                    y_pred_test1)
#     all_fold_auc_test1.append(test1_fold_roc_auc)
#     all_fold_acc_test1.append(test1_fold_acc)
#     all_fold_sens_test1.append(test1_fold_sens)
#     all_fold_spec_test1.append(test1_fold_spec)
#     all_fold_ppv_test1.append(test1_fold_ppv)
#     all_fold_npv_test1.append(test1_fold_npv)
#     # test2
#     test2_fpr, test2_tpr, test2_thresholds = roc_curve(test2_labels, y_pred_test2)
#     test2_best_threshold = test2_thresholds[np.argmax(test2_tpr - test2_fpr)]
#     test2_fold_roc_auc = auc(test2_fpr, test2_tpr)
#     test2_fold_acc, test2_fold_sens, test2_fold_spec, test2_fold_ppv, test2_fold_npv = cal_acc_sens_spec_ppv_npv(0.74273, test2_labels,
#                                                                                    y_pred_test2)
#     all_fold_auc_test2.append(test2_fold_roc_auc)
#     all_fold_acc_test2.append(test2_fold_acc)
#     all_fold_sens_test2.append(test2_fold_sens)
#     all_fold_spec_test2.append(test2_fold_spec)
#     all_fold_ppv_test2.append(test2_fold_ppv)
#     all_fold_npv_test2.append(test2_fold_npv)
#
# train_auc_mean, train_auc_ci = cal_mean_CI(all_fold_auc_train)
# train_acc_mean, train_acc_ci = cal_mean_CI(all_fold_acc_train)
# train_sens_mean, train_sens_ci = cal_mean_CI(all_fold_sens_train)
# train_spec_mean, train_spec_ci = cal_mean_CI(all_fold_spec_train)
# train_ppv_mean, train_ppv_ci = cal_mean_CI(all_fold_ppv_train)
# train_npv_mean, train_npv_ci = cal_mean_CI(all_fold_npv_train)
# print(f"train的结果是")
# print("auc:", train_auc_mean, train_auc_ci)
# print("acc:", train_acc_mean, train_acc_ci)
# print("sens:", train_sens_mean, train_sens_ci)
# print("spec:", train_spec_mean, train_spec_ci)
# print("ppv:", train_ppv_mean, train_ppv_ci)
# print("npv:", train_npv_mean, train_npv_ci)
#
# val_auc_mean, val_auc_ci = cal_mean_CI(all_fold_auc_val)
# val_acc_mean, val_acc_ci = cal_mean_CI(all_fold_acc_val)
# val_sens_mean, val_sens_ci = cal_mean_CI(all_fold_sens_val)
# val_spec_mean, val_spec_ci = cal_mean_CI(all_fold_spec_val)
# val_ppv_mean, val_ppv_ci = cal_mean_CI(all_fold_ppv_val)
# val_npv_mean, val_npv_ci = cal_mean_CI(all_fold_npv_val)
# print(f"val的结果是")
# print("auc:", val_auc_mean, val_auc_ci)
# print("acc:", val_acc_mean, val_acc_ci)
# print("sens:", val_sens_mean, val_sens_ci)
# print("spec:", val_spec_mean, val_spec_ci)
# print("ppv:", val_ppv_mean, val_ppv_ci)
# print("npv:", val_npv_mean, val_npv_ci)
#
# test0_auc_mean, test0_auc_ci = cal_mean_CI(all_fold_auc_test0)
# test0_acc_mean, test0_acc_ci = cal_mean_CI(all_fold_acc_test0)
# test0_sens_mean, test0_sens_ci = cal_mean_CI(all_fold_sens_test0)
# test0_spec_mean, test0_spec_ci = cal_mean_CI(all_fold_spec_test0)
# test0_ppv_mean, test0_ppv_ci = cal_mean_CI(all_fold_ppv_test0)
# test0_npv_mean, test0_npv_ci = cal_mean_CI(all_fold_npv_test0)
# print(f"test0的结果是")
# print("auc:", test0_auc_mean, test0_auc_ci)
# print("acc:", test0_acc_mean, test0_acc_ci)
# print("sens:", test0_sens_mean, test0_sens_ci)
# print("spec:", test0_spec_mean, test0_spec_ci)
# print("ppv:", test0_ppv_mean, test0_ppv_ci)
# print("npv:", test0_npv_mean, test0_npv_ci)
#
# test1_auc_mean, test1_auc_ci = cal_mean_CI(all_fold_auc_test1)
# test1_acc_mean, test1_acc_ci = cal_mean_CI(all_fold_acc_test1)
# test1_sens_mean, test1_sens_ci = cal_mean_CI(all_fold_sens_test1)
# test1_spec_mean, test1_spec_ci = cal_mean_CI(all_fold_spec_test1)
# test1_ppv_mean, test1_ppv_ci = cal_mean_CI(all_fold_ppv_test1)
# test1_npv_mean, test1_npv_ci = cal_mean_CI(all_fold_npv_test1)
# print(f"test1的结果是")
# print("auc:", test1_auc_mean, test1_auc_ci)
# print("acc:", test1_acc_mean, test1_acc_ci)
# print("sens:", test1_sens_mean, test1_sens_ci)
# print("spec:", test1_spec_mean, test1_spec_ci)
# print("ppv:", test1_ppv_mean, test1_ppv_ci)
# print("npv:", test1_npv_mean, test1_npv_ci)
#
# test2_auc_mean, test2_auc_ci = cal_mean_CI(all_fold_auc_test2)
# test2_acc_mean, test2_acc_ci = cal_mean_CI(all_fold_acc_test2)
# test2_sens_mean, test2_sens_ci = cal_mean_CI(all_fold_sens_test2)
# test2_spec_mean, test2_spec_ci = cal_mean_CI(all_fold_spec_test2)
# test2_ppv_mean, test2_ppv_ci = cal_mean_CI(all_fold_ppv_test2)
# test2_npv_mean, test2_npv_ci = cal_mean_CI(all_fold_npv_test2)
# print(f"test2的结果是")
# print("auc:", test2_auc_mean, test2_auc_ci)
# print("acc:", test2_acc_mean, test2_acc_ci)
# print("sens:", test2_sens_mean, test2_sens_ci)
# print("spec:", test2_spec_mean, test2_spec_ci)
# print("ppv:", test2_ppv_mean, test2_ppv_ci)
# print("npv:", test2_npv_mean, test2_npv_ci)
# {'C': 100, 'penalty': 'l2'}

