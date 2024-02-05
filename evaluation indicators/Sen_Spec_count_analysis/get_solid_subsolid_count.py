# encoding=utf-8
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


# 0.7823308478453083
# 分析医师和模型预测的良恶性正确个数
def get_argmax(threshold, y_pred):
    y_pred = copy.deepcopy(y_pred)
    for j in range(len(y_pred)):
        if y_pred[j] > threshold:
            y_pred[j] = 1
        else:
            y_pred[j] = 0
    y_pred_argmax = y_pred
    return y_pred_argmax
def get_count(label, y_pred):
    benign_count = 0
    malignant_count = 0
    for i in range(len(label)):
        if label[i] == 0:
            benign_count += 1
        else:
            malignant_count += 1
    benign_wrong_conunt = 0
    malignant_wrong_count = 0
    for j in range(len(label)):
        if label[j] == 0 and y_pred[j] == 1:
            benign_wrong_conunt += 1
        elif label[j] == 1 and y_pred[j] == 0:
            malignant_wrong_count += 1
        else:
            pass
    return benign_wrong_conunt, benign_count, malignant_wrong_count, malignant_count
def cal_mean_CI(list):
    # 计算均值
    mean = np.mean(list)
    # 5折交叉验证所以自由度是5
    n = 5
    # 计算标准差
    std = np.std(list)
    # 双边alpha0.05 = 2.571
    a = 2.571

    return mean, [mean - (a * std/np.sqrt(n)), mean + (a * std/np.sqrt(n))]
fold_best_thresshold = [0.7823308478453083,
                        0.7602601808665441,
                        0.8011549933860141,
                        0.7797335702530113,
                        0.868434179822722]
def get_solid_or_subsolid_result(Sub_solid):
    all_csv_name = ['test0', 'test1', 'test2']
    for i in range(len(all_csv_name)):
        print(all_csv_name[i])
        fold_benign_wrong_conunt = []
        fold_benign_count = []
        fold_malignant_wrong_count = []
        fold_malignant_count = []
        for fold in range(1, 6):
            data = pd.read_csv(
                f'../../Utils/nomogram/dif_features_combin_result/fold_{fold}/{all_csv_name[i]}_data_{fold}.csv', )
            screen_data = data[data['Sub_solid(yes=1, no=0)'] == Sub_solid]
            label = np.array(screen_data.iloc[:, 1]).astype(int)
            DLRN = np.array(screen_data.iloc[:, 7]).astype(float)
            fpr, tpr, thresholds = roc_curve(label, DLRN)
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            DLRN_argmax = get_argmax(fold_best_thresshold[fold-1], DLRN).astype(int)
            benign_wrong_conunt, benign_count, malignant_wrong_count, malignant_count = get_count(label, DLRN_argmax)
            fold_benign_wrong_conunt.append(benign_wrong_conunt)
            fold_malignant_wrong_count.append(malignant_wrong_count)
            fold_benign_count.append(benign_count)
            fold_malignant_count.append(malignant_count)

        benign_wrong_count_mean, benign_wrong_count_CI = cal_mean_CI(fold_benign_wrong_conunt)
        benign_count_mean, benign_count_CI = cal_mean_CI(fold_benign_count)
        malignant_wrong_count_mean, malignant_wrong_count_CI = cal_mean_CI(fold_malignant_wrong_count)
        malignant_count_mean, malignant_count_CI = cal_mean_CI(fold_malignant_count)

        print(f'DLRN模型预测良性错误{benign_wrong_count_mean}个,95置信区间是{benign_wrong_count_CI}, 良性个数是{benign_count_mean}')
        print(f'DLRN模型预测恶性错误{malignant_wrong_count_mean}个,95置信区间是{malignant_wrong_count_CI}, 恶性个数是{malignant_count_mean}')

        physician1 = pd.read_csv(f'../../Utils/nomogram/physician/physician_1_result/{all_csv_name[i]}.csv')
        physician2 = pd.read_csv(f'../../Utils/nomogram/physician/physician_2_result/{all_csv_name[i]}.csv')
        physician3 = pd.read_csv(f'../../Utils/nomogram/physician/physician_3_result/{all_csv_name[i]}.csv')
        physician1 = physician1[physician1['Sub_solid(yes=1, no=0)'] == Sub_solid]
        physician2 = physician2[physician2['Sub_solid(yes=1, no=0)'] == Sub_solid]
        physician3 = physician3[physician3['Sub_solid(yes=1, no=0)'] == Sub_solid]

        physician1_label = np.array(physician1.iloc[:, 1]).astype(int)
        physician1_pred = np.array(physician1.iloc[:, 2]).astype(int)
        physician2_label = np.array(physician2.iloc[:, 1]).astype(int)
        physician2_pred = np.array(physician2.iloc[:, 2]).astype(int)
        physician3_label = np.array(physician3.iloc[:, 1]).astype(int)
        physician3_pred = np.array(physician3.iloc[:, 2]).astype(int)

        physician1_result = get_count(physician1_label, physician1_pred)
        physician2_result = get_count(physician2_label, physician2_pred)
        physician3_result = get_count(physician3_label, physician3_pred)
        print(f'医师1预测良性错误{physician1_result[0]}个, 良性个数是{physician1_result[1]}')
        print(f'医师1预测恶性错误{physician1_result[2]}个, 恶性个数是{physician1_result[3]}')
        print(f'医师2预测良性错误{physician2_result[0]}个, 良性个数是{physician2_result[1]}')
        print(f'医师2预测恶性错误{physician2_result[2]}个, 恶性个数是{physician2_result[3]}')
        print(f'医师3预测良性错误{physician3_result[0]}个, 良性个数是{physician3_result[1]}')
        print(f'医师3预测恶性错误{physician3_result[2]}个, 恶性个数是{physician3_result[3]}')
if __name__ == '__main__':
    print('单独讨论实性结节结果')
    get_solid_or_subsolid_result(0)
    print('单独讨论非实性结节结果')
    get_solid_or_subsolid_result(1)