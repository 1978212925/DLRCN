# encoding=utf-8
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from sklearn.model_selection import KFold
import numpy as np
def cal_mean_CI(list):
    # 计算均值
    mean = np.mean(list)
    # 5折交叉验证所以自由度是5
    n = 5
    # 计算标准差
    std = np.std(list)
    # 双边alpha0.05 = 2.571
    a = 2.571

    return mean, [mean - (a * std / np.sqrt(n)), mean + (a * std / np.sqrt(n))]
# 假设有5个CSV文件
csv_files = ['./fold', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv']

fold_feature_correlations = []
fold_feature_p = []
# 遍历每个CSV文件
for fold in range(1, 6):
    # 读取CSV文件
    df = pd.read_csv(f'./fold_{fold}/selected_train_data_{fold}.csv', encoding='GBK')

    # 获取特征列和标签列
    feature_columns = np.array(df.iloc[:, 2:]).astype(float)
    label_column = np.array(df.iloc[:, 1]).astype(int)

    # 初始化存储特征相关系数的列表
    feature_correlations = []
    feature_p = []
    for i in range(0,22):
        # 计算每个特征和标签的Spearman相关系数
        correlation, pvalue = spearmanr(feature_columns[:, i], label_column)
        feature_correlations.append(correlation)
        feature_p.append(pvalue)
    fold_feature_correlations.append(feature_correlations)
    fold_feature_p.append(feature_p)


for i in range(0, 22):
    corr_list = []
    p_list = []
    for j in range(0,5):
        corr = fold_feature_correlations[j][i]
        corr_list.append(corr)
        p = fold_feature_p[j][i]
        p_list.append(p)

    corr_result = cal_mean_CI(corr_list)
    print(corr_result)
    p_result = cal_mean_CI(p_list)
    print(p_result)
    print('---------------------')
