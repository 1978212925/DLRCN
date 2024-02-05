# encoding=utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from scipy.stats import spearmanr# 加载数据集
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# data = pd.read_csv('../../Utils/save_features/radiomic_fat_data/train_data.csv')
# test1_data = pd.read_csv('../../Utils/save_features/radiomic_fat_data/test2_data.csv')
# radiomic_data = data.iloc[:, 2:]
# radiomic_label = data.iloc[:, 1]
# columns = radiomic_data.columns.values
# radiomic_data = np.array(radiomic_data)
# radiomic_label = np.array(radiomic_label)
# radiomic_test1_data = np.array(test1_data.iloc[:, 2:]).astype(float)
# radiomic_test1_label = np.array(test1_data.iloc[:, 1]).astype(int)
#
# X, y = radiomic_data, radiomic_label# 计算每个特征与目标变量之间的互信息
# # scores = mutual_info_classif(X, y)# 根据得分进行特征选择，选择得分排名前5的特征
# # k_best = SelectKBest(mutual_info_classif, k=5).fit(X, y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# 训练逻辑回归模型
# model = LogisticRegression()
# model.fit(X_train, y_train)
# scores = np.abs(model.coef_[0])# 按重要性得分排序
# sorted_idx = np.argsort(scores)[::-1]# 选择前5个特征进行后向消元
# num_features = 5
# for i in range(X_train.shape[1], num_features, -1):# 选择前i个特征
#        idx = sorted_idx[:i]
#        X_train_subset = X_train[:, idx]
#        X_test_subset = X_test[:, idx]# 训练模型并计算准确率
#        model.fit(X_train_subset, y_train)
#        y_pred = model.predict(X_test_subset)
#        acc = accuracy_score(y_test, y_pred)
#        print(f"Selected features: {columns[idx]}")
#        print(f"Accuracy: {acc:.4f}")# 找到最低得分的特征并删除
#        if i > num_features:
#               worst_idx = np.argmin(scores[idx])
#               sorted_idx = np.delete(sorted_idx, np.where(sorted_idx == idx[worst_idx]))# 最终选择的特征子集
# final_idx = sorted_idx[:num_features]
# final_features = columns[final_idx]
# print(f"Final selected features: {final_features}")

name_list = ['ID', 'Label',
             "original_glrlm_ShortRunHighGrayLevelEmphasis",
 "original_glszm_GrayLevelNonUniformityNormalized",
 "original_glszm_SizeZoneNonUniformity",
 "original_glszm_SmallAreaHighGrayLevelEmphasis"]
train_data = pd.read_csv('../../Utils/save_features/radiomic_fat_data/train_data.csv')
test0_data = pd.read_csv('../../Utils/save_features/radiomic_fat_data/test0_data.csv')
test1_data = pd.read_csv('../../Utils/save_features/radiomic_fat_data/test1_data.csv')
test2_data = pd.read_csv('../../Utils/save_features/radiomic_fat_data/test2_data.csv')

train_data_new = train_data[name_list]
test0_data_new = test0_data[name_list]
test1_data_new = test1_data[name_list]
test2_data_new = test2_data[name_list]

train_data_new.to_csv('../../Utils/save_features/radiomic_fat_data_select/train_data_selected.csv', encoding='GBK', index=False)
test0_data_new.to_csv('../../Utils/save_features/radiomic_fat_data_select/test0_data_selected.csv', encoding='GBK', index=False)
test1_data_new.to_csv('../../Utils/save_features/radiomic_fat_data_select/test1_data_selected.csv', encoding='GBK', index=False)
test2_data_new.to_csv('../../Utils/save_features/radiomic_fat_data_select/test2_data_selected.csv', encoding='GBK', index=False)
