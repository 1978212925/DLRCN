import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from scipy.stats import chi2
plt.rc('font', family='Times New Roman')
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
import os
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, auc


def hosmer_lemeshow_test(Y, y_pred):
    Y = np.array(Y)
    pihat = np.array(y_pred)
    pihatcat = pd.cut(pihat, np.percentile(pihat, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), labels=False,
                      include_lowest=True)  # here we've chosen only 4 groups

    meanprobs = [0] * 4
    expevents = [0] * 4
    obsevents = [0] * 4
    meanprobs2 = [0] * 4
    expevents2 = [0] * 4
    obsevents2 = [0] * 4

    for i in range(4):
        meanprobs[i] = np.mean(pihat[pihatcat == i])
        expevents[i] = np.sum(pihatcat == i) * np.array(meanprobs[i])
        obsevents[i] = np.sum(Y[pihatcat == i])
        meanprobs2[i] = np.mean(1 - pihat[pihatcat == i])
        expevents2[i] = np.sum(pihatcat == i) * np.array(meanprobs2[i])
        obsevents2[i] = np.sum(1 - Y[pihatcat == i])

    data1 = {'meanprobs': meanprobs, 'meanprobs2': meanprobs2}
    data2 = {'expevents': expevents, 'expevents2': expevents2}
    data3 = {'obsevents': obsevents, 'obsevents2': obsevents2}
    m = pd.DataFrame(data1)
    e = pd.DataFrame(data2)
    o = pd.DataFrame(data3)

    # The statistic for the test, which follows, under the null hypothesis,
    # The chi-squared distribution with degrees of freedom equal to amount of groups - 2. Thus 4 - 2 = 2
    tt = sum(sum((np.array(o) - np.array(e)) ** 2 / np.array(e)))
    pvalue = 1 - chi2.cdf(tt, 2)

    return pvalue
train_all_fold_label = []
train_all_fold_pred = []
val_all_fold_label = []
val_all_fold_pred = []
test0_all_fold_label = []
test0_all_fold_pred = []
test1_all_fold_label = []
test1_all_fold_pred = []
test2_all_fold_label = []
test2_all_fold_pred = []
for fold in range(1, 6):
    # 读取csv文件中的数据，不同的cohort，都读取第6列的nomo预测值
    train_data = pd.read_csv(f'../../Utils/nomogram/dif_features_combin_result/fold_{fold}/train_data_{fold}.csv')
    val_data = pd.read_csv(f'../../Utils/nomogram/dif_features_combin_result/fold_{fold}/val_data_{fold}.csv')
    test0_data = pd.read_csv(f'../../Utils/nomogram/dif_features_combin_result/fold_{fold}/test0_data_{fold}.csv')
    test1_data = pd.read_csv(f'../../Utils/nomogram/dif_features_combin_result/fold_{fold}/test1_data_{fold}.csv')
    test2_data = pd.read_csv(f'../../Utils/nomogram/dif_features_combin_result/fold_{fold}/test2_data_{fold}.csv')

    train_label = train_data.iloc[:, 1]
    train_pred = train_data.iloc[:, 7]
    train_all_fold_label.extend(train_label)
    train_all_fold_pred.extend(train_pred)

    val_label = val_data.iloc[:, 1]
    val_pred = val_data.iloc[:, 7]
    val_all_fold_label.extend(val_label)
    val_all_fold_pred.extend(val_pred)

    test0_label = test0_data.iloc[:, 1]
    test0_pred = test0_data.iloc[:, 7]
    test0_all_fold_label.extend(test0_label)
    test0_all_fold_pred.extend(test0_pred)

    test1_label = test1_data.iloc[:, 1]
    test1_pred = test1_data.iloc[:, 7]
    test1_all_fold_label.extend(test1_label)
    test1_all_fold_pred.extend(test1_pred)

    test2_label = test2_data.iloc[:, 1]
    test2_pred = test2_data.iloc[:, 7]
    test2_all_fold_label.extend(test2_label)
    test2_all_fold_pred.extend(test2_pred)

# 定义SciPy通用颜色列表
colors = list(mcolors.TABLEAU_COLORS.keys())

# 绘制校准曲线
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

# 分类器1的校准曲线
fraction_of_positives_1, mean_predicted_value_1 = calibration_curve(train_all_fold_label, train_all_fold_pred,
                                                                    n_bins=8, strategy='quantile')
p_value_1 = hosmer_lemeshow_test(train_all_fold_label[:200], train_all_fold_pred[:200])
plt.plot(mean_predicted_value_1, fraction_of_positives_1, "s-", color=colors[0],
         label="Training cohort ($P$ = {:.3f})".format(p_value_1))

# 分类器2的校准曲线
fraction_of_positives_2, mean_predicted_value_2 = calibration_curve(val_all_fold_label, val_all_fold_pred,
                                                                    n_bins=8, strategy='quantile')
p_value_2 = hosmer_lemeshow_test(val_all_fold_label, val_all_fold_pred)
plt.plot(mean_predicted_value_2, fraction_of_positives_2, "s-", color=colors[1],
         label="Validation cohort ($P$ = {:.3f})".format(p_value_2))

# # 分类器3的校准曲线
# fraction_of_positives_3, mean_predicted_value_3 = calibration_curve(test0_all_fold_label, test0_all_fold_pred,
#                                                                     n_bins=8, strategy='quantile')
# p_value_3 = hosmer_lemeshow_test(test0_all_fold_label, test0_all_fold_pred)
# plt.plot(mean_predicted_value_3, fraction_of_positives_3, "s-", color=colors[2],
#          label="Internal test cohort ($P$ = {:.3f})".format(p_value_3))

# # 分类器5的校准曲线
# fraction_of_positives_4, mean_predicted_value_4 = calibration_curve(test1_all_fold_label, test1_all_fold_pred,
#                                                                     n_bins=8, strategy='quantile')
# p_value_4 = hosmer_lemeshow_test(test1_all_fold_label[:50], test1_all_fold_pred[:50])
# plt.plot(mean_predicted_value_4, fraction_of_positives_4, "s-", color=colors[3],
#          label="External test cohort 1 ($P$ = {:.3f})".format(p_value_4))
#
# 分类器6的校准曲线
fraction_of_positives_5, mean_predicted_value_5 = calibration_curve(test2_all_fold_label, test2_all_fold_pred,
                                                                    n_bins=8, strategy='quantile')
p_value_5 = hosmer_lemeshow_test(test2_all_fold_label[:300], test2_all_fold_pred[:300])
plt.plot(mean_predicted_value_5, fraction_of_positives_5, "s-", color=colors[4],
         label="External test cohort 2 ($P$ = {:.3f})".format(p_value_5))
plt.legend(fontsize=16)
plt.xlabel("Nomogram predicted probability", fontsize=16)
plt.ylabel("Actual frequency", fontsize=16)
plt.title("")
plt.legend(loc="lower right")

# 保存图表为JPEG格式，分辨率为1000
output_file_path = './img/DC_test2.png'
plt.savefig(output_file_path, dpi=1000)

plt.show()
