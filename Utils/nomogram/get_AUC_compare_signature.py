# encoding=utf-8
import copy

import joblib
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc


class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        self._show_result()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A,
                                                                                                    auc_A) * 1 / len(
            V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B,
                                                                                                    auc_B) * 1 / len(
            V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,
                                                                                                       auc_A,
                                                                                                       auc_B) * 1 / len(
            V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z)) * 2

        return z, p

    def _show_result(self):
        z, p = self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold:
            print("There is a significant difference")
        else:
            print("There is NO significant difference")
def cal_mean_CI(list):
    # 计算均值
    mean = np.mean(list)
    # 十折交叉验证所以自由度是10
    n = 5
    # 计算标准差
    std = np.std(list)
    # 双边alpha0.05 = 2.228
    a = 2.571

    return mean, [mean - (a * std / np.sqrt(n)), mean + (a * std / np.sqrt(n))]


def cal_acc_sens_spec_ppv_npv(threshold, y_true, y_pred):
    y_pred = copy.deepcopy(y_pred)
    for j in range(len(y_pred)):
        if y_pred[j] > threshold:
            y_pred[j] = 1
        else:
            y_pred[j] = 0
    y_pred_argmax = y_pred
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_pred_argmax)):
        if y_pred_argmax[i] == y_true[i]:
            if y_true[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_true[i] == 0:
                fp += 1
            else:
                fn += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn + 0.00001)
    spec = tn / (tn + fp + 0.00001)
    ppv = tp / (tp + fp + 0.00001)
    npv = tn / (tn + fn + 0.00001)
    return acc, sens, spec, ppv, npv
all_combin_name = ['Clinical_pred', 'IPN_pred', 'Adipose_pred', 'Clinical_adipose_pred', 'Clinical_IPN_pred', 'Nomogram_pred']
all_csv_name = ['train_data', 'val_data', 'test0_data', 'test1_data', 'test2_data']

for csv_name in all_csv_name:
    print(csv_name)
    # dif_combin
    for combin_index in range(2, 8):
        print(all_combin_name[combin_index-2])
        all_fold_auc = []
        all_fold_acc = []
        all_fold_sens = []
        all_fold_spec = []
        all_fold_ppv = []
        all_fold_npv = []
        for fold in range(1, 6):
            train_data = pd.read_csv(f'../nomogram/dif_features_combin_result/fold_{fold}/train_data_{fold}.csv')
            train_label = np.array(train_data.iloc[:, 1]).astype(int)
            train_y_pred = np.array(train_data.iloc[:, combin_index]).astype(float)
            fpr, tpr, thresholds_train = roc_curve(train_label, train_y_pred)
            train_best_threshold = thresholds_train[np.argmax(tpr - fpr)]
            print(train_best_threshold)
            data = pd.read_csv(f'../nomogram/dif_features_combin_result/fold_{fold}/{csv_name}_{fold}.csv')
            label = np.array(data.iloc[:, 1]).astype(int)
            y_pred = np.array(data.iloc[:, combin_index]).astype(float)
            y_pred_nomo = np.array(data.iloc[:, 7]).astype(float)

            DelongTest(y_pred, y_pred_nomo, label)
            fpr, tpr, thresholds = roc_curve(label, y_pred)
            fold_roc_auc = auc(fpr, tpr)
            fold_acc, fold_sens, fold_spec, fold_ppv, fold_npv = cal_acc_sens_spec_ppv_npv(train_best_threshold, label,
                                                                                           y_pred)
            all_fold_auc.append(fold_roc_auc)
            all_fold_acc.append(fold_acc)
            all_fold_sens.append(fold_sens)
            all_fold_spec.append(fold_spec)
            all_fold_ppv.append(fold_ppv)
            all_fold_npv.append(fold_npv)
        auc_mean, auc_ci = cal_mean_CI(all_fold_auc)
        acc_mean, acc_ci = cal_mean_CI(all_fold_acc)
        sens_mean, sens_ci = cal_mean_CI(all_fold_sens)
        spec_mean, spec_ci = cal_mean_CI(all_fold_spec)
        ppv_mean, ppv_ci = cal_mean_CI(all_fold_ppv)
        npv_mean, npv_ci = cal_mean_CI(all_fold_npv)

        print('AUC: {:.3f} [{:.3f}, {:.3f}]'.format(auc_mean, auc_ci[0], auc_ci[1]))
        print('ACC: {:.3f} [{:.3f}, {:.3f}]'.format(acc_mean, acc_ci[0], acc_ci[1]))
        print('SENS: {:.3f} [{:.3f}, {:.3f}]'.format(sens_mean, sens_ci[0], sens_ci[1]))
        print('SPEC: {:.3f} [{:.3f}, {:.3f}]'.format(spec_mean, spec_ci[0], spec_ci[1]))
        print('PPV: {:.3f} [{:.3f}, {:.3f}]'.format(ppv_mean, ppv_ci[0], ppv_ci[1]))
        print('NPV: {:.3f} [{:.3f}, {:.3f}]'.format(npv_mean, npv_ci[0], npv_ci[1]))
