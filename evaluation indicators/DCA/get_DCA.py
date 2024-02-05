import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model1, net_benefit_model2, net_benefit_model3, net_benefit_model4, net_benefit_model5
    ,net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model1, color='pink', label='Clinical model')
    ax.plot(thresh_group, net_benefit_model2, color='purple', label='IPN signature')
    ax.plot(thresh_group, net_benefit_model3, color='blue', label='Adipose radiomic model')
    ax.plot(thresh_group, net_benefit_model4, color='green', label='Clinical IPN model')
    ax.plot(thresh_group, net_benefit_model5, color='red', label='DLRN')


    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'All')
    ax.plot((0, 1), (0, 0), color='gray', linestyle=':', label='None')
    ax.legend(loc='lower left')
    ax.legend(fontsize=15)
    # #Fill，显示出模型较于treat all和treat none好的部分
    # y2 = np.maximum(net_benefit_all, 0)
    # y1 = np.maximum(net_benefit_model, y2)
    # ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(-0.2, 1)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold probability',fontsize=15
        )
    ax.set_ylabel(
        ylabel = 'Net benefit',fontsize=15
        )
    # 去掉网格线
    ax.grid(False)
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))

    return ax

def find_optimal_threshold(net_benefit_scores):
    optimal_start = 0
    optimal_end = 0
    max_length = 0
    current_start = 0

    for i in range(1, len(net_benefit_scores[0])):
        is_optimal = True
        for score in net_benefit_scores[1:]:
            if net_benefit_scores[0][i] <= score[i]:
                is_optimal = False
                break

        if is_optimal:
            current_end = i
            if current_end - current_start > max_length:
                max_length = current_end - current_start
                optimal_start = current_start
                optimal_end = current_end
        else:
            current_start = i

    return optimal_start, optimal_end


if __name__ == '__main__':
    all_fold_y_true = []
    all_fold_clinical_pred = []
    all_fold_adipose_pred = []
    all_fold_IPN_pred = []
    all_fold_clinical_IPN_pred = []
    all_fold_DLRN_pred = []
    data_name = 'test2'
    for fold in range(1, 6):
        data = pd.read_csv(f'../../Utils/nomogram/dif_features_combin_result/fold_{fold}/{data_name}_data_{fold}.csv')
        label = np.array(data.iloc[:, 1]).astype(int)
        y_pred_clinical = np.array(data.iloc[:, 2]).astype(float)
        y_pred_adipose = np.array(data.iloc[:, 3]).astype(float)
        y_pred_ipn = np.array(data.iloc[:, 4]).astype(float)
        y_pred_clinical_ipn = np.array(data.iloc[:, 5]).astype(float)
        y_pred_nomo = np.array(data.iloc[:, 7]).astype(float)
        all_fold_y_true.extend(label)
        all_fold_clinical_pred.extend(y_pred_clinical)
        all_fold_adipose_pred.extend(y_pred_adipose)
        all_fold_IPN_pred.extend(y_pred_ipn)
        all_fold_clinical_IPN_pred.extend(y_pred_clinical_ipn)
        all_fold_DLRN_pred.extend(y_pred_nomo)
    all_fold_y_true = np.array(all_fold_y_true)
    all_fold_clinical_pred = np.array(all_fold_clinical_pred)
    all_fold_adipose_pred = np.array(all_fold_adipose_pred)
    all_fold_IPN_pred = np.array(all_fold_IPN_pred)
    all_fold_clinical_IPN_pred = np.array(all_fold_clinical_IPN_pred)
    all_fold_DLRN_pred = np.array(all_fold_DLRN_pred)

    thresh_group = np.arange(0, 1, 0.01)
    net_benefit_clinical = calculate_net_benefit_model(thresh_group, all_fold_clinical_pred, all_fold_y_true)
    net_benefit_adipose = calculate_net_benefit_model(thresh_group, all_fold_adipose_pred, all_fold_y_true)
    net_benefit_ipn = calculate_net_benefit_model(thresh_group, all_fold_IPN_pred, all_fold_y_true)
    net_benefit_clinical_ipn = calculate_net_benefit_model(thresh_group, all_fold_clinical_IPN_pred, all_fold_y_true)
    net_benefit_nomo = calculate_net_benefit_model(thresh_group, all_fold_DLRN_pred, all_fold_y_true)


    net_benefit_scores = [net_benefit_nomo, net_benefit_clinical_ipn, net_benefit_adipose, net_benefit_ipn, net_benefit_clinical]

    optimal_start, optimal_end = find_optimal_threshold(net_benefit_scores)

    if optimal_start == 0 and optimal_end == 0:
        print("No optimal threshold range found")
    else:
        print(f"Optimal threshold range: [{thresh_group[optimal_start]}, {thresh_group[optimal_end]}]")
    net_benefit_all = calculate_net_benefit_all(thresh_group, all_fold_y_true)
    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_clinical, net_benefit_ipn, net_benefit_adipose, net_benefit_clinical_ipn, net_benefit_nomo,
                  net_benefit_all)
    fig.savefig(f'./img/DCA_{data_name}.png', dpi=1000)
    plt.show()