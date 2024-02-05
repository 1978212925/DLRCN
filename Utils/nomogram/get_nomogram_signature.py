import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, make_scorer, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset
from runner.Runner_utils.evaluation_index import ClassificationMetric
from Utils.machine_learning.evaluate_cal_mean_ci import cal_mean_CI, cal_acc_sens_spec_ppv_npv
def get_intra_peri_signature():
    # {'n_estimators': 25, 'min_samples_leaf': 12, 'max_depth': 5}
    for fold in range(1, 6):
        train_data = pd.read_csv(
            f'../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_train_data_{fold}.csv')
        val_data = pd.read_csv(
            f'../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_val_data_{fold}.csv')
        test0_data = pd.read_csv(
            f"../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_test0_data_{fold}.csv")
        test1_data = pd.read_csv(
            f"../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_test1_data_{fold}.csv")
        test2_data = pd.read_csv(
            f"../feature_selection/LASSO/node_fold_data_selected_R_LASSO/fold_{fold}/selected_test2_data_{fold}.csv")

        train_img_ids = np.array(train_data.iloc[:, 0]).astype(int)
        train_labels = np.array(train_data.iloc[:, 1]).astype(int)
        train_features = np.array(train_data.iloc[:, 2:]).astype(float)
        random_forest_regressor = RandomForestRegressor(n_estimators=25, random_state=42, min_samples_leaf=12,
                                                        max_depth=5)
        random_forest_regressor.fit(train_features, train_labels)
        y_pred_train = random_forest_regressor.predict(train_features)
        train_final_data = {'Image_ID': train_img_ids, 'Label': train_labels, 'IPN_signature': y_pred_train}
        train_df = pd.DataFrame(train_final_data)
        train_df.to_csv(f'nomogram_signature/fold_{fold}/train_data_{fold}.csv', index=False)

        val_img_ids = np.array(val_data.iloc[:, 0]).astype(int)
        val_labels = np.array(val_data.iloc[:, 1]).astype(int)
        val_features = np.array(val_data.iloc[:, 2:]).astype(float)
        y_pred_valid = random_forest_regressor.predict(val_features)
        val_final_data = {'Image_ID': val_img_ids, 'Label': val_labels, 'IPN_signature': y_pred_valid}
        val_df = pd.DataFrame(val_final_data)
        val_df.to_csv(f'nomogram_signature/fold_{fold}/val_data_{fold}.csv', index=False)

        test0_img_ids = np.array(test0_data.iloc[:, 0]).astype(int)
        test0_labels = np.array(test0_data.iloc[:, 1]).astype(int)
        test0_features = np.array(test0_data.iloc[:, 2:]).astype(float)
        y_pred_test0 = random_forest_regressor.predict(test0_features)
        test0_final_data = {'Image_ID': test0_img_ids, 'Label': test0_labels, 'IPN_signature': y_pred_test0}
        test0_df = pd.DataFrame(test0_final_data)
        test0_df.to_csv(f'nomogram_signature/fold_{fold}/test0_data_{fold}.csv', index=False)

        test1_img_ids = np.array(test1_data.iloc[:, 0]).astype(int)
        test1_labels = np.array(test1_data.iloc[:, 1]).astype(int)
        test1_features = np.array(test1_data.iloc[:, 2:]).astype(float)
        y_pred_test1 = random_forest_regressor.predict(test1_features)
        test1_final_data = {'Image_ID': test1_img_ids, 'Label': test1_labels, 'IPN_signature': y_pred_test1}
        test1_df = pd.DataFrame(test1_final_data)
        test1_df.to_csv(f'nomogram_signature/fold_{fold}/test1_data_{fold}.csv', index=False)

        test2_img_ids = np.array(test2_data.iloc[:, 0]).astype(int)
        test2_labels = np.array(test2_data.iloc[:, 1]).astype(int)
        test2_features = np.array(test2_data.iloc[:, 2:]).astype(float)
        y_pred_test2 = random_forest_regressor.predict(test2_features)
        test2_final_data = {'Image_ID': test2_img_ids, 'Label': test2_labels, 'IPN_signature': y_pred_test2}
        test2_df = pd.DataFrame(test2_final_data)
        test2_df.to_csv(f'nomogram_signature/fold_{fold}/test2_data_{fold}.csv', index=False)
if __name__ == '__main__':
    get_intra_peri_signature()


