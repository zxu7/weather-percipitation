import os
import pickle
import multiprocessing
import numpy as np
import pandas as pd
import statsmodels.api as sm
from netCDF4 import Dataset
from datetime import datetime
from percipitation.read_data import ncdump
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBRegressor


DATA_DIR = '/Users/harryxu/Downloads/data/因子/' if \
    os.environ['PWD'] == '/Users/harryxu/repos/weather-percipitation' \
    else '/home/wangli/code/projects/weather/nc/'
AIR_LEVEL = 850
HGT_LEVEL = 500
UWIND_LEVEL = 850
VWIND_LEVEL = 850
HGT_PATH = DATA_DIR + 'hgt.mon.mean.nc'
AIR_PATH = DATA_DIR + 'air.mon.mean.nc'
SLP_PATH = DATA_DIR + 'slp.mon.mean.nc'
SST_PATH = DATA_DIR + 'sst.mnmean.v4.nc'
UWND_PATH = DATA_DIR + 'uwnd.mon.mean.nc'
VWND_PATH = DATA_DIR + 'vwnd.mon.mean.nc'
LABEL_PATH = DATA_DIR + 'percipitation.xlsx'
K_FOLD_SPLITS = 5
K_FOLDS_TO_RUN = 3 # K_FOLDS_TO_RUN <= K_FOLD_SPLITS
MAX_N_FEATURE_SELECT = 10
CITIES = '0' # tuple(range(140)) # tuple(range(140)) # tuple([str(x) for x in range(3)]) + ('50', '100')
ALGOS_TO_RUN = ('lasso', 'xgboost', 'knn') # lasso, xgboost, knn, stepwise
TRAIN_RECIPE_PATH = 'data/recipe.p'
OUTPUT_PATH = 'data/output.p'


def run_ncdump(path, print_features=False):
    out1, out2, out3 = None, None, None
    dataset = Dataset(path, 'r')
    if print_features:
        out1, out2, out3 = ncdump(dataset)
    return dataset, out1, out2, out3


def get_data(hgt_dataset, air_dataset, slp_dataset, sst_dataset, uwnd_dataset, vwnd_dataset):
    hgt_data = hgt_dataset['hgt'][:][:, int(np.where(hgt_dataset['level'][:] == HGT_LEVEL)[0][0]), :, :]
    air_data = air_dataset['air'][:][:, int(np.where(hgt_dataset['level'][:] == AIR_LEVEL)[0][0]), :, :]
    uwnd_data = uwnd_dataset['uwnd'][:][:, int(np.where(hgt_dataset['level'][:] == UWIND_LEVEL)[0][0]), :, :]
    vwnd_data = vwnd_dataset['vwnd'][:][:, int(np.where(hgt_dataset['level'][:] == VWIND_LEVEL)[0][0]), :, :]

    slp_data = slp_dataset['slp'][:]
    sst_data = sst_dataset['sst'][:][-845:]
    return hgt_data, air_data, slp_data, sst_data, uwnd_data, vwnd_data


def get_label(path):
    """52 by 140 matrix; first 2 rows are lattitude, longitude info"""
    pd_data = pd.read_excel(path)
    n_features, n_cities = pd_data.shape
    features = pd_data.values[1:, :]
    return features


def prepare_data_for_knn(all_X, all_Y, t_before=0, t_after=0):
    """preprocess data into format:
    [station]: year (1961 - 2010) - flatten features around [July - t_before, July + t_after]
    for instance, t_before=t_after=1 yields features from June, July, August
    """

    # 845 timestamps start from 1948-1-1, 1948-2-1, ...
    # all_Y starts from summer (June July August) of 1961

    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    def get_months_data(t_before, t_after):
        all_features = []
        hgt_data, air_data, slp_data, sst_data, uwnd_data, vwnd_data = all_X
        index_1961_7_1 = diff_month(datetime(1961, 7, 1), datetime(1948, 1, 1))
        index_2018_5_1 = diff_month(datetime(2018, 5, 1), datetime(1948, 1, 1))  # 844
        for data1 in all_X:
            selected_list_indices = [list(range(i - t_before, i + t_after + 1)) for i
                                     in range(index_1961_7_1, index_2018_5_1, 12)]
            data1_features = np.array([data1[l].reshape(-1) for l in selected_list_indices])
            all_features.append(data1_features)
        return np.column_stack(all_features)

    knn_data = {}
    n_features, n_cities = all_Y.shape
    for i in tqdm(range(n_cities)):
        knn_data[str(i)] = {}
        city_Y_data = all_Y[2:, i]
        city_X_data = get_months_data(t_before, t_after)

        knn_data[str(i)]['lon'] = all_Y[0, i]
        knn_data[str(i)]['lat'] = all_Y[1, i]
        knn_data[str(i)]['X'] = city_X_data[:len(city_Y_data), ]  # 50 years
        knn_data[str(i)]['Y'] = city_Y_data
        knn_data[str(i)]['test_X'] = city_X_data[len(city_Y_data):]
    return knn_data


def create_pickle(file, filepath):
    dir_filepath = os.path.dirname(filepath)
    if not os.path.isdir(dir_filepath):
        os.makedirs(dir_filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)
    print("file created at {}...".format(filepath))


def train(X, Y):
    """search params and try different algs in ALGOS_TO_RUN;
    save a recipe, which contains the best algo (xgboost, ensemble, lasso, etc.)
     for the city"""
    # K-fold crossvalidation
    kfold = KFold(n_splits=K_FOLD_SPLITS)
    train_Ys, valid_Ys, train_metrics, valid_metrics, train_ensemble, valid_ensemble = {}, {}, {}, {}, {}, {}

    for algo in ALGOS_TO_RUN:
        train_Ys[algo] = []
        valid_Ys[algo] = []
        train_metrics[algo] = []
        valid_metrics[algo] = []
        train_ensemble = []
        valid_ensemble = []

    for kf_id, (train_indices, valid_indices) in enumerate(kfold.split(X)):
        if kf_id >= K_FOLDS_TO_RUN:
            break
        train_X = X[train_indices]
        val_X = X[valid_indices]
        train_Y = Y[train_indices]
        val_Y = Y[valid_indices]
        train_ensemble1 = []
        valid_ensemble1 = []

        # PCA
        if 'xgboost' in ALGOS_TO_RUN or 'lasso' in ALGOS_TO_RUN:
            pca = PCA(n_components=40)
            pca.fit(train_X)
            print("pca explained variance ratio: {}...".format(sum(pca.explained_variance_ratio_)))
            pca_train_X = pca.transform(train_X)
            pca_val_X = pca.transform(val_X)

        if 'lasso' in ALGOS_TO_RUN:
            # lasso
            model = Lasso(alpha=1.0)
            model.fit(pca_train_X, train_Y)
            pred_train_Y = model.predict(pca_train_X)
            pred_valid_Y = model.predict(pca_val_X)

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['lasso'].append(train_Y)
            valid_Ys['lasso'].append(val_Y)
            train_metrics['lasso'].append(fold_train_mae)
            valid_metrics['lasso'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

        if 'xgboost' in ALGOS_TO_RUN:
            # Xgboost
            model = XGBRegressor(
                learning_rate=0.01,  # 默认0.3
                n_estimators=500,  # 树的个数
                max_depth=3,
                # min_child_weight=1,
                # gamma=0,
                # subsample=0.8,
                # colsample_bytree=0.8,
                # scale_pos_weight=1
            )
            model.fit(pca_train_X, train_Y)
            pred_train_Y = model.predict(pca_train_X)
            pred_valid_Y = model.predict(pca_val_X)

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['xgboost'].append(train_Y)
            valid_Ys['xgboost'].append(val_Y)
            train_metrics['xgboost'].append(fold_train_mae)
            valid_metrics['xgboost'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

        if 'stepwise' in ALGOS_TO_RUN:
            # stepwise forward selection (by p value)
            all_feature_indices = set(range(train_X.shape[1]))
            selected_feature_indices = [0, 1, 2, 3, 4, 5]

            def mp_get_pvalue(ind):
                """get p value for newly added feature, which is the last feature"""
                model = sm.OLS(train_Y, train_X[:, list(selected_feature_indices) + [ind]]).fit()
                pvalue = model.pvalues[-1]
                return pvalue

            def get_pvalue(Y, X):
                """get p value for newly added feature, which is the last feature"""
                model = sm.OLS(Y, X).fit()
                pvalue = model.pvalues[-1]
                return pvalue

            while len(selected_feature_indices) < MAX_N_FEATURE_SELECT:
                unselected_feature_indices = all_feature_indices - set(selected_feature_indices)
                unselected_feature_indice1_pvalue0 = 100  # some random large p-value
                selected_feature_index = 0  # some random index

                # multi-processing (doesn't seem to speed up, moreover, costs a lot more time)
                # import time
                # pool = multiprocessing.Pool(4)
                # unselected_feature_indices_list = list(unselected_feature_indices)
                # start_time = time.time()
                # unselected_feature_pvalues = pool.map(mp_get_pvalue, unselected_feature_indices_list)
                # print("takes {}...".format(time.time() - start_time))
                # selected_feature_index = unselected_feature_indices_list[int(np.argmin(unselected_feature_pvalues))]
                # selected_feature_indices += [selected_feature_index]

                # construct array of pvalues
                unselected_feature_indices_list = list(unselected_feature_indices)
                unselected_feature_pvalues = [get_pvalue(train_Y, train_X[:, list(selected_feature_indices + [ind])])
                                              for ind in tqdm(unselected_feature_indices_list)]
                selected_feature_index = unselected_feature_indices_list[int(np.argmin(unselected_feature_pvalues))]
                selected_feature_indices += [selected_feature_index]

            model = sm.OLS(train_Y, train_X[:, selected_feature_indices]).fit()
            pred_train_Y = model.predict(train_X[:, selected_feature_indices])
            pred_valid_Y = model.predict(val_X[:, selected_feature_indices])
            # print("avg Y: {}, train mae is: {}, val mae is: {}".format(np.mean(Y),
            #                                                            np.mean(abs(train_Y - pred_train_Y)),
            #                                                            np.mean(abs(val_Y - pred_valid_Y))))
            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['stepwise'].append(train_Y)
            valid_Ys['stepwise'].append(val_Y)
            train_metrics['stepwise'].append(fold_train_mae)
            valid_metrics['stepwise'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

        if 'knn' in ALGOS_TO_RUN:
            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(train_X, train_Y)
            pred_train_Y = model.predict(train_X)
            factor = np.mean([train_Y[i] / pred_train_Y[i] for i in range(len(pred_train_Y))])
            pred_train_Y = factor * pred_train_Y
            pred_valid_Y = factor * model.predict(val_X)

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['knn'].append(train_Y)
            valid_Ys['knn'].append(val_Y)
            train_metrics['knn'].append(fold_train_mae)
            valid_metrics['knn'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

        train_ensemble.append(np.mean(abs(np.sum(np.array(train_ensemble1), axis=0) / len(ALGOS_TO_RUN))))
        valid_ensemble.append(np.mean(abs(np.sum(np.array(valid_ensemble1), axis=0) / len(ALGOS_TO_RUN))))

    for k in train_metrics:
        print("{}: avg train_Y: {}, avg val_Y: {}, train mae: {}, val mae: {}".format(k,
                                                                                      np.mean(train_Ys[k]),
                                                                                      np.mean(valid_Ys[k]),
                                                                                      np.mean(train_metrics[k]),
                                                                                      np.mean(valid_metrics[k])))
    print("ensemble: train mae: {}, val mae: {}".format(np.mean(train_ensemble),
                                                        np.mean(valid_ensemble)))
    algos = [k for k in train_metrics] + ['ensemble-{}'.format('-'.join(train_metrics.keys()))]
    algos_scores = [np.mean(valid_metrics[k]) for k in train_metrics] + [np.mean(valid_ensemble)]
    return algos[int(np.argmin(algos_scores))]


def predict(X, Y, test_X, best_algo):
    # PCA
    pred_test_Ys = []
    if 'xgboost' in best_algo or 'lasso' in best_algo:
        pca = PCA(n_components=40)
        pca.fit(X)
        print("pca explained variance ratio: {}...".format(sum(pca.explained_variance_ratio_)))
        pca_X = pca.transform(X)
        pca_test_X = pca.transform(test_X)

    if 'lasso' in best_algo:
        # lasso
        model = Lasso(alpha=1.0)
        model.fit(pca_X, Y)
        pred_test_Y = model.predict(pca_test_X)
        pred_test_Ys.append(pred_test_Y)

    if 'xgboost' in best_algo:
        # Xgboost
        model = XGBRegressor(
            learning_rate=0.01,  # 默认0.3
            n_estimators=500,  # 树的个数
            max_depth=3,
            # min_child_weight=1,
            # gamma=0,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # scale_pos_weight=1
        )
        model.fit(pca_X, Y)
        pred_test_Y = model.predict(pca_test_X)
        pred_test_Ys.append(pred_test_Y)

    if 'stepwise' in best_algo:
        # stepwise forward selection (by p value)
        all_feature_indices = set(range(X.shape[1]))
        selected_feature_indices = [0, 1, 2, 3, 4, 5]

        def mp_get_pvalue(ind):
            """get p value for newly added feature, which is the last feature"""
            model = sm.OLS(Y, X[:, list(selected_feature_indices) + [ind]]).fit()
            pvalue = model.pvalues[-1]
            return pvalue

        def get_pvalue(Y, X):
            """get p value for newly added feature, which is the last feature"""
            model = sm.OLS(Y, X).fit()
            pvalue = model.pvalues[-1]
            return pvalue

        while len(selected_feature_indices) < MAX_N_FEATURE_SELECT:
            unselected_feature_indices = all_feature_indices - set(selected_feature_indices)
            unselected_feature_indice1_pvalue0 = 100  # some random large p-value
            selected_feature_index = 0  # some random index

            # multi-processing (doesn't seem to speed up, moreover, costs a lot more time)
            # import time
            # pool = multiprocessing.Pool(4)
            # unselected_feature_indices_list = list(unselected_feature_indices)
            # start_time = time.time()
            # unselected_feature_pvalues = pool.map(mp_get_pvalue, unselected_feature_indices_list)
            # print("takes {}...".format(time.time() - start_time))
            # selected_feature_index = unselected_feature_indices_list[int(np.argmin(unselected_feature_pvalues))]
            # selected_feature_indices += [selected_feature_index]

            # construct array of pvalues
            unselected_feature_indices_list = list(unselected_feature_indices)
            unselected_feature_pvalues = [get_pvalue(Y, X[:, list(selected_feature_indices + [ind])])
                                          for ind in tqdm(unselected_feature_indices_list)]
            selected_feature_index = unselected_feature_indices_list[int(np.argmin(unselected_feature_pvalues))]
            selected_feature_indices += [selected_feature_index]

        model = sm.OLS(Y, X[:, selected_feature_indices]).fit()
        pred_test_Y = model.predict(test_X[:, selected_feature_indices])
        pred_test_Ys.append(pred_test_Y)

    if 'knn' in best_algo:
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X, Y)
        pred_train_Y = model.predict(X)
        factor = np.mean([Y[i] / pred_train_Y[i] for i in range(len(pred_train_Y))])
        pred_test_Y = factor * model.predict(test_X)
        pred_test_Ys.append(pred_test_Y)

    if 'ensemble' in best_algo:
        n_algo = len(pred_test_Ys)
        pred_test_Y = np.sum(np.array(pred_test_Ys), axis=0) / n_algo

    return pred_test_Y


if __name__ == "__main__":
    # Processing data
    # hgt
    hgt_dataset, hgt_nc_attrs, hgt_nc_dims, hgt_nc_vars = run_ncdump(HGT_PATH, print_features=True)  # 845, 17, 73, 144
    # air
    air_dataset, air_nc_attrs, air_nc_dims, air_nc_vars = run_ncdump(AIR_PATH)  # 845, 17, 73, 144
    # slp
    slp_dataset, slp_nc_attrs, slp_nc_dims, slp_nc_vars = run_ncdump(SLP_PATH)  # 845, 73, 144
    # sst, time_bnds
    sst_dataset, sst_nc_attrs, sst_nc_dims, sst_nc_vars = run_ncdump(
        SST_PATH)  # 1961, 89, 180 (might be missing values here)
    # uwnd
    uwnd_dataset, uwnd_nc_attrs, uwnd_nc_dims, uwnd_nc_vars = run_ncdump(UWND_PATH)  # 845, 17, 73, 144
    # vwnd
    vwnd_dataset, vwnd_nc_attrs, vwnd_nc_dims, vwnd_nc_vars = run_ncdump(VWND_PATH)  # 845, 17, 73, 144
    print("all .nc files loaded...")

    hgt_data, air_data, slp_data, sst_data, uwnd_data, vwnd_data = get_data(hgt_dataset, air_dataset, slp_dataset,
                                                                            sst_dataset, uwnd_dataset, vwnd_dataset)
    print("LEVEL data extracted...")

    raw_X = (hgt_data, air_data, slp_data, sst_data, uwnd_data, vwnd_data)
    raw_Y = get_label(LABEL_PATH)
    knn_data = prepare_data_for_knn(raw_X, raw_Y, t_before=1, t_after=1)
    recipe = {}

    # train
    for i in CITIES:
        print("building model for city {}...".format(i))

        X = knn_data[i]['X']
        Y = knn_data[i]['Y']

        # deal with -9.xx e36 values
        X[X < -1000] = 0

        best_algo = train(X, Y)
        recipe[i] = best_algo
    create_pickle(recipe, TRAIN_RECIPE_PATH)

    # predict
    output = {}
    # TODO: load recipe
    for i in CITIES:
        print("predicting for city {} based on trained best_algo...".format(i))

        X = knn_data[i]['X']
        Y = knn_data[i]['Y']
        test_X = knn_data[i]['test_X']

        # deal with -9.xx e36 values
        X[X < -1000] = 0
        test_X[test_X < -1000] = 0

        best_algo = recipe[i]
        pred_test_Y = predict(X, Y, test_X, best_algo)
        output[i] = pred_test_Y
    create_pickle(output, OUTPUT_PATH)