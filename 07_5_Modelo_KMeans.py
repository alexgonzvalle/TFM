import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import pandas as pd
from data import get_data
from stats import stats


def create_model_kmeans(K, X_train_norm, X_train, y_train, X_test_norm, X_test):
    model = KMeans(n_clusters=K, random_state=1, n_init='auto').fit(X_train_norm)
    k_train = model.predict(X_train_norm)
    k_test = model.predict(X_test_norm)
    y_cal_train = np.zeros_like(X_train_norm[:, 0])
    y_cal_test = np.zeros_like(X_test_norm[:, 0])

    factor_hs = []
    for k in range(K):
        ind_train_c = np.argwhere(k_train == k).reshape(-1)
        ind_test_c = np.argwhere(k_test == k).reshape(-1)

        y = y_train.ravel()[ind_train_c]
        y[y == 0] = np.NaN
        factor_hs.append(np.nanmean(y / X_train[ind_train_c, 0]))

        y_cal_train[ind_train_c] = X_train[ind_train_c, 0] * factor_hs[k]
        y_cal_test[ind_test_c] = X_test[ind_test_c, 0] * factor_hs[k]

    return y_cal_train, y_cal_test


def get_best_kmeans(k_min, k_max, X_train_norm, X_train, y_train, X_test_norm, X_test, y_test, loc, name_model):
    K = range(k_min, k_max)

    out = []
    for k in K:
        model = KMeans(n_clusters=k, random_state=1, n_init='auto').fit(X_train_norm)

        k_train = model.predict(X_train_norm)
        k_test = model.predict(X_test_norm)

        factor_hs = []
        y_cal_test = np.zeros_like(X_test_norm[:, 0])
        for kk in range(k):
            ind_train_c = np.argwhere(k_train == kk).reshape(-1)
            ind_test_c = np.argwhere(k_test == kk).reshape(-1)

            y = y_train.ravel()[ind_train_c]
            y[y == 0] = np.NaN
            factor_hs.append(np.nanmean(y / X_train[ind_train_c, 0]))
            y_cal_test[ind_test_c] = X_test[ind_test_c, 0] * factor_hs[kk]

        mse = mean_squared_error(y_test.ravel(), y_cal_test)

        dict_res = {}
        dict_res['k'] = k
        dict_res['mse'] = mse
        dict_res['loc'] = loc
        dict_res['model'] = name_model
        out.append(dict_res)

    pd.DataFrame(out).to_csv('out_model_kmeans.csv', mode='a', index=False, header=False)


plot = False
df_boya = pd.read_csv('boyas.csv')
df_res = pd.read_csv('res.csv')

# MODELO LINEAL
for nombre in df_boya['Nombre']:
    boya, copernicus, gow = get_data(nombre)

    # Train y test
    goal_time = np.datetime64('2013-01-01T00:00:00.000000')
    inds_time = np.argwhere(boya.time.values == goal_time)
    while len(inds_time) == 0:
        goal_time -= pd.DateOffset(months=1)
        inds_time = np.argwhere(boya.time.values == goal_time)
    ind_time = int(inds_time[0])
    ind_train = np.array([i for i in range(ind_time)])
    ind_test = np.array([i for i in range(ind_time, len(boya.time))])

    # Separar las variables predictoras (X) y la variable objetivo (y)
    X_gow_train = np.array([gow.hs.values[ind_train], np.deg2rad(gow.dir.values[ind_train])]).T
    X_gow_test = np.array([gow.hs.values[ind_test], np.deg2rad(gow.dir.values[ind_test])]).T

    X_cop_train = np.array([copernicus.VHM0.values[ind_train], np.deg2rad(copernicus.VMDR.values[ind_train])]).T
    X_cop_test = np.array([copernicus.VHM0.values[ind_test], np.deg2rad(copernicus.VMDR.values[ind_test])]).T

    # Variable objetivo que queremos predecir/corregir
    y_train = boya.hs.values[ind_train].reshape(-1, 1)
    y_test = boya.hs.values[ind_test].reshape(-1, 1)

    # Normalizar los datos
    scaler = StandardScaler()

    X_gow_train_norm = scaler.fit_transform(X_gow_train)
    X_gow_test_norm = scaler.fit_transform(X_gow_test)
    X_cop_train_norm = scaler.fit_transform(X_cop_train)
    X_cop_test_norm = scaler.fit_transform(X_cop_test)

    # Encontrar los mejores hiperparámetros
    k = 200
    # get_best_kmeans(2, 501, X_gow_train_norm, X_gow_train, y_train, X_gow_test_norm, X_gow_test, y_test, nombre, 'GOW')
    y_cal_gow_train, y_cal_gow_test = create_model_kmeans(k, X_gow_train_norm, X_gow_train, y_train, X_gow_test_norm, X_gow_test)

    # get_best_kmeans(2, 501, X_cop_train_norm, X_cop_train, y_train, X_cop_test_norm, X_cop_test, y_test, nombre, 'COP')
    y_cal_cop_train, y_cal_cop_test = create_model_kmeans(k, X_cop_train_norm, X_cop_train, y_train, X_cop_test_norm, X_cop_test)

    # Dibujar
    title = f'Modelo KMeans {nombre}. K: {k}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values,
                                                    ind_train, y_cal_gow_train, ind_test, y_cal_gow_test,
                                                    'GOW', title, c='purple', plot=plot, fname=f'plot/model/05_KMeans/{nombre}_kmeans_gow.png')

    title = f'Modelo KMeans {nombre}. K: {k}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values,
                                                    ind_train, y_cal_cop_train, ind_test, y_cal_cop_test,
                                                    'IBI', title, c='orange', plot=plot, fname=f'plot/model/05_KMeans/{nombre}_kmeans_ibi.png')

    df_res.loc[len(df_res.index)] = [nombre, 'KMeans', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
