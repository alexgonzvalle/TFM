import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from data import get_data
from stats import stats


def model_function(X, beta, gamma):
    return beta * X ** gamma


plot = False
df_boya = pd.read_csv('boyas.csv')
df_res = pd.read_csv('res.csv')

# MODELO Polinomial
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
    X_train = boya.hs.values[ind_train]
    X_test = boya.hs.values[ind_test]

    # Variable objetivo que queremos predecir/corregir
    y_gow_train = gow.hs.values[ind_train]
    y_gow_test = gow.hs.values[ind_test]

    y_cop_train = copernicus.VHM0.values[ind_train]
    y_cop_test = copernicus.VHM0.values[ind_test]

    # Modelo
    params, params_covariance = curve_fit(model_function, X_train, y_gow_train, p0=[1, 1])
    beta_gow, gamma_gow = params
    y_cal_gow_train = ((1 / beta_gow) ** (1 / gamma_gow)) * y_gow_train ** (1 / gamma_gow)
    y_cal_gow_test = ((1 / beta_gow) ** (1 / gamma_gow)) * y_gow_test ** (1 / gamma_gow)

    params, params_covariance = curve_fit(model_function, X_train, y_cop_train, p0=[1, 1])
    beta_cop, gamma_cop = params
    y_cal_cop_train = ((1 / beta_cop) ** (1 / gamma_cop)) * y_cop_train ** (1 / gamma_cop)
    y_cal_cop_test = ((1 / beta_cop) ** (1 / gamma_cop)) * y_cop_test ** (1 / gamma_cop)

    # import matplotlib.pyplot as plt
    #
    # x = [i for i in range(len(y_gow))]
    # plt.plot(x, boya.hs.values, label='Boya')
    # plt.plot(x, y_gow, label='GOW')
    # plt.plot(x, y_cal_gow, label='Calibrada')
    # plt.legend()
    # plt.show()

    # Dibujar
    title = f'Modelo No Lineal {nombre}: y_cal = {((1 / beta_gow) ** (1 / gamma_gow)):.2f}*Hs^{(1 / gamma_gow):.2f}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values,
                                                    ind_train, y_cal_gow_train, ind_test, y_cal_gow_test,
                                                    'GOW', title, c='purple', plot=plot, fname=f'plot/model/02_NoLineal/{nombre}_noLineal_gow.png')

    title = f'Modelo No Lineal {nombre}: y_cal = {((1 / beta_cop) ** (1 / gamma_cop)):.2f}*Hs^{(1 / gamma_cop):.2f}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values,
                                                    ind_train, y_cal_cop_train, ind_test, y_cal_cop_test,
                                                    'Copernicus', title, c='orange', plot=plot, fname=f'plot/model/02_NoLineal/{nombre}_noLineal_cop.png')

    df_res.loc[len(df_res.index)] = [nombre, 'No_Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
