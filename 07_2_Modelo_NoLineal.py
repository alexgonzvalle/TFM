import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from data import get_data
from stats import stats


def model_function(X, beta, gamma):
    return beta * X ** gamma


plot = True
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
    X_gow_train = np.squeeze([gow.hs.values[ind_train]])
    X_gow_test = np.squeeze([gow.hs.values[ind_test]])

    X_cop_train = np.squeeze([copernicus.VHM0.values[ind_train]])
    X_cop_test = np.squeeze([copernicus.VHM0.values[ind_test]])

    # Variable objetivo que queremos predecir/corregir
    y_train = boya.hs.values[ind_train]
    y_test = boya.hs.values[ind_test]

    # Modelo
    params, params_covariance = curve_fit(model_function, X_gow_train, y_train, p0=[1, 1])
    beta_gow, gamma_gow = params
    y_cal_gow_train = beta_gow * X_gow_train ** gamma_gow
    y_cal_gow_test = beta_gow * X_gow_test ** gamma_gow

    params, params_covariance = curve_fit(model_function, X_cop_train, y_train, p0=[1, 1])
    beta_cop, gamma_cop = params
    y_cal_cop_train = beta_cop * X_cop_train ** gamma_cop
    y_cal_cop_test = beta_cop * X_cop_test ** gamma_cop

    # import matplotlib.pyplot as plt
    #
    # x = [i for i in range(len(y_gow))]
    # plt.plot(x, boya.hs.values, label='Boya')
    # plt.plot(x, y_gow, label='GOW')
    # plt.plot(x, y_cal_gow, label='Calibrada')
    # plt.legend()
    # plt.show()

    # Dibujar
    aux_title = r'$Hs_{cal}$'
    title = f'Modelo No Lineal {nombre}: {aux_title}={beta_gow:.2f}*Hs^{gamma_gow:.2f}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values,
                                                    ind_train, y_cal_gow_train, ind_test, y_cal_gow_test,
                                                    'GOW', title, c='purple', plot=plot, fname=f'plot/model/02_NoLineal/{nombre}_noLineal_gow.png')

    title = f'Modelo No Lineal {nombre}: {aux_title}={beta_cop:.2f}*Hs^{gamma_cop:.2f}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values,
                                                    ind_train, y_cal_cop_train, ind_test, y_cal_cop_test,
                                                    'IBI', title, c='orange', plot=plot, fname=f'plot/model/02_NoLineal/{nombre}_noLineal_ibi.png')

    df_res.loc[len(df_res.index)] = [nombre, 'No_Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
