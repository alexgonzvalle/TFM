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
    hs_max = 14  # int(max([boya.hs.max(), gow.hs.max(), copernicus.VHM0.max(), hs_cal_gow.max(), hs_cal_cop.max()])) + 1

    x_plot = np.linspace(0, hs_max, 11)
    y_gow_plot = beta_gow * x_plot ** gamma_gow

    params, params_covariance = curve_fit(model_function, X_train, y_cal_gow_train, p0=[1, 1])
    beta_gow_cal, gamma_gow_cal = params
    y_cal_gow_plot_train = beta_gow_cal * x_plot ** gamma_gow_cal

    params, params_covariance = curve_fit(model_function, X_test, y_cal_gow_test, p0=[1, 1])
    beta_gow_cal, gamma_gow_cal = params
    y_cal_gow_plot_test = beta_gow_cal * x_plot ** gamma_gow_cal

    y_cop_plot = beta_cop * x_plot ** gamma_cop

    params, params_covariance = curve_fit(model_function, X_train, y_cal_cop_train, p0=[1, 1])
    beta_cop_cal, gamma_cop_cal = params
    y_cal_cop_plot_train = beta_cop_cal * x_plot ** gamma_cop_cal  # ((1 / beta_cop) ** (1 / gamma_cop)) * x_plot ** (1 / gamma_cop)

    params, params_covariance = curve_fit(model_function, X_test, y_cal_cop_test, p0=[1, 1])
    beta_cop_cal, gamma_cop_cal = params
    y_cal_cop_plot_test = beta_cop_cal * x_plot ** gamma_cop_cal

    title = f'Modelo No Lineal {nombre}: y_cal = {((1 / beta_gow) ** (1 / gamma_gow)):.2f}*Hs^{(1 / gamma_gow):.2f}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values,
                                                    ind_train, y_cal_gow_train, y_cal_gow_plot_train,
                                                    ind_test, y_cal_gow_test, y_cal_gow_plot_test,
                                                    y_gow_plot, hs_max,
                                                    'GOW', title, c='purple', fname=f'plot/model/02_NoLineal/{nombre}_noLineal_gow.png', plot=plot)

    title = f'Modelo No Lineal {nombre}: y_cal = {((1 / beta_cop) ** (1 / gamma_cop)):.2f}*Hs^{(1 / gamma_cop):.2f}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values,
                                                    ind_train, y_cal_cop_train, y_cal_cop_plot_train,
                                                    ind_test, y_cal_cop_test, y_cal_cop_plot_test,
                                                    y_cop_plot, hs_max,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/02_NoLineal/{nombre}_noLineal_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'No_Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
