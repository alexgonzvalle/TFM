import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from data import get_data
import datetime as dt
from stats import stats


plot = True
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

    # Separar las variables predictoras (X) y la variable objetivo (y)
    X_train = np.array([boya.hs.values[:ind_time]]).T
    X_test = np.array([boya.hs.values[ind_time:]]).T

    y_gow_train = gow.hs.values[:ind_time]  # Variable objetivo que queremos predecir/corregir
    y_gow_test = gow.hs.values[ind_time:]  # Variable objetivo que queremos predecir/corregir

    y_cop_train = copernicus.VHM0.values[:ind_time]  # Variable objetivo que queremos predecir/corregir
    y_cop_test = copernicus.VHM0.values[ind_time:]  # Variable objetivo que queremos predecir/corregir

    # Modelo
    modelo_regresion = LinearRegression()

    modelo_regresion.fit(X_train, y_gow_train)
    beta_gow = modelo_regresion.coef_[0]
    y_cal_gow_train = (1 / beta_gow) * y_gow_train
    y_cal_gow_test = (1 / beta_gow) * y_gow_test

    modelo_regresion.fit(X_train, y_cop_train)
    beta_cop = modelo_regresion.coef_[0]
    y_cal_cop_train = (1 / beta_cop) * y_cop_train
    y_cal_cop_test = (1 / beta_cop) * y_cop_test

    # import matplotlib.pyplot as plt
    #
    # x = [i for i in range(len(y_gow))]
    # plt.plot(x, boya.hs.values, label='Boya')
    # plt.plot(x, y_gow, label='GOW')
    # plt.plot(x, y_cal_gow, label='Calibrada')
    # plt.legend()
    # plt.show()

    # Dibujar
    hs_max = 14  # int(max([boya.hs.max(), gow.hs.max(), copernicus.VHM0.max(), y_cal_gow.max(), y_cal_cop.max()])) + 1

    x_plot = np.linspace(0, hs_max, 11)

    y_gow_plot = beta_gow * x_plot
    modelo_regresion.fit(X_train, y_cal_gow_train)
    y_cal_gow_plot_train = modelo_regresion.predict(x_plot.reshape(-1, 1))  # (1 / beta_gow) * x_plot
    modelo_regresion.fit(X_test, y_cal_gow_test)
    y_cal_gow_plot_test = modelo_regresion.predict(x_plot.reshape(-1, 1))

    y_cop_plot = beta_cop * x_plot
    modelo_regresion.fit(X_train, y_cal_cop_train)
    y_cal_cop_plot_train = modelo_regresion.predict(x_plot.reshape(-1, 1))  # (1 / beta_cop) * x_plot
    modelo_regresion.fit(X_test, y_cal_cop_test)
    y_cal_cop_plot_test = modelo_regresion.predict(x_plot.reshape(-1, 1))  # (1 / beta_cop) * x_plot

    title = f'Modelo Lineal {nombre}: y_cal={1 / beta_gow:.2f}*Hs'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values,
                                                    X_train.ravel(), y_cal_gow_train, y_cal_gow_plot_train,
                                                    X_test.ravel(), y_cal_gow_test, y_cal_gow_plot_test,
                                                    y_gow_plot, hs_max,
                                                    'GOW', title, c='purple', fname=f'plot/model/01_Lineal/{nombre}_lineal_gow.png', plot=plot)

    title = f'Modelo Lineal {nombre}: y_cal={1 / beta_cop:.2f}*Hs'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values,
                                                    X_train.ravel(), y_cal_cop_train, y_cal_cop_plot_train,
                                                    X_test.ravel(), y_cal_cop_test, y_cal_cop_plot_test,
                                                    y_cop_plot, hs_max,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/01_Lineal/{nombre}_lineal_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
