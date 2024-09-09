import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from data import get_data
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
    ind_train = np.array([i for i in range(ind_time)])
    ind_test = np.array([i for i in range(ind_time, len(boya.time))])

    # Separar las variables predictoras (X) y la variable objetivo (y)
    X_gow_train = np.array([gow.hs.values[ind_train]]).T
    X_gow_test = np.array([gow.hs.values[ind_test]]).T

    X_cop_train = np.array([copernicus.VHM0.values[ind_train]]).T
    X_cop_test = np.array([copernicus.VHM0.values[ind_test]]).T

    # Variable objetivo que queremos predecir/corregir
    y_train = boya.hs.values[ind_train]
    y_test = boya.hs.values[ind_test]

    # Modelo
    modelo_regresion = LinearRegression()

    modelo_regresion.fit(X_gow_train, y_train)
    beta_gow = modelo_regresion.coef_[0]
    y_cal_gow_train = np.squeeze(beta_gow * X_gow_train)
    y_cal_gow_test = np.squeeze(beta_gow * X_gow_test)

    modelo_regresion.fit(X_cop_train, y_train)
    beta_cop = modelo_regresion.coef_[0]
    y_cal_cop_train = np.squeeze(beta_cop * X_cop_train)
    y_cal_cop_test = np.squeeze(beta_cop * X_cop_test)

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
    title = f'Modelo Lineal {nombre}: {aux_title}={beta_gow:.2f}*Hs'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values,
                                                    ind_train, y_cal_gow_train, ind_test, y_cal_gow_test,
                                                    'GOW', title, c='purple', plot=plot, fname=f'plot/model/01_Lineal/{nombre}_lineal_gow.png')

    title = f'Modelo Lineal {nombre}: {aux_title}={beta_cop:.2f}*Hs'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values,
                                                    ind_train, y_cal_cop_train, ind_test, y_cal_cop_test,
                                                    'IBI', title, c='orange', plot=plot, fname=f'plot/model/01_Lineal/{nombre}_lineal_ibi.png')

    df_res.loc[len(df_res.index)] = [nombre, 'Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
# df_res.to_csv('res.csv', index=False)
