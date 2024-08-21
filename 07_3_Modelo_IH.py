import numpy as np
import pandas as pd
from data import get_data
from scipy.io import loadmat
from stats import stats

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
    X_train = np.array([boya.hs.values[ind_train], boya.dir.values[ind_train]]).T
    X_test = np.array([boya.hs.values[ind_test], boya.dir.values[ind_test]]).T

    # Obtener salida del modelo IH
    data_ih = loadmat(f'data/cal_IH/{nombre}_gow.mat')['dataGraph_gow'][0][0]
    y_cal_gow_train = data_ih['Hscal'].ravel()
    y_cal_gow_test = data_ih['Hscal_test'].ravel()

    data_ih = loadmat(f'data/cal_IH/{nombre}_cop.mat')['dataGraph_cop'][0][0]
    y_cal_cop_train = data_ih['Hscal'].ravel()
    y_cal_cop_test = data_ih['Hscal_test'].ravel()

    # import matplotlib.pyplot as plt
    #
    # x = [i for i in range(len(y_gow))]
    # plt.plot(x, boya.hs.values, label='Boya')
    # plt.plot(x, y_gow, label='GOW')
    # plt.plot(x, y_cal_gow, label='Calibrada')
    # plt.legend()
    # plt.show()

    # Dibujar
    title = f'Modelo IH {nombre}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values,
                                                    ind_train, y_cal_gow_train, ind_test, y_cal_gow_test,
                                                    'GOW', title, c='purple', plot=plot, fname=f'plot/model/03_IH/{nombre}_ih_gow.png')

    title = f'Modelo IH {nombre}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values,
                                                    ind_train, y_cal_cop_train, ind_test, y_cal_cop_test,
                                                    'IBI', title, c='orange', plot=plot, fname=f'plot/model/03_IH/{nombre}_ih_ibi.png')

    df_res.loc[len(df_res.index)] = [nombre, 'IH', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
