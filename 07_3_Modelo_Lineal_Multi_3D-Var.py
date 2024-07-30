import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from data import get_data
from stats import stats
import dask.array as da


def cost_function(x_flat, x_b, B_inv, y, R_inv, H):
    n_obs = len(x_b)
    x = x_flat.reshape((n_obs, -1))

    x_b_dask = da.from_array(x_b, chunks=(1000, 3))
    y_dask = da.from_array(y, chunks=(1000, 1))

    cost = da.sum([
        0.5 * da.dot((x[i, :] - x_b_dask[i, :]).T, da.dot(B_inv, (x[i, :] - x_b_dask[i, :]))) +
        0.5 * da.dot((H(x[i, :]) - y_dask[i, :]).T, da.dot(R_inv, (H(x[i, :]) - y_dask[i, :])))
        for i in range(n_obs)
    ])
    return cost.compute()


def jacobian(x_flat, x_b, B_inv, y, R_inv, H):
    n_obs = len(x_b)
    x = x_flat.reshape((n_obs, -1))

    x_b_dask = da.from_array(x_b, chunks=(1000, 3))
    y_dask = da.from_array(y, chunks=(1000, 1))

    jac = da.concatenate([
        da.dot(B_inv, (x[i, :] - x_b_dask[i, :])) + da.dot(R_inv, (H(x[i, :]) - y_dask[i, :]))
        for i in range(n_obs)
    ])
    return jac.compute().flatten()


def H(x):
    # Supongamos que x contiene Hs, Tp y Dir en ese orden
    Hs_model = x[0]  # Ajusta esto según el modelo real
    return np.array([Hs_model])


plot = True
df_boya = pd.read_csv('boyas.csv')
df_res = pd.read_csv('res.csv')

# MODELO LINEAL
for nombre in df_boya['Nombre']:
    boya, copernicus, gow = get_data(nombre)

    # Separar las variables predictoras (X) y la variable objetivo (y)
    # x = np.array([boya.hs.values, boya.tp.values, boya.dir.values]).T
    x = np.array([boya.hs.values, boya.dir.values]).T
    y_gow = gow.hs.values.reshape(-1, 1)  # Variable objetivo que queremos predecir/corregir
    y_cop = copernicus.VHM0.values.reshape(-1, 1)  # Variable objetivo que queremos predecir/corregir

    # Matriz de covarianza del error del estado de referencia
    B = np.eye(x.shape[1])  # Ejemplo simple con identidad
    B_inv = np.linalg.inv(B)

    # Matriz de covarianza del error de las observaciones
    R_gow = np.eye(1)  # Ejemplo simple con identidad
    R_gow_inv = np.linalg.inv(R_gow)

    # Minimizar la función de costo
    res_gow = minimize(cost_function, x.flatten(), args=(x, B_inv, y_gow, R_gow_inv, H), method='L-BFGS-B', jac=jacobian, options={'disp': True})
    y_cal_gow = res_gow.x.reshape(x.shape)[:, 0]

    res_cop = minimize(cost_function, x.flatten(), args=(x, B_inv, y_cop, R_gow_inv, H), method='L-BFGS-B', jac=jacobian, options={'disp': True})
    y_cal_cop = res_cop.x.reshape(x.shape)[:, 0]

    # Dibujar
    hs_max = 14  # int(max([boya.hs.max(), gow.hs.max(), copernicus.VHM0.max(), y_cal_gow.max(), y_cal_cop.max()])) + 1
    tp_max = int(max([boya.tp.max(), gow.tp.max(), copernicus.VTPK.max()])) + 1

    x_plot = np.linspace(0, hs_max, 11)

    modelo_regresion = LinearRegression()

    modelo_regresion.fit(x[:, 0].reshape(-1, 1), y_gow)
    y_gow_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    modelo_regresion.fit(x[:, 0].reshape(-1, 1), y_cal_gow)
    y_cal_gow_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    modelo_regresion.fit(x[:, 0].reshape(-1, 1), y_cop)
    y_cop_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    modelo_regresion.fit(x[:, 0].reshape(-1, 1), y_cal_cop)
    y_cal_cop_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    title = f'Modelo Multi Lineal {nombre}: y=a*Hs + b*Tp + c*Dire'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values, y_cal_gow, hs_max, y_gow_plot, y_cal_gow_plot,
                                                    'GOW', title, c='purple', fname=f'plot/model/03_MultiLineal/{nombre}_multilineal_gow.png', plot=plot)

    title = f'Modelo Multi Lineal {nombre}: y=a*Hs + b*Tp + c*Dire'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, y_cal_cop, hs_max, y_cop_plot, y_cal_cop_plot,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/03_MultiLineal/{nombre}_multilineal_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
