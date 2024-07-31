import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from data import get_data
from stats import stats


def cost_function(x_flat, x_b, B_inv, y, R_inv, H):
    x = x_flat.reshape(x_b.shape)

    cost = 0
    for i in range(len(x_b)):
        x_i = x[i, :]
        y_i = y[i, :]
        term1 = 0.5 * np.dot((x_i - x_b[i, :]).T, np.dot(B_inv, (x_i - x_b[i, :])))
        term2 = 0.5 * np.dot((H(x_i) - y_i).T, np.dot(R_inv, (H(x_i) - y_i)))
        cost += term1 + term2
    return cost


def jacobian(x_flat, x_b, B_inv, y, R_inv, H):
    n_obs = len(x_b)
    x = x_flat.reshape((n_obs, -1))
    jac = np.zeros_like(x)
    for i in range(n_obs):
        x_i = x[i, :]
        y_i = y[i, :]
        jac[i, :] = np.dot(B_inv, (x_i - x_b[i, :])) + np.dot(R_inv, (H(x_i) - y_i))
    return jac.flatten()


def H(x):
    return np.array([x[0]])


plot = False
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

    title = f'Modelo 3DVar {nombre}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values, y_cal_gow, hs_max, y_gow_plot, y_cal_gow_plot,
                                                    'GOW', title, c='purple', fname=f'plot/model/03_3DVar/{nombre}_3dvar_gow.png', plot=plot)

    title = f'Modelo 3DVar {nombre}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, y_cal_cop, hs_max, y_cop_plot, y_cal_cop_plot,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/03_3DVar/{nombre}_3dvar_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, '3DVar', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
