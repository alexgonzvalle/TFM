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

    # Separar las variables predictoras (X) y la variable objetivo (y)
    # X = np.array([boya.hs.values, boya.tp.values, boya.dir.values]).T
    X = boya.hs.values
    y_gow = gow.hs.values  # Variable objetivo que queremos predecir/corregir
    y_cop = copernicus.VHM0.values  # Variable objetivo que queremos predecir/corregir

    # Modelo
    params, params_covariance = curve_fit(model_function, X, gow.hs.values)
    beta_gow, gamma_gow = params
    hs_cal_gow = ((1 / beta_gow) ** (1 / gamma_gow)) * y_gow ** (1 / gamma_gow)

    params, params_covariance = curve_fit(model_function, X, y_cop, p0=[1, 1])
    beta_cop, gamma_cop = params
    hs_cal_cop = ((1 / beta_cop) ** (1 / gamma_cop)) * y_gow ** (1 / gamma_cop)

    # Dibujar
    hs_max = int(max([boya.hs.max(), gow.hs.max(), copernicus.VHM0.max(), hs_cal_gow.max(), hs_cal_cop.max()])) + 1

    x_plot = np.linspace(0, hs_max, 11)
    y_raw_gow = beta_gow * x_plot ** gamma_gow
    y_cal_gow = ((1 / beta_gow) ** (1 / gamma_gow)) * x_plot ** (1 / gamma_gow)

    y_raw_cop = beta_cop * x_plot ** gamma_cop
    y_cal_cop = ((1 / beta_cop) ** (1 / gamma_cop)) * x_plot ** (1 / gamma_cop)

    title = f'Modelo {nombre}: y_cal = {((1 / beta_gow) ** (1 / gamma_gow)):.2f}*Hs^{(1 / gamma_gow):.2f}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values, hs_cal_gow, hs_max, y_raw_gow, y_cal_gow,
                                                    'GOW', title, c='purple', fname=f'plot/model/02_NoLineal/{nombre}_noLineal_gow.png', plot=plot)

    title = f'Modelo {nombre}: y_cal = {((1 / beta_cop) ** (1 / gamma_cop)):.2f}*Hs^{(1 / gamma_cop):.2f}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, hs_cal_cop, hs_max, y_raw_cop, y_cal_cop,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/02_NoLineal/{nombre}_noLineal_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'No Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
