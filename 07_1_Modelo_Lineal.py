import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from data import get_data
from stats import stats


df_boya = pd.read_csv('boyas.csv')
df_res = pd.read_csv('res.csv')

# MODELO LINEAL
for nombre in df_boya['Nombre']:
    boya, copernicus, gow = get_data(nombre)

    # Separar las variables predictoras (X) y la variable objetivo (y)
    # X = np.array([boya.hs, boya.tp, boya.dir]).T
    X = np.array([boya.hs.values]).T
    y_gow = gow.hs.values  # Variable objetivo que queremos predecir/corregir
    y_cop = copernicus.VHM0.values  # Variable objetivo que queremos predecir/corregir

    # Modelo
    modelo_regresion = LinearRegression()

    modelo_regresion.fit(X, y_gow)
    coef_gow, intercept_gow = modelo_regresion.coef_, modelo_regresion.intercept_
    hs_cal_gow = (1 / coef_gow[0]) * y_gow + intercept_gow

    modelo_regresion.fit(X, y_cop)
    coef_cop, intercept_cop = modelo_regresion.coef_, modelo_regresion.intercept_
    hs_cal_cop = (1 / coef_cop[0]) * y_cop + intercept_cop

    # Dibujar
    hs_max = int(max([boya.hs.max(), gow.hs.max(), copernicus.VHM0.max(), hs_cal_gow.max(), hs_cal_cop.max()])) + 1

    x_plot = np.linspace(0, hs_max, 11)
    y_raw_gow = coef_gow[0] * x_plot + intercept_gow
    y_cal_gow = (1 / coef_gow[0]) * x_plot + intercept_gow

    y_raw_cop = coef_cop[0] * x_plot + intercept_cop
    y_cal_cop = (1 / coef_cop[0]) * x_plot + intercept_cop

    # title = f'Modelo Lineal {nombre}: y={coef_gow[0]:.2f}*Hs + {coef_gow[1]:.2f}*Tp + {coef_gow[2]:.2f}*Dire + {intercept_gow:.2f}'
    title = f'Modelo Lineal {nombre}: y_cal={1 / coef_gow[0]:.2f}*Hs + {intercept_gow:.2f}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values, hs_cal_gow, hs_max, y_raw_gow, y_cal_gow,
                                                    'GOW', title, c='purple', fname=f'plot/model/lineal/{nombre}_lineal_gow.png')

    # title = f'Modelo Lineal {nombre}: y={coef_cop[0]:.2f}*Hs + {coef_cop[1]:.2f}*Tp + {coef_cop[2]:.2f}*Dire + {intercept_cop:.2f}'
    title = f'Modelo Lineal {nombre}: y_cal={1 / coef_cop[0]:.2f}*Hs + {intercept_cop:.2f}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, hs_cal_cop, hs_max, y_raw_cop, y_cal_cop,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/lineal/{nombre}_lineal_cop.png')

    df_res.loc[len(df_res.index)] = [nombre, 'Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
