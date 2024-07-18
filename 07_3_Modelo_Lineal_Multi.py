import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from data import get_data
from stats import stats


plot = False
df_boya = pd.read_csv('boyas.csv')
df_res = pd.read_csv('res.csv')

# MODELO LINEAL
for nombre in df_boya['Nombre']:
    boya, copernicus, gow = get_data(nombre)

    # Separar las variables predictoras (X) y la variable objetivo (y)
    X = np.array([boya.hs.values, boya.tp.values, boya.dir.values]).T
    y_gow = gow.hs.values  # Variable objetivo que queremos predecir/corregir
    y_cop = copernicus.VHM0.values  # Variable objetivo que queremos predecir/corregir

    # Modelo
    modelo_regresion = LinearRegression()

    modelo_regresion.fit(X, y_gow)
    coef_gow = modelo_regresion.coef_
    y_cal_gow = (1 / coef_gow[0]) * y_gow + (1 / coef_gow[1]) * y_gow + (1 / coef_gow[2]) * y_gow

    modelo_regresion.fit(X, y_cop)
    coef_cop = modelo_regresion.coef_
    y_cal_cop = (1 / coef_cop[0]) * y_cop + (1 / coef_cop[1]) * y_cop + (1 / coef_cop[2]) * y_cop

    # Dibujar
    hs_max = int(max([boya.hs.max(), gow.hs.max(), copernicus.VHM0.max(), y_cal_gow.max(), y_cal_cop.max()])) + 1

    title = f'Modelo Lineal {nombre}: y={1/coef_gow[0]:.2f}*Hs + {1/coef_gow[1]:.2f}*Tp + {1/coef_gow[2]:.2f}*Dire'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values, y_cal_gow, hs_max, None, None,
                                                    'GOW', title, c='purple', fname=f'plot/model/lineal/{nombre}_lineal_gow.png', plot=plot)

    title = f'Modelo Lineal {nombre}: y={1/coef_cop[0]:.2f}*Hs + {1/coef_cop[1]:.2f}*Tp + {1/coef_cop[2]:.2f}*Dire'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, y_cal_cop, hs_max, None, None,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/lineal/{nombre}_lineal_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
