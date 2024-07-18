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
    tp_max = int(max([boya.tp.max(), gow.tp.max(), copernicus.VTPK.max()])) + 1

    x1_plot = np.linspace(0, hs_max, 11)
    x2_plot = np.linspace(0, tp_max, 11)
    x3_plot = np.linspace(0, 360, 7)
    y_gow_plot = coef_gow[0] * x1_plot + coef_gow[1] * x2_plot + coef_gow[2] * x2_plot
    y_cal_gow_plot = (1 / coef_gow[0]) * x1_plot

    y_cop_plot = coef_cop[0] * x1_plot + coef_cop[1] * x2_plot + coef_cop[2] * x2_plot
    y_cal_cop_plot = (1 / coef_cop[0]) * x1_plot

    title = f'Modelo Lineal {nombre}: y={1/coef_gow[0]:.2f}*Hs + {1/coef_gow[1]:.2f}*Tp + {1/coef_gow[2]:.2f}*Dire'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values, y_cal_gow, hs_max, y_gow_plot, y_cal_gow_plot,
                                                    'GOW', title, c='purple', fname=f'plot/model/03_MultiLineal/{nombre}_multilineal_gow.png', plot=plot)

    title = f'Modelo Lineal {nombre}: y={1/coef_cop[0]:.2f}*Hs + {1/coef_cop[1]:.2f}*Tp + {1/coef_cop[2]:.2f}*Dire'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, y_cal_cop, hs_max, y_cop_plot, y_cal_cop_plot,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/03_MultiLineal/{nombre}_multilineal_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
