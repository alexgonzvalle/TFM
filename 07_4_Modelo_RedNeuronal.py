import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
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
    y_gow = gow.hs.values.reshape(-1, 1)  # Variable objetivo que queremos predecir/corregir
    y_cop = copernicus.VHM0.values.reshape(-1, 1)  # Variable objetivo que queremos predecir/corregir

    # Normalizar los datos
    scaler_x = StandardScaler()
    X_norm = scaler_x.fit_transform(X)

    scaler_y = StandardScaler()
    y_norm_gow = scaler_y.fit_transform(y_gow)
    y_norm_cop = scaler_y.fit_transform(y_cop)

    # Definir la arquitectura de la red neuronal
    model = Sequential([
        LSTM(64, input_shape=(X_norm.shape[1], 1)),
        Dense(1)
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='mse')

    # Entrenar la red neuronal
    model.fit(X_norm, y_norm_gow, epochs=20)
    y_cal_gow = scaler_y.inverse_transform(model.predict(X_norm)).ravel()

    model.fit(X_norm, y_norm_cop, epochs=20)
    y_cal_cop = scaler_y.inverse_transform(model.predict(X_norm)).ravel()

    # Dibujar
    hs_max = 14  # int(max([boya.hs.max(), gow.hs.max(), copernicus.VHM0.max(), y_cal_gow.max(), y_cal_cop.max()])) + 1

    x_plot = np.linspace(0, hs_max, 11)

    modelo_regresion = LinearRegression()

    modelo_regresion.fit(X[:, 0].reshape(-1, 1), y_gow)
    y_gow_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    modelo_regresion.fit(X[:, 0].reshape(-1, 1), y_cal_gow)
    y_cal_gow_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    modelo_regresion.fit(X[:, 0].reshape(-1, 1), y_cop)
    y_cop_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    modelo_regresion.fit(X[:, 0].reshape(-1, 1), y_cal_cop)
    y_cal_cop_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    title = f'Modelo Red Neuronal {nombre}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values, y_cal_gow, hs_max, y_gow_plot, y_cal_gow_plot,
                                                    'GOW', title, c='purple', fname=f'plot/model/03_MultiLineal/{nombre}_multilineal_gow.png', plot=plot)

    title = f'Modelo Red Neuronal {nombre}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, y_cal_cop, hs_max, y_cop_plot, y_cal_cop_plot,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/03_MultiLineal/{nombre}_multilineal_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
