import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
from data import get_data
from stats import stats

plot = True
df_boya = pd.read_csv('boyas.csv')
df_res = pd.read_csv('res.csv')

# MODELO LINEAL
for nombre in df_boya['Nombre'][:1]:
    boya, copernicus, gow = get_data(nombre)

    # Separar las variables predictoras (X) y la variable objetivo (y)
    # X = np.array([boya.hs.values, boya.tp.values, boya.dir.values]).T
    X = np.array([boya.hs.values, boya.dir.values]).T
    y_gow = gow.hs.values.reshape(-1, 1)  # Variable objetivo que queremos predecir/corregir
    y_cop = copernicus.VHM0.values.reshape(-1, 1)  # Variable objetivo que queremos predecir/corregir

    # Normalizar los datos
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_norm = scaler_x.fit_transform(X)
    y_norm_gow = scaler_y.fit_transform(y_gow)
    y_norm_cop = scaler_y.fit_transform(y_cop)

    # Definir la arquitectura de la red neuronal
    model = Sequential()
    model.add(Dense(128, input_shape=(X_norm.shape[1], ), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Salida: Hs

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Entrenar la red neuronal
    model.fit(X_norm, y_norm_gow, epochs=100, batch_size=32)
    X_norm_gow = scaler_x.fit_transform(np.array([gow.hs.values, gow.dir.values]).T)
    y_cal_gow = scaler_y.inverse_transform(model.predict(X_norm_gow)).ravel()

    # import matplotlib.pyplot as plt
    #
    # x = [i for i in range(len(y_gow))]
    # plt.plot(x, boya.hs.values, label='Boya')
    # plt.plot(x, y_gow, label='GOW')
    # plt.plot(x, y_cal_gow, label='Calibrada')
    # plt.legend()
    # plt.show()

    model.fit(X_norm, y_norm_cop, epochs=20, batch_size=32)
    X_norm_cop = scaler_x.fit_transform(np.array([copernicus.VHM0.values, copernicus.VMDR.values]).T)
    y_cal_cop = scaler_y.inverse_transform(model.predict(X_norm_cop)).ravel()

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
                                                    'GOW', title, c='purple', fname=f'plot/model/05_RedNeuronal/{nombre}_red_gow.png', plot=plot)

    title = f'Modelo Red Neuronal {nombre}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, y_cal_cop, hs_max, y_cop_plot, y_cal_cop_plot,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/05_RedNeuronal/{nombre}_red_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'Lineal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
# df_res.to_csv('res.csv', index=False)
