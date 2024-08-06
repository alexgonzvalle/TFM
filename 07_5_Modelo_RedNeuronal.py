import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
from data import get_data
from stats import stats


def create_sequences(value_in, seq_length):
    dim_var = value_in.shape[1]
    value_in = np.vstack((np.zeros((seq_length, dim_var)), value_in))
    value_out = []

    for i in range(len(value_in) - seq_length):
        if dim_var == 1:
            value_out.append(value_in[i + seq_length])
        else:
            value_out.append(value_in[i:i + seq_length])

    return np.array(value_out)


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

    # Separar las variables predictoras (X) y la variable objetivo (y)
    X_train = np.array([boya.hs.values[:ind_time], boya.dir.values[:ind_time]]).T
    X_test = np.array([boya.hs.values[ind_time:], boya.dir.values[ind_time:]]).T

    # Variable objetivo que queremos predecir/corregir
    y_gow_train = gow.hs.values[:ind_time]
    y_gow_test = gow.hs.values[ind_time:]

    y_cop_train = copernicus.VHM0.values[:ind_time]
    y_cop_test = copernicus.VHM0.values[ind_time:]

    # Normalizar los datos
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_norm = scaler_x.fit_transform(X_train)
    y_norm_gow_train = scaler_y.fit_transform(y_gow_train)
    y_norm_cop_train = scaler_y.fit_transform(y_cop_train)

    # Crear secuencia
    # seq_length = 24  # Longitud de la secuencia
    # X_norm = create_sequences(X_norm, seq_length)
    # y_norm_gow = create_sequences(y_norm_gow, seq_length)
    # y_norm_cop = create_sequences(y_norm_cop, seq_length)

    # Definir la arquitectura de la red neuronal
    model = Sequential()
    # model.add(LSTM(128, return_sequences=True, input_shape=(seq_length, X_norm.shape[2])))
    # model.add(LSTM(64))
    model.add(Dense(units=10, activation='sigmoid', input_dim=X_train_norm.shape[1]))
    model.add(Dense(units=6, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))  # Salida: Hs

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Entrenar la red neuronal
    history = model.fit(X_train_norm, y_norm_gow_train, epochs=100)

    X_norm_gow = scaler_x.fit_transform(np.array([gow.hs.values, gow.dir.values]).T)
    # X_norm_gow = create_sequences(X_norm_gow, seq_length)
    y_cal_gow = scaler_y.inverse_transform(model.predict(X_norm_gow)).ravel()

    # import matplotlib.pyplot as plt
    #
    # x = [i for i in range(len(y_gow))]
    # plt.plot(x, boya.hs.values, label='Boya')
    # plt.plot(x, y_gow, label='GOW')
    # plt.plot(x, y_cal_gow, label='Calibrada')
    # plt.legend()
    # plt.show()

    model.fit(X_train_norm, y_norm_cop_train, epochs=100)
    X_norm_cop = scaler_x.fit_transform(np.array([copernicus.VHM0.values, copernicus.VMDR.values]).T)
    # X_norm_cop = create_sequences(X_norm_cop, seq_length)
    y_cal_cop = scaler_y.inverse_transform(model.predict(X_norm_cop)).ravel()

    # Dibujar
    hs_max = 14  # int(max([boya.hs.max(), gow.hs.max(), copernicus.VHM0.max(), y_cal_gow.max(), y_cal_cop.max()])) + 1

    x_plot = np.linspace(0, hs_max, 11)

    modelo_regresion = LinearRegression()

    modelo_regresion.fit(X_train[:, 0].reshape(-1, 1), y_gow_train)
    y_gow_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    modelo_regresion.fit(X_train[:, 0].reshape(-1, 1), y_cal_gow)
    y_cal_gow_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    modelo_regresion.fit(X_train[:, 0].reshape(-1, 1), y_cop_train)
    y_cop_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    modelo_regresion.fit(X_train[:, 0].reshape(-1, 1), y_cal_cop)
    y_cal_cop_plot = modelo_regresion.predict(x_plot.reshape(-1, 1))

    title = f'Modelo Red Neuronal {nombre}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values, y_cal_gow, hs_max, y_gow_plot, y_cal_gow_plot,
                                                    'GOW', title, c='purple', fname=f'plot/model/05_RedNeuronal/{nombre}_red_gow.png', plot=plot)

    title = f'Modelo Red Neuronal {nombre}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, y_cal_cop, hs_max, y_cop_plot, y_cal_cop_plot,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/05_RedNeuronal/{nombre}_red_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'Red_Neuronal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
# df_res.to_csv('res.csv', index=False)
