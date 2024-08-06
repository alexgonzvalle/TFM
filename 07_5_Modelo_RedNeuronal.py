import numpy as np
import keras
from sklearn.preprocessing import StandardScaler
import pandas as pd
import itertools
from data import get_data
from stats import stats


# Función para crear el modelo de Keras
def create_model_reg(xtrain_scaled, optimizer='sgd', learning_rate=0.01, n_layers=1, n_neurons=10):
    model = keras.Sequential()

    for i in range(n_layers):
        if i == 0:
            model.add(keras.layers.Dense(n_neurons, input_dim=xtrain_scaled.shape[1], activation='relu'))
        else:
            model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(1))

    opt = None
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model


def get_best_model(X_train_norm, y_train, X_test_norm, y_test):
    params = {
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'learning_rate': [0.001, 0.01, 0.1],
        'n_layers': [1, 2, 3],
        'n_neurons': [20, 50, 100]
    }
    param_combinations = list(itertools.product(*params.values()))

    best_mse, best_params, model_reg = 100, None, None
    for combination in param_combinations:
        param_dict = dict(zip(params.keys(), combination))
        print(param_dict)

        model_reg_prop = create_model_reg(X_train_norm, **param_dict)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        history = model_reg_prop.fit(X_train_norm, y_train, epochs=100, batch_size=32, verbose=0,
                                     validation_split=0.2, callbacks=[early_stop])

        m_eval = model_reg_prop.evaluate(X_test_norm, y_test, batch_size=64)
        mse = m_eval[1]

        if mse < best_mse:
            best_mse = mse
            best_params = param_dict
            model_reg = model_reg_prop

    return best_mse, best_params, model_reg


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
    ind_train = np.array([i for i in range(ind_time)])
    ind_test = np.array([i for i in range(ind_time, len(boya.time))])

    # Separar las variables predictoras (X) y la variable objetivo (y)
    X_gow_train = np.array([gow.hs.values[ind_train], np.deg2rad(gow.dir.values[ind_train])]).T
    X_gow_test = np.array([gow.hs.values[ind_test], np.deg2rad(gow.dir.values[ind_test])]).T

    X_cop_train = np.array([copernicus.VHM0.values[ind_train], np.deg2rad(copernicus.VMDR.values[ind_train])]).T
    X_cop_test = np.array([copernicus.VHM0.values[ind_test], np.deg2rad(copernicus.VMDR.values[ind_test])]).T

    # Variable objetivo que queremos predecir/corregir
    y_train = boya.hs.values[ind_train].reshape(-1, 1)
    y_test = boya.hs.values[ind_test].reshape(-1, 1)

    # Normalizar los datos
    scaler = StandardScaler()

    X_gow_train_norm = scaler.fit_transform(X_gow_train)
    X_gow_test_norm = scaler.fit_transform(X_gow_test)
    X_cop_train_norm = scaler.fit_transform(X_cop_train)
    X_cop_test_norm = scaler.fit_transform(X_cop_test)

    # Encontrar los mejores hiperparámetros
    _, best_params_gow, model_reg = get_best_model(X_gow_train_norm, y_train, X_gow_test_norm, y_test)
    y_cal_gow_train = model_reg.predict(X_gow_train_norm).ravel()
    y_cal_gow_test = model_reg.predict(X_gow_test_norm).ravel()

    # import matplotlib.pyplot as plt
    #
    # x = [i for i in range(len(y_gow))]
    # plt.plot(x, boya.hs.values, label='Boya')
    # plt.plot(x, y_gow, label='GOW')
    # plt.plot(x, y_cal_gow, label='Calibrada')
    # plt.legend()
    # plt.show()

    _, best_params_cop, model_reg = get_best_model(X_cop_train_norm, y_train, X_gow_test_norm, y_test)
    y_cal_cop_train = model_reg.predict(X_cop_train_norm).ravel()
    y_cal_cop_test = model_reg.predict(X_cop_test_norm).ravel()

    # Dibujar
    hs_max = 14  # int(max([boya.hs.max(), gow.hs.max(), copernicus.VHM0.max(), y_cal_gow.max(), y_cal_cop.max()])) + 1
    y_cal_gow_plot_train = None
    y_cal_gow_plot_test = None
    y_gow_plot = None
    y_cal_cop_plot_train = None
    y_cal_cop_plot_test = None
    y_cop_plot = None

    title = f'Modelo Red Neuronal {nombre}. {best_params_gow}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values,
                                                    ind_train, y_cal_gow_train, y_cal_gow_plot_train,
                                                    ind_test, y_cal_gow_test, y_cal_gow_plot_test,
                                                    y_gow_plot, hs_max,
                                                    'GOW', title, c='purple', fname=f'plot/model/05_RedNeuronal/{nombre}_red_gow.png', plot=plot)

    title = f'Modelo Red Neuronal {nombre}. {best_params_cop}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values,
                                                    ind_train, y_cal_cop_train, y_cal_cop_plot_train,
                                                    ind_test, y_cal_cop_test, y_cal_cop_plot_test,
                                                    y_cop_plot, hs_max,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/05_RedNeuronal/{nombre}_red_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'Red_Neuronal', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
