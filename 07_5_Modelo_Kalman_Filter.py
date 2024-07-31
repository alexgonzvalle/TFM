import numpy as np
import pandas as pd
from data import get_data
from stats import stats
from sklearn.linear_model import LinearRegression


class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = A  # Matrix de transición de estado
        self.B = B  # Matrix de control de entrada
        self.H = H  # Matrix de observación
        self.Q = Q  # Covarianza del ruido del proceso
        self.R = R  # Covarianza del ruido de observación
        self.x = x0  # Estado inicial
        self.P = P0  # Covarianza inicial del estado

    def predict(self, u=0):
        # Predicción del estado siguiente
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.x[1][self.x[1] > 360] -= 360
        self.x[1][self.x[1] < 0] += 360

        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Actualización con la medida z
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self):
        return self.x


plot = True
df_boya = pd.read_csv('boyas.csv')
df_res = pd.read_csv('res.csv')

for nombre in df_boya['Nombre']:
    boya, copernicus, gow = get_data(nombre)

    # Separar las variables predictoras (X) y la variable objetivo (y)
    # x_b = np.array([boya.hs.values, boya.tp.values, boya.dir.values]).T
    x = np.array([boya.hs.values, boya.dir.values]).T
    y_gow = gow.hs.values.reshape(-1, 1)  # Variable objetivo que queremos predecir/corregir
    y_cop = copernicus.VHM0.values.reshape(-1, 1)  # Variable objetivo que queremos predecir/corregir

    # Inicialización de los parámetros del filtro
    A = np.array([[1, 0], [0, 1]])  # Asumiendo un modelo simple donde los estados siguen siendo los mismos
    B = np.array([[0], [0]])  # No hay control de entrada
    H = np.array([[1, 0], [0, 1]])  # Observamos directamente ambos estados
    Q = np.array([[0.0001, 0], [0, 0.0001]])  # Suposición de una pequeña incertidumbre en el modelo
    R = np.array([[0.01, 0], [0, 0.01]])  # Suposición de una mayor incertidumbre en la medida
    x0 = np.array([[boya.hs.values[0]], [boya.dir.values[0]]])
    P0 = np.array([[1, 0], [0, 1]])  # Inicialización de la covarianza del estado

    # Aplicación del filtro de Kalman a los datos
    kf_gow = KalmanFilter(A, B, H, Q, R, x0, P0)
    kf_cop = KalmanFilter(A, B, H, Q, R, x0, P0)
    y_cal_gow, y_cal_cop = [], []
    for i in range(len(boya.hs.values)):
        kf_gow.predict()
        kf_gow.update(np.array([[y_gow[i]], [gow.dir.values.reshape(-1, 1)[i]]]))
        y_cal_gow.append(kf_gow.get_state()[0][0])

        kf_cop.predict()
        kf_cop.update(np.array([[y_cop[i]], [copernicus.VMDR.values.reshape(-1, 1)[i]]]))
        y_cal_cop.append(kf_cop.get_state()[0][0])

    y_cal_gow = np.array(y_cal_gow).ravel()
    y_cal_cop = np.array(y_cal_cop).ravel()

    # import matplotlib.pyplot as plt
    #
    # x = [i for i in range(len(y_gow))]
    # plt.plot(x, boya.hs.values, label='Boya')
    # plt.plot(x, y_gow, label='GOW')
    # plt.plot(x, y_cal_gow, label='Calibrada')
    # plt.legend()
    # plt.show()

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

    title = f'Modelo Kalman Filter {nombre}'
    bias_gow, rmse_gow, pearson_gow, si_gow = stats(boya.dir.values, boya.hs.values, gow.dir.values, gow.hs.values, y_cal_gow, hs_max, y_gow_plot, y_cal_gow_plot,
                                                    'GOW', title, c='purple', fname=f'plot/model/05_Kalman/{nombre}_kalman_gow.png', plot=plot)

    title = f'Modelo Kalman Filter {nombre}'
    bias_cop, rmse_cop, pearson_cop, si_cop = stats(boya.dir.values, boya.hs.values, copernicus.VMDR.values, copernicus.VHM0.values, y_cal_cop, hs_max, y_cop_plot, y_cal_cop_plot,
                                                    'Copernicus', title, c='orange', fname=f'plot/model/05_Kalman/{nombre}_kalman_cop.png', plot=plot)

    df_res.loc[len(df_res.index)] = [nombre, 'Kalman', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
