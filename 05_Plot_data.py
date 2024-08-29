import pandas as pd
from data import get_data
from plot import plot_data
from scipy.io import savemat
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


def to_datenum(value_datetime):
    return 366 + value_datetime.toordinal() + (value_datetime - dt.datetime.fromordinal(value_datetime.toordinal())).total_seconds() / (24 * 60 * 60)


lbl_size = 14
goal_time = np.datetime64('2013-01-01T00:00:00.000000')
fig, ax_crontime = plt.subplots()

df_boya = pd.read_csv('boyas.csv')
for nombre in df_boya['Nombre']:
    boya, copernicus, gow = get_data(nombre)

    # time = np.array([to_datenum(dt.datetime.utcfromtimestamp(t.tolist()/1e9)) for t in boya.time.values])
    # savemat(f'data/processed/boya_{nombre}.mat', {'hs': boya.hs.values, 't02': boya.t02.values, 'tp': boya.tp.values, 'dir': boya.dir.values, 'time': time})
    # savemat(f'data/processed/copernicus_{nombre}.mat', {'hs': copernicus.VHM0.values, 't02': copernicus.VTM02.values, 'tp': copernicus.VTPK.values, 'dir': copernicus.VMDR.values, 'time': time})
    # savemat(f'data/processed/gow_{nombre}.mat', {'hs': gow.hs.values, 't02': gow.t02.values, 'tp': gow.tp.values, 'dir': gow.dir.values, 'time': time})

    # time_str = boya["time"].dt.strftime('%d/%m/%Y')
    # title = f'{nombre}. {time_str.values[0]} - {time_str.values[-1]} (N={len(time_str)})'
    # plot_data(boya, copernicus, gow, title, f'plot/data_procesed/{nombre}.png')

    # ax_crontime.barh(nombre, copernicus.time.values[-1] - copernicus.time.values[0], left=copernicus.time.values[0], color='skyblue', edgecolor='black', alpha=0.8)
    # ax_crontime.text(copernicus.time.values[-1], nombre, f'{len(copernicus.time.values)}', ha='right', va='center', fontsize=lbl_size)

    # Separar tiempos a la izquierda y derecha de goal_time
    left_times = copernicus.time.values[copernicus.time.values < goal_time]
    right_times = copernicus.time.values[copernicus.time.values >= goal_time]

    # Dibujar la parte izquierda de la barra
    if len(left_times) > 0:
        ax_crontime.barh(nombre, left_times[-1] - left_times[0],
                         left=left_times[0], color='skyblue',
                         edgecolor='black', alpha=0.8)
        ax_crontime.text(left_times[-1], nombre, f'{len(left_times)*100 / len(copernicus.time.values):.2f}%',
                         ha='right', va='center', fontsize=lbl_size, color='black')

    # Dibujar la parte derecha de la barra
    if len(right_times) > 0:
        ax_crontime.barh(nombre, right_times[-1] - right_times[0],
                         left=right_times[0], color='darkblue',
                         edgecolor='black', alpha=0.8)
        ax_crontime.text(right_times[-1], nombre, f'{len(right_times)*100 / len(copernicus.time.values):.2f}%',
                         ha='right', va='center', fontsize=lbl_size, color='black')

ax_crontime.axvline(goal_time, color='red', linestyle='--', linewidth=2, label='Goal Time')
ax_crontime.set_xlabel('Tiempo', fontsize=lbl_size)
ax_crontime.set_ylabel('Boyas', fontsize=lbl_size)
ax_crontime.set_title('Mediciones', fontsize=lbl_size)
ax_crontime.grid(True)
plt.show()
