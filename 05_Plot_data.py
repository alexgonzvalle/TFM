import pandas as pd
from data import get_data
from plot import plot_data
from scipy.io import savemat
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


def to_datenum(value_datetime):
    return 366 + value_datetime.toordinal() + (value_datetime - dt.datetime.fromordinal(value_datetime.toordinal())).total_seconds() / (24 * 60 * 60)


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


    ax_crontime.barh(nombre, gow.time.values[-1] - gow.time.values[0], left=gow.time.values[0], color='skyblue', edgecolor='black', alpha=0.8)
    ax_crontime.text(gow.time.values[-1], nombre, f'{len(gow.time.values)}', ha='right', va='center')
    ax_crontime.set_xlabel('Tiempo')
    ax_crontime.set_ylabel('Boyas')
    ax_crontime.set_title('Mediciones')
    ax_crontime.grid(True)
plt.show()
