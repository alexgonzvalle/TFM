import pandas as pd
from data import get_data
from plot import plot_data
from scipy.io import savemat


df_boya = pd.read_csv('boyas.csv')
for nombre in df_boya['Nombre']:
    boya, copernicus, gow = get_data(nombre)

    savemat(f'data/processed/boya_{nombre}.mat', {'hs': boya.hs.values, 'tp': boya.tp.values, 'dir': boya.dir.values, 'time': boya.time.values})
    savemat(f'data/processed/copernicus_{nombre}.mat', {'VHM0': copernicus.VHM0.values, 'tp': copernicus.VTPK.values, 'dir': copernicus.VMDR.values, 'time': copernicus.time.values})
    savemat(f'data/processed/gow_{nombre}.mat', {'hs': gow.hs.values, 'tp': gow.tp.values, 'dir': gow.dir.values, 'time': gow.time.values})

    # time_str = boya["time"].dt.strftime('%d/%m/%Y')
    # title = f'{nombre}. {time_str.values[0]} - {time_str.values[-1]} (N={len(time_str)})'
    # plot_data(boya, copernicus, gow, title, f'plot/data_procesed/{nombre}.png')
