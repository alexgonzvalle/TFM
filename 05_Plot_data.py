import pandas as pd
from data import get_data
from plot import plot_data


df_boya = pd.read_csv('boyas.csv')
for nombre in df_boya['Nombre']:
    boya, copernicus, gow = get_data(nombre)

    time_str = boya["time"].dt.strftime('%d/%m/%Y')
    title = f'{nombre}. {time_str.values[0]} - {time_str.values[-1]} (N={len(time_str)})'
    plot_data(boya, copernicus, gow, title, f'plot/data_procesed/{nombre}.png')
