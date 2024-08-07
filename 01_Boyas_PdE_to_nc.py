import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

# Load the data
df = pd.read_csv('data/raw/boyas/PdE/17418_28069_2136_ALL_19901107082707_20240710072707.csv', sep='\t', header=1)
name = 'Bilbao-Vizcaya'

# Tiempo
time = np.array([dt.datetime.strptime(t, '%Y %m %d %H') for t in df['Fecha (GMT)'].tolist()])

# Hs
hs = np.array(df['Altura Signif. del Oleaje(m)'].tolist())
hs[hs == -9999.9] = np.nan
# Crear dataset de xr
ds_hs = xr.Dataset({'hs': ('time', hs)},
                   coords={'time': time}
                   )

# Tm02
tm02 = np.array(df['Periodo Medio Tm02(s)'].tolist())
tm02[tm02 == -9999.9] = np.nan
ds_tm02 = xr.Dataset({'t02': ('time', tm02)},
                     coords={'time': time}
                     )

# Tp
tp = np.array(df['Periodo de Pico(s)'].tolist())
tp[tp == -9999.9] = np.nan
ds_tp = xr.Dataset({'tp': ('time', tp)},
                   coords={'time': time}
                   )

# Dir_m
dm = np.array(df['Direcc. Media de Proced.(0=N,90=E)'].tolist())
dm[dm == -9999.9] = np.nan
ds_dm = xr.Dataset({'dir': ('time', dm)},
                   coords={'time': time}
                   )

# Crear dataset final
ds = xr.Dataset()
ds = ds.merge(ds_hs)
ds = ds.merge(ds_tm02)
ds = ds.merge(ds_tp)
ds = ds.merge(ds_dm)

# ds['time'].attrs = dict({'standard_name': 'time',
#                          'units': 'days since 1900-01-01 00:00:00',
#                          'calendar': 'standard'})

ds['hs'].attrs = dict({'standard_name': 'wave_significant_height',
                       'long_name': 'Wave significant height',
                       'units': 'm'})

ds['t02'].attrs = dict({'standard_name': 'wave_mean_period',
                        'long_name': 'Wave mean period',
                        'units': 's'})

ds['tp'].attrs = dict({'standard_name': 'peak_wave_period',
                       'long_name': 'Peak wave period',
                       'units': 's'})

ds['dir'].attrs = dict({'standard_name': 'wave_mean_direction',
                        'long_name': 'Wave mean direction',
                        'units': 'degrees',
                        'units_long_name': 'degrees_from_north_0 º-east_90 º'})

ds.to_netcdf(f'data/raw/boyas/PdE/{name}_Ext.nc')

# # Plot
# ds_old = xr.open_dataset(f'data/raw/boyas/{name}_Ext.nc')
# fig, ax = plt.subplots()
# ds['hs'].plot(ax=ax, label='PdE')
# ds_old['hs'].plot(ax=ax, label='IH')
# plt.legend()
# plt.show()
