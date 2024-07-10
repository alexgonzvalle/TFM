import xarray as xr
import matplotlib.pyplot as plt

name = 'Cabo_Begur'
boya = xr.open_dataset(f'data/processed/boya_{name}_Ext.nc')
copernicus = xr.open_dataset(f'data/processed/IBI_REANALYSIS_WAV_005_006_{name}.nc')
gow = xr.open_dataset(f'data/processed/GOW_CFS_{name}.nc')

diff_gow = boya['hs'] - gow['hs']
diff_copernicus = boya['hs'] - copernicus['VHM0']

fig, ax = plt.subplots(2, 3, figsize=(10, 5))

boya['hs'].plot(ax=ax[0, 0], label='Boya')
gow['hs'].plot(ax=ax[0, 1], color='darkorange', label='GOW')
boya['hs'].plot(ax=ax[0, 1], alpha=0.5, label='Boya')
copernicus['VHM0'].plot(ax=ax[0, 2], color='green', label='Copernicus')
boya['hs'].plot(ax=ax[0, 2], alpha=0.5, label='Boya')

ax[0, 1].legend()
ax[0, 2].legend()

diff_gow.plot(ax=ax[1, 1], color='darkorange', label='GOW')
diff_copernicus.plot(ax=ax[1, 2], color='green', label='Copernicus')

ax[0, 0].set_title('Boya')
ax[0, 1].set_title('GOW')
ax[0, 2].set_title('Copernicus')
ax[1, 1].set_title('Diff GOW')
ax[1, 2].set_title('Diff Copernicus')

y_max = max(boya['hs'].max(), gow['hs'].max(), copernicus['VHM0'].max()) + 0.5
y_lim = [0, y_max]
ax[0, 0].set_ylim(y_lim)
ax[0, 1].set_ylim(y_lim)
ax[0, 2].set_ylim(y_lim)

y_min = min(diff_gow.min(), diff_copernicus.min()) - 0.5
y_max = max(diff_gow.max(), diff_copernicus.max()) + 0.5
y_lim = [y_min, y_max]
ax[1, 1].set_ylim(y_lim)
ax[1, 2].set_ylim(y_lim)

ax[0, 0].grid(True)
ax[0, 1].grid(True)
ax[0, 2].grid(True)
ax[1, 1].grid(True)
ax[1, 2].grid(True)

ax[1, 0].axis('off')

plt.tight_layout()
plt.show()
