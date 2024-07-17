import xarray as xr


def get_data(name):
    boya = xr.open_dataset(f'data/processed/boya_{name}_Ext.nc')
    copernicus = xr.open_dataset(f'data/processed/IBI_REANALYSIS_WAV_005_006_{name}.nc')
    gow = xr.open_dataset(f'data/processed/GOW_CFS_{name}.nc')

    return boya, copernicus, gow
