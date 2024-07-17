import pandas as pd
from data import get_data
from stats import calculate_stats


df_res = pd.DataFrame(columns=['Nombre', 'Modelo', 'bias_gow', 'bias_cop', 'rmse_gow', 'rmse_cop', 'pearson_gow', 'pearson_cop', 'si_gow', 'si_cop'])
df_res.to_csv('res.csv', index=False)

df_boya = pd.read_csv('boyas.csv')

# MODELO Polinomial
for nombre in df_boya['Nombre']:
    boya, copernicus, gow = get_data(nombre)
    bias_gow, rmse_gow, pearson_gow, si_gow = calculate_stats(boya.hs.values, gow.hs.values)
    bias_cop, rmse_cop, pearson_cop, si_cop = calculate_stats(boya.hs.values, copernicus.VHM0.values)

    df_res.loc[len(df_res.index)] = [nombre, 'Base', bias_gow, bias_cop, rmse_gow, rmse_cop, pearson_gow, pearson_cop, si_gow, si_cop]
df_res.to_csv('res.csv', index=False)
