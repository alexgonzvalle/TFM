import pandas as pd
import numpy as np
from plot import plot_stats_comp

df_res = pd.read_csv('res.csv')

models = df_res['Modelo'].unique()
n_models = len(models)

for nombre in df_res['Nombre'].unique():
    df_res_nombre = df_res[df_res['Nombre'] == nombre]

    bias_gow, bias_cop = np.zeros((n_models, 1)), np.zeros((n_models, 1))
    rmse_gow, rmse_cop = np.zeros((n_models, 1)), np.zeros((n_models, 1))
    p_gow, p_cop = np.zeros((n_models, 1)), np.zeros((n_models, 1))
    si_gow, si_cop = np.zeros((n_models, 1)), np.zeros((n_models, 1))
    for i, modelo in enumerate(models):
        df_res_model = df_res_nombre[df_res_nombre['Modelo'] == modelo]
        bias_gow[i] = df_res_model['bias_gow']
        bias_cop[i] = df_res_model['bias_cop']
        rmse_gow[i] = df_res_model['rmse_gow']
        rmse_cop[i] = df_res_model['rmse_cop']
        p_gow[i] = df_res_model['pearson_gow']
        p_cop[i] = df_res_model['pearson_cop']
        si_gow[i] = df_res_model['si_gow']
        si_cop[i] = df_res_model['si_cop']

    plot_stats_comp(nombre, models, bias_gow, bias_cop, rmse_gow, rmse_cop, p_gow, p_cop, si_gow, si_cop, f'plot/model/00_Comp/{nombre}.png')

bias_gow_mean = [df_res[df_res['Modelo'] == model]['bias_gow'].mean() for model in models]
bias_cop_mean = [df_res[df_res['Modelo'] == model]['bias_cop'].mean() for model in models]
rmse_gow_mean = [df_res[df_res['Modelo'] == model]['rmse_gow'].mean() for model in models]
rmse_cop_mean = [df_res[df_res['Modelo'] == model]['rmse_cop'].mean() for model in models]
p_gow_mean = [df_res[df_res['Modelo'] == model]['pearson_gow'].mean() for model in models]
p_cop_mean = [df_res[df_res['Modelo'] == model]['pearson_cop'].mean() for model in models]
si_gow_mean = [df_res[df_res['Modelo'] == model]['si_gow'].mean() for model in models]
si_cop_mean = [df_res[df_res['Modelo'] == model]['si_cop'].mean() for model in models]

plot_stats_comp('Media', models, bias_gow_mean, bias_cop_mean, rmse_gow_mean, rmse_cop_mean, p_gow_mean, p_cop_mean, si_gow_mean, si_cop_mean,
                f'plot/model/00_Comp/mean.png')
