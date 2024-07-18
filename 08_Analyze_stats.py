import pandas as pd
import numpy as np
from plot import plot_stats_comp

bias_gow_mean, bias_cop_mean = None, None
rmse_gow_mean, rmse_cop_mean = None, None
p_gow_mean, p_cop_mean = None, None
si_gow_mean, si_cop_mean = None, None

df_res = pd.read_csv('res.csv')
for nombre in df_res['Nombre'].unique():
    df_res_nombre = df_res[df_res['Nombre'] == nombre]
    models = df_res_nombre['Modelo'].unique()
    n_models = len(models)

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

    plot_stats_comp(nombre, models, bias_gow, bias_cop, rmse_gow, rmse_cop, p_gow, p_cop, si_gow, si_cop, f'plot/model/comp/{nombre}.png')

    bias_gow_mean = (bias_gow_mean + bias_gow) / 2 if bias_gow_mean is not None else bias_gow
    bias_cop_mean = (bias_cop_mean + bias_cop) / 2 if bias_cop_mean is not None else bias_cop
    rmse_gow_mean = (rmse_gow_mean + rmse_gow) / 2 if rmse_gow_mean is not None else rmse_gow
    rmse_cop_mean = (rmse_cop_mean + rmse_cop) / 2 if rmse_cop_mean is not None else rmse_cop
    p_gow_mean = (p_gow_mean + p_gow) / 2 if p_gow_mean is not None else p_gow
    p_cop_mean = (p_cop_mean + p_cop) / 2 if p_cop_mean is not None else p_cop
    si_gow_mean = (si_gow_mean + si_gow) / 2 if si_gow_mean is not None else si_gow
    si_cop_mean = (si_cop_mean + si_cop) / 2 if si_cop_mean is not None else si_cop

plot_stats_comp('Media', models, bias_gow_mean, bias_cop_mean, rmse_gow_mean, rmse_cop_mean, p_gow_mean, p_cop_mean, si_gow_mean, si_cop_mean,
                f'plot/model/comp/mean.png')
