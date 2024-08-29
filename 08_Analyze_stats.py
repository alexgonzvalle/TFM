import pandas as pd
import numpy as np
from plot import plot_stats_comp

df_res = pd.read_csv('res.csv')

models = df_res['Modelo'].unique()
n_models = len(models)

bias_gow = np.array([df_res[df_res['Modelo'] == model]['bias_gow'] for model in models])
bias_cop = np.array([df_res[df_res['Modelo'] == model]['bias_cop'] for model in models])
rmse_gow = np.array([df_res[df_res['Modelo'] == model]['rmse_gow'] for model in models])
rmse_cop = np.array([df_res[df_res['Modelo'] == model]['rmse_cop'] for model in models])
p_gow = np.array([df_res[df_res['Modelo'] == model]['pearson_gow'] for model in models])
p_cop = np.array([df_res[df_res['Modelo'] == model]['pearson_cop'] for model in models])
si_gow = np.array([df_res[df_res['Modelo'] == model]['si_gow'] for model in models])
si_cop = np.array([df_res[df_res['Modelo'] == model]['si_cop'] for model in models])

plot_stats_comp('Media GOW', models, bias_gow, rmse_gow, p_gow, si_gow, f'plot/model/00_Comp/mean_gow.png')
plot_stats_comp('Media IBI', models, bias_cop, rmse_cop, p_cop, si_cop, f'plot/model/00_Comp/mean_ibi.png')

plot_stats_comp('Media', models,
                np.hstack(([bias_gow, bias_cop])),
                np.hstack(([rmse_gow, rmse_cop])),
                np.hstack(([p_gow, p_cop])),
                np.hstack(([si_gow, si_cop])),
                f'plot/model/00_Comp/mean.png')
