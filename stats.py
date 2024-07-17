import numpy as np
from sklearn.metrics import mean_squared_error
from plot import plot_stats


def stats(dir_boya, hs_boya, dir_model, hs_model, hs_cal, hs_max, y_raw, y_cal, name_model, title, c='', plot=True, fname=None):
    # Calcular estadisticas
    bias_model = (np.mean((np.mean(hs_model) - hs_boya) ** 2)) - np.var(hs_model)
    rmse_model = np.sqrt(mean_squared_error(hs_boya, hs_model))
    pearson_model = np.corrcoef(hs_boya, hs_model)[0, 1]

    bias_cal = (np.mean((np.mean(hs_cal) - hs_boya) ** 2)) - np.var(hs_cal)
    rmse_cal = np.sqrt(mean_squared_error(hs_boya, hs_cal))
    pearson_cal = np.corrcoef(hs_boya, hs_cal)[0, 1]

    if plot:
        plot_stats(dir_boya, hs_boya, dir_model, hs_model, hs_cal, hs_max, y_raw, y_cal,
                   bias_model, rmse_model, pearson_model, bias_cal, rmse_cal, pearson_cal,
                   name_model, title, c, fname)

    return bias_cal, rmse_cal, pearson_cal


def eval_eq(equation, value_str, value):
    equation_eval = equation.replace('y_cal = ', '')
    equation_eval = equation_eval.replace('^', '**')
    return eval(equation_eval, {f'{value_str}': value})
