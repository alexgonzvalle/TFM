import numpy as np
from sklearn.metrics import mean_squared_error
from plot import plot_stats


def calculate_stats(value_obs, value_pred):
    bias = (np.mean((np.mean(value_pred) - value_obs) ** 2)) - np.var(value_pred)
    rmse = np.sqrt(mean_squared_error(value_obs, value_pred))
    pearson = np.corrcoef(value_obs, value_pred)[0, 1]
    si = rmse / np.mean(value_obs)

    return bias, rmse, pearson, si


def stats(dir_boya, hs_boya, dir_model, hs_model, hs_cal, hs_max, y_raw, y_cal, name_model, title, c='', plot=True, fname=None):
    # Calcular estadisticas
    bias_model, rmse_model, pearson_model, si_model = calculate_stats(hs_boya, hs_model)
    bias_cal, rmse_cal, pearson_cal, si_cal = calculate_stats(hs_boya, hs_cal)

    if plot:
        plot_stats(dir_boya, hs_boya, dir_model, hs_model, hs_cal, hs_max, y_raw, y_cal,
                   bias_model, rmse_model, pearson_model, si_model, bias_cal, rmse_cal, pearson_cal, si_cal,
                   name_model, title, c, fname)

    return bias_cal, rmse_cal, pearson_cal, si_cal


def eval_eq(equation, value_str, value):
    equation_eval = equation.replace('y_cal = ', '')
    equation_eval = equation_eval.replace('^', '**')
    return eval(equation_eval, {f'{value_str}': value})
