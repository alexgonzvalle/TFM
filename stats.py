import numpy as np
from sklearn.metrics import mean_squared_error
from plot import plot_stats


def calculate_stats(value_obs, value_pred):
    bias = (np.mean((np.mean(value_pred) - value_obs) ** 2)) - np.var(value_pred)
    rmse = np.sqrt(mean_squared_error(value_obs, value_pred))
    pearson = np.corrcoef(value_obs, value_pred)[0, 1]
    si = rmse / np.mean(value_obs)

    return bias, rmse, pearson, si


def stats(dir_boya, hs_boya, dir_model, hs_model,
          x_train, y_cal_train, y_cal_train_plot, x_test, y_cal_test, y_cal_test_plot,
          y_raw, y_max, name_model, title, c='', plot=True, fname=None):
    # Calcular estadisticas
    bias_model, rmse_model, pearson_model, si_model = calculate_stats(hs_boya, hs_model)
    bias_cal_train, rmse_cal_train, pearson_cal_train, si_cal_train = calculate_stats(x_train, y_cal_train)
    bias_cal_test, rmse_cal_test, pearson_cal_test, si_cal_test = calculate_stats(x_test, y_cal_test)

    if plot:
        plot_stats(dir_boya, hs_boya, dir_model, hs_model,
                   x_train, y_cal_train, y_cal_train_plot,
                   x_test, y_cal_test, y_cal_test_plot,
                   y_raw, y_max,
                   bias_model, rmse_model, pearson_model, si_model,
                   bias_cal_train, rmse_cal_train, pearson_cal_train, si_cal_train,
                   bias_cal_test, rmse_cal_test, pearson_cal_test, si_cal_test,
                   name_model, title, c, fname)

    return bias_cal_test, rmse_cal_test, pearson_cal_test, si_cal_test


def eval_eq(equation, value_str, value):
    equation_eval = equation.replace('y_cal = ', '')
    equation_eval = equation_eval.replace('^', '**')
    return eval(equation_eval, {f'{value_str}': value})
