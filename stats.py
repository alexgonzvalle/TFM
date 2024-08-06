import numpy as np
from sklearn.metrics import mean_squared_error
from plot import plot_stats


def calculate_stats(value_obs, value_pred):
    bias = np.sum(value_pred-value_obs) / len(value_obs)
    rmse = np.sqrt(mean_squared_error(value_obs, value_pred))
    pearson = np.corrcoef(value_obs, value_pred)[0, 1]
    si = rmse / np.mean(value_obs)

    return bias, rmse, pearson, si


def stats(dir_boya, hs_boya, dir_model, hs_model,
          ind_train, y_cal_train, y_cal_train_plot, ind_test, y_cal_test, y_cal_test_plot,
          y_raw, y_max, name_model, title, c='', plot=True, fname=None):
    # Calcular estadisticas
    bias_model_train, rmse_model_train, pearson_model_train, si_model_train = calculate_stats(hs_boya[ind_train], hs_model[ind_train])
    bias_cal_train, rmse_cal_train, pearson_cal_train, si_cal_train = calculate_stats(hs_boya[ind_train], y_cal_train)

    bias_model_test, rmse_model_test, pearson_model_test, si_model_test = calculate_stats(hs_boya[ind_test], hs_model[ind_test])
    bias_cal_test, rmse_cal_test, pearson_cal_test, si_cal_test = calculate_stats(hs_boya[ind_test], y_cal_test)

    if plot:
        plot_stats(dir_boya, hs_boya, dir_model, hs_model,
                   ind_train, y_cal_train, y_cal_train_plot,
                   ind_test, y_cal_test, y_cal_test_plot,
                   y_raw, y_max,
                   bias_model_train, rmse_model_train, pearson_model_train, si_model_train,
                   bias_cal_train, rmse_cal_train, pearson_cal_train, si_cal_train,
                   bias_model_test, rmse_model_test, pearson_model_test, si_model_test,
                   bias_cal_test, rmse_cal_test, pearson_cal_test, si_cal_test,
                   name_model, title, c, fname)

    return bias_cal_test, rmse_cal_test, pearson_cal_test, si_cal_test


def eval_eq(equation, value_str, value):
    equation_eval = equation.replace('y_cal = ', '')
    equation_eval = equation_eval.replace('^', '**')
    return eval(equation_eval, {f'{value_str}': value})
