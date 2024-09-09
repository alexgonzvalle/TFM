import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec


def format_m(value, _):
    return f'{value:.0f} m'


def format_s(value, _):
    return f'{value:.0f} s'


def plot_rose(fig, ax, dire, hs, title, c, alpha=0.3, _max=None, func_format=format_m, fontsize=6):
    labels = ['N', 'N-E', 'E', 'S-E', 'S', 'S-W', 'W', 'N-W']

    _ax = fig.add_subplot(ax, projection='polar')
    _ax.set_title(title, fontsize=fontsize)
    _ax.plot(np.deg2rad(dire), hs, 'o', color=c, alpha=alpha, markersize=0.5)

    if _max:
        _ax.set_rmax(_max)
    _ax.set_theta_offset(np.pi / 2)
    _ax.set_theta_direction(-1)
    _ax.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
    _ax.set_xticklabels(labels, fontsize=fontsize)
    _ax.yaxis.set_major_formatter(FuncFormatter(func_format))

    return _ax


def plot_rose_contourf(fig, ax, dire, hs, value, title, cmap, levels=None, alpha=1, _max=None, func_format=format_m, fontsize=6):
    labels = ['N', 'N-E', 'E', 'S-E', 'S', 'S-W', 'W', 'N-W']

    _ax = fig.add_subplot(ax, projection='polar')
    # _ax.set_title(title, fontsize=fontsize)
    sc = _ax.scatter(np.deg2rad(dire), hs, c=value, alpha=alpha, s=0.5, cmap=cmap, vmin=levels[0], vmax=levels[-1])
    cbar = plt.colorbar(sc, location='bottom', ax=_ax, label='Factor (m)', format=FuncFormatter(lambda x, pos: '{:.1f}'.format(x)))
    cbar.ax.tick_params(labelsize=fontsize)

    if _max:
        _ax.set_rmax(_max)
    _ax.set_theta_offset(np.pi / 2)
    _ax.set_theta_direction(-1)
    _ax.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
    _ax.set_xticklabels(labels, fontsize=fontsize)
    _ax.yaxis.set_major_formatter(FuncFormatter(func_format))

    return _ax


def plot_data(boya, copernicus, gow, title, fname=None, fontsize=6):
    plt.rcParams.update({'font.size': fontsize})

    fig = plt.figure()
    plt.suptitle(title)
    gs = GridSpec(nrows=3, ncols=3)

    hs_max = 10  # max(boya.hs.max(), copernicus.VHM0.max(), gow.hs.max()) + 1
    # t02_max = max(boya.t02.max(), copernicus.VTM02.max(), gow.t02.max()) + 1
    # tp_max = max(boya.tp.max(), copernicus.VTPK.max(), gow.tp.max()) + 1
    t_max = 20  # max(t02_max, tp_max)

    # Boya
    plot_rose(fig, gs[0, 0], boya.dir, boya.hs, r'$Hs_{Boya}$', 'blue', _max=hs_max, func_format=format_m, fontsize=fontsize)
    plot_rose(fig, gs[1, 0], boya.dir, boya.t02, r'$T02_{Boya}$', 'blue', _max=t_max, func_format=format_s, fontsize=fontsize)
    plot_rose(fig, gs[2, 0], boya.dir, boya.tp, r'$Tp_{Boya}$', 'blue', _max=t_max, func_format=format_s, fontsize=fontsize)

    # GOW
    plot_rose(fig, gs[0, 1], gow.dir, gow.hs, r'$Hs_{GOW}$', 'purple', _max=hs_max, func_format=format_m, fontsize=fontsize)
    plot_rose(fig, gs[1, 1], gow.dir, gow.t02, r'$T02_{GOW}$', 'purple', _max=t_max, func_format=format_s, fontsize=fontsize)
    plot_rose(fig, gs[2, 1], gow.dir, gow.tp, r'$Tp_{GOW}$', 'purple', _max=t_max, func_format=format_s, fontsize=fontsize)

    # copernicus
    plot_rose(fig, gs[0, 2], copernicus.VMDR, copernicus.VHM0, r'$Hs_{IBI}$', 'orange', _max=hs_max, func_format=format_m, fontsize=fontsize)
    plot_rose(fig, gs[1, 2], copernicus.VMDR, copernicus.VTM02, r'$T02_{IBI}$', 'orange', _max=t_max, func_format=format_s, fontsize=fontsize)
    plot_rose(fig, gs[2, 2], copernicus.VMDR, copernicus.VTPK, r'$Tp_{IBI}$', 'orange', _max=t_max, func_format=format_s, fontsize=fontsize)

    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300)
        plt.close(fig)


def plot_stats(dir_boya, hs_boya, dir_model, hs_model,
               ind_train, y_cal_train, ind_test, y_cal_test,
               bias_model_train, rmse_model_train, pearson_model_train, si_model_train,
               bias_cal_train, rmse_cal_train, pearson_cal_train, si_cal_train,
               bias_model_test, rmse_model_test, pearson_model_test, si_model_test,
               bias_cal_test, rmse_cal_test, pearson_cal_test, si_cal_test,
               name_model, title, c='', fname=None, fontsize=6):
    plt.rcParams.update({'font.size': fontsize})
    size_p = 0.5
    alpha = 0.3

    hs_calibrada = np.hstack((y_cal_train, y_cal_test))
    hs_max = max([max(hs_boya), max(hs_model), max(hs_calibrada)]) + 1

    ticks = [i for i in range(0, int(hs_max) + 1, 2)]

    fig = plt.figure()
    plt.suptitle(title, fontsize=fontsize)
    gs = GridSpec(nrows=2, ncols=3)

    # Rosa de Altura de ola de la Boya
    plot_rose(fig, gs[0, 0], dir_boya, hs_boya, r'$Hs_{Boya}$', 'blue', _max=hs_max, func_format=format_m, fontsize=fontsize)

    # Rosa de Altura de ola del Modelo
    plot_rose(fig, gs[0, 1], dir_model, hs_model, r'$Hs_{' + name_model + '}$', c, _max=hs_max, func_format=format_m, fontsize=fontsize)

    # Rosa de Altura de ola Calibrada
    plot_rose(fig, gs[0, 2], dir_model, hs_calibrada, r'$Hs_{Calibrada}$', 'green', _max=hs_max, func_format=format_m, fontsize=fontsize)

    # Scatter Plot de Altura de ola de la Boya vs Altura de ola del Modelo
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(hs_boya[ind_train], hs_model[ind_train], color=c, s=size_p, alpha=alpha)
    ax3.scatter(hs_boya[ind_train], y_cal_train, color='green', s=size_p, alpha=alpha)
    ax3.plot([0, hs_max], [0, hs_max], color='black', linewidth=0.5, alpha=0.7)
    ax3.set_xlabel(r'$Hs_{Boya} (m)$', fontsize=fontsize)
    ax3.set_ylabel('Hs (m)', fontsize=fontsize)
    ax3.set_title(f'Train. Tamaño de la muestra: {len(ind_train)}', fontsize=fontsize)
    ax3.grid(True)
    ax3.set_aspect('equal', 'box')
    ax3.set_xlim([0, hs_max])
    ax3.set_xticks(ticks)
    ax3.set_ylim([0, hs_max])
    ax3.set_yticks(ticks)

    ax3.text(0.01, 0.95, 'bias: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.15, 0.95, f'{bias_model_train:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.35, 0.95, f'{bias_cal_train:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    ax3.text(0.01, 0.9, 'rmse: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.17, 0.9, f'{rmse_model_train:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.4, 0.9, f'{rmse_cal_train:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    ax3.text(0.01, 0.85, 'Pearson: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.25, 0.85, f'{pearson_model_train:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.45, 0.85, f'{pearson_cal_train:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    ax3.text(0.01, 0.8, 'SI: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.15, 0.8, f'{si_model_train:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.35, 0.8, f'{si_cal_train:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    # Scatter Plot de Altura de ola de la Boya vs Altura de ola del Modelo
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(hs_boya[ind_test], hs_model[ind_test], color=c, s=size_p, alpha=alpha)
    ax4.scatter(hs_boya[ind_test], y_cal_test, color='green', s=size_p, alpha=alpha)
    ax4.plot([0, hs_max], [0, hs_max], color='black', linewidth=0.5, alpha=0.7)
    ax4.set_xlabel(r'$Hs_{Boya} (m)$', fontsize=fontsize)
    ax4.set_ylabel('Hs (m)', fontsize=fontsize)
    ax4.set_title(f'Test. Tamaño de la muestra: {len(ind_test)}', fontsize=fontsize)
    ax4.grid(True)
    ax4.set_aspect('equal', 'box')
    ax4.set_xlim([0, hs_max])
    ax4.set_xticks(ticks)
    ax4.set_ylim([0, hs_max])
    ax4.set_yticks(ticks)

    ax4.text(0.01, 0.95, 'bias: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax4.text(0.15, 0.95, f'{bias_model_test:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax4.text(0.35, 0.95, f'{bias_cal_test:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    ax4.text(0.01, 0.9, 'rmse: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax4.text(0.17, 0.9, f'{rmse_model_test:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax4.text(0.4, 0.9, f'{rmse_cal_test:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    ax4.text(0.01, 0.85, 'Pearson: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax4.text(0.25, 0.85, f'{pearson_model_test:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax4.text(0.45, 0.85, f'{pearson_cal_test:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    ax4.text(0.01, 0.8, 'SI: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax4.text(0.15, 0.8, f'{si_model_test:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax4.text(0.35, 0.8, f'{si_cal_test:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    # Rosa de Altura de ola Calibrada / Model
    factor_hs = hs_calibrada / hs_model
    plot_rose_contourf(fig, gs[1, 2], dir_model, hs_model, factor_hs, r'$Hs_{Calibrada} / Hs_{' + name_model + '}$',
                       cmap='seismic', levels=np.linspace(0, 2, 11), _max=hs_max, func_format=format_m)

    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300)
        plt.close(fig)


def plot_stats_comp(nombre, models, bias, rmse, p, si, fname=None, fontsize=6):
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].set_title(nombre + ' - Bias', fontsize=fontsize)
    ax[0, 0].boxplot(bias.T, patch_artist=True, tick_labels=models)
    ax[0, 0].set_ylim(-1, 1)
    ax[0, 0].tick_params(axis='both', labelsize=fontsize, rotation=25)
    ax[0, 0].grid()

    ax[1, 0].set_title(nombre + ' - RMSE', fontsize=fontsize)
    ax[1, 0].boxplot(rmse.T, patch_artist=True, tick_labels=models)
    ax[1, 0].tick_params(axis='both', labelsize=fontsize, rotation=25)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].grid()

    ax[0, 1].set_title(nombre + ' - Pearson', fontsize=fontsize)
    ax[0, 1].boxplot(p.T, patch_artist=True, tick_labels=models)
    ax[0, 1].tick_params(axis='both', labelsize=fontsize, rotation=25)
    ax[0, 1].set_ylim(-1, 1)
    ax[0, 1].grid()

    ax[1, 1].set_title(nombre + ' - SI', fontsize=fontsize)
    ax[1, 1].boxplot(si.T, patch_artist=True, tick_labels=models)
    ax[1, 1].tick_params(axis='both', labelsize=fontsize, rotation=25)
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].grid()

    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300)
        plt.close(fig)


def plot_mse_k(k, k_sel, mse, fname=None):
    k_opt = np.argmin(mse)

    fig, ax = plt.subplots()

    ax.plot(k, mse)
    ax.set_xticks([2, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

    ax.axvline(k_sel, color='red', linestyle='--')
    ax.axhline(mse[k_sel], color='red', linestyle='--')
    ax.axvline(k_opt, color='green', linestyle='--')
    ax.axhline(mse[k_opt], color='green', linestyle='--')

    ax.legend(['MSE', f'k_sel={k_sel}', f'MSE={mse[k_sel+2]:.4f}', f'k_opt={k_opt+2}', f'MSE={mse[k_opt]:.4f}'])

    ax.set_xlabel('k')
    ax.set_ylabel('MSE')
    ax.set_title('Evolucion del MSE en funcion de ')

    plt.grid()
    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300)
        plt.close(fig)
    else:
        plt.show()
