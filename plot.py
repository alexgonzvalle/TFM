from turtledemo.__main__ import font_sizes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata


def format_m(value, _):
    return f'{value:.0f} m'


def format_s(value, _):
    return f'{value:.0f} s'


def plot_rose(fig, ax, dire, hs, title, c, alpha=1, _max=None, func_format=format_m, fontsize=6):
    labels = ['N', 'N-E', 'E', 'S-E', 'S', 'S-W', 'W', 'N-W']

    _ax = fig.add_subplot(ax, projection='polar')
    _ax.set_title(title, fontsize=fontsize)
    _ax.plot(np.deg2rad(dire), hs, 'o', color=c, alpha=alpha, markersize=1)

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
    dire_c = np.deg2rad(dire)
    _x = np.linspace(dire_c.min(), dire_c.max(), 100)
    _y = np.linspace(hs.min(), hs.max(), 100)
    x, y = np.meshgrid(_x, _y)
    z = griddata((dire_c, hs), value, (x, y))

    _ax = fig.add_subplot(ax, projection='polar')
    _ax.set_title(title, fontsize=fontsize)
    cf = _ax.contourf(x, y, z, cmap=cmap, levels=levels, alpha=alpha)
    cbar = plt.colorbar(cf, location='bottom', ax=_ax, label='Factor (m)', format=FuncFormatter(lambda x, pos: '{:.1f}'.format(x)))
    cbar.ax.tick_params(labelsize=fontsize)

    if _max:
        _ax.set_rmax(_max)
    _ax.set_theta_offset(np.pi / 2)
    _ax.set_theta_direction(-1)
    _ax.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
    _ax.set_xticklabels(labels, fontsize=fontsize)
    _ax.yaxis.set_major_formatter(FuncFormatter(func_format))

    return _ax


def plot_data(boya, copernicus, gow, title, fname=None):
    fig = plt.figure()
    plt.suptitle(title)
    gs = GridSpec(nrows=2, ncols=3)

    hs_max = max(boya.hs.max(), copernicus.VHM0.max(), gow.hs.max()) + 1
    tp_max = max(boya.tp.max(), copernicus.VTPK.max(), gow.tp.max()) + 1

    # Boya
    plot_rose(fig, gs[0, 0], boya.dir, boya.hs, 'Hs Boya', 'blue', _max=hs_max, func_format=format_m)
    plot_rose(fig, gs[1, 0], boya.dir, boya.tp, 'Tp Boya', 'blue', _max=tp_max, func_format=format_s)

    # GOW
    plot_rose(fig, gs[0, 1], gow.dir, gow.hs, 'Hs gow', 'purple', _max=hs_max, func_format=format_m)
    plot_rose(fig, gs[1, 1], gow.dir, gow.tp, 'Tp gow', 'purple', _max=tp_max, func_format=format_s)

    # copernicus
    plot_rose(fig, gs[0, 2], copernicus.VMDR, copernicus.VHM0, 'Hs copernicus', 'orange', _max=hs_max, func_format=format_m)
    plot_rose(fig, gs[1, 2], copernicus.VMDR, copernicus.VTPK, 'Tp copernicus', 'orange', _max=tp_max, func_format=format_s)

    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300)
        plt.close(fig)


def plot_stats(dir_boya, hs_boya, dir_model, hs_model, hs_cal, hs_max, y_raw, y_cal,
               bias_model, rmse_model, pearson_model, bias_cal, rmse_cal, pearson_cal,
               name_model, title, c='', fname=None, fontsize=6):
    fig = plt.figure()
    plt.suptitle(title, fontsize=fontsize)
    gs = GridSpec(nrows=2, ncols=4)

    # Rosa de Altura de ola de la Boya
    plot_rose(fig, gs[0, 0], dir_boya, hs_boya, r'$Hs_{Boya}$', 'blue', _max=hs_max, func_format=format_m, fontsize=fontsize)

    # Rosa de Altura de ola del Modelo
    plot_rose(fig, gs[0, 1], dir_model, hs_model, r'$Hs_{' + name_model + '}$', c, _max=hs_max, func_format=format_m, fontsize=fontsize)

    # Rosa de Altura de ola Calibrada
    plot_rose(fig, gs[0, 2], dir_model, hs_cal, r'$Hs_{Calibrada}$', 'green', _max=hs_max, func_format=format_m, fontsize=fontsize)

    # Rosa de Altura de ola Calibrada / Model
    factor_hs = hs_cal / hs_model
    plot_rose_contourf(fig, gs[0, 3], dir_model, hs_model, factor_hs, r'$Hs_{Calibrada} / Hs_{' + name_model + '}$',
                       cmap='seismic', levels=np.linspace(0, 2, 11),
                       _max=hs_max, func_format=format_m)

    # Scatter Plot de Altura de ola de la Boya vs Altura de ola del Modelo
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.scatter(hs_boya, hs_model, color=c, s=2, alpha=0.6)
    ax3.scatter(hs_boya, hs_cal, color='green', s=2, alpha=0.6)
    ax3.plot(np.linspace(0, hs_max, 11), y_raw, color=c, linestyle='--', linewidth=1)
    ax3.plot(np.linspace(0, hs_max, 11), y_cal, color='green', linestyle='--', linewidth=1)
    ax3.plot([0, hs_max], [0, hs_max], color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel(r'$Hs_{Boya} (m)$', fontsize=fontsize)
    ax3.set_ylabel('Hs (m)', fontsize=fontsize)
    ax3.set_xlim(0, hs_max)
    ax3.set_ylim(0, hs_max)

    ax3.text(0.01, 0.95, 'bias: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.1, 0.95, f'{bias_model:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.23, 0.95, f'{bias_cal:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    ax3.text(0.01, 0.9, 'rmse: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.12, 0.9, f'{rmse_model:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.25, 0.9, f'{rmse_cal:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    ax3.text(0.01, 0.85, 'Pearson: ', fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.16, 0.85, f'{pearson_model:.4f}', color=c, fontsize=fontsize, transform=plt.gca().transAxes)
    ax3.text(0.29, 0.85, f'{pearson_cal:.4f}', color='green', fontsize=fontsize, transform=plt.gca().transAxes)

    # Distribuccion acumulada de la altura de ola
    boya_sorted = np.sort(hs_boya)
    p_boya = 100 * (np.arange(len(boya_sorted)) / (len(boya_sorted) - 1))
    model_sorted = np.sort(hs_model)
    p_model = 100 * (np.arange(len(model_sorted)) / (len(model_sorted) - 1))
    cal_sorted = np.sort(hs_cal)
    p_cal = 100 * (np.arange(len(cal_sorted)) / (len(cal_sorted) - 1))

    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.set_title('Distribución Acumulada', fontsize=fontsize)
    ax5.plot(p_boya, boya_sorted, lw=2, label='Boya', color='blue')
    ax5.plot(p_model, model_sorted, lw=2, label=name_model, color=c)
    ax5.plot(p_cal, cal_sorted, lw=2, label='Calibrada', color='green')
    ax5.set_ylim(0, hs_max)
    ax5.set_xlabel('Probabilidad', fontsize=fontsize)
    ax5.set_ylabel('Hs (m)', fontsize=fontsize)
    ax5.legend(fontsize=fontsize)

    plt.rcParams.update({'font.size': fontsize})
    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300)
        plt.close(fig)

