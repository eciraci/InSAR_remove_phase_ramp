#!/usr/bin/env python
u"""
remove_phase_ramp.py
Written by Enrico Ciraci' (03/2022)

Estimate and Remove a Linear Phase Ramp (or background fringes) characterizing
the differential interferogram provided as input.

A first guess of the spatial frequencies characterizing the phase ramp must be
provided by the user - see freq_r & freq_c input parameters.
A GRID-SEARCH around the first guess frequencies is performed to obtain
the best estimate of the ramp parameters.

COMMAND LINE OPTIONS:
usage: remove_phase_ramp.py [-h] [--slope {-1,1}] [--s_radius S_RADIUS]
                [--s_step S_STEP] path_to_intf freq_r freq_c

Estimate and Remove Linear Phase Ramp characterizing the
considered Differential Interferogram.

Positional arguments:
  path_to_intf          Absolute path to input interferogram.
  freq_r                Number of Cycles -> Rows-axis.
  freq_c                Number of Cycles -> Columns-axis.

Optional arguments:
  -h, --help            show this help message and exit
  --slope {-1,1}, -S {-1,1} Ramp Slope.
  --s_radius S_RADIUS, -U S_RADIUS
              Grid Search Radius around the provided reference.
  --s_step S_STEP, -T S_STEP
              Grid Search Step

PYTHON DEPENDENCIES:
    argparse: Parser for command-line options, arguments and sub-commands
           https://docs.python.org/3/library/argparse.html
    numpy: The fundamental package for scientific computing with Python
           https://numpy.org/
    matplotlib: library for creating static, animated, and interactive
           visualizations in Python.
           https://matplotlib.org
    rasterio: Access to geospatial raster data
           https://rasterio.readthedocs.io
    datetime: Basic date and time types
           https://docs.python.org/3/library/datetime.html#module-datetime
    tqdm: Progress Bar in Python.
          https://tqdm.github.io/

UPDATE HISTORY:
"""
# - - python dependencies
from __future__ import print_function
import argparse
import numpy as np
import rasterio
import rasterio.mask
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
# - program dependencies
from utils.mpl_utils import add_colorbar


def estimate_phase_ramp(dd_phase_complex, freq_r: int, freq_c: int,
                        slope: int = 1, s_radius: float = 2,
                        s_step: float = 0.1):
    """
    Estimate a phase ramp from the provided input interferogram
    :param dd_phase_complex: interferogram pahse expressed as complex array
    :param freq_r: phase ramp frequency along rows
    :param freq_c: phase ramp frequency along columns
    :param slope: phase ramp slope (1: increasing phase, -1: decreasing phase)
    :param s_radius: grid-search domain radius
    :param s_step: grid-search step
    :return: Python dictionary containing the results of the grid-search
    """
    # - Generate synthetic field domain
    array_dim = dd_phase_complex.shape
    n_rows = array_dim[0]
    n_columns = array_dim[1]
    raster_mask = np.ones(array_dim)
    raster_mask[np.isnan(dd_phase_complex)] = 0

    # - Integration Domain used to define the phase ramp
    xx_m, yy_m = np.meshgrid(np.arange(n_columns), np.arange(n_rows))

    if freq_r - s_radius <= 0:
        n_cycle_r_vect_f = np.arange(s_step, freq_r + s_radius + s_step,
                                     s_step)
    else:
        n_cycle_r_vect_f = np.arange(freq_r - s_radius,
                                     freq_r + s_radius + s_step,
                                     s_step)
    if freq_c - s_radius <= 0:
        n_cycle_c_vect_f = np.arange(s_step, freq_c + s_radius + s_step,
                                     s_step)
    else:
        n_cycle_c_vect_f = np.arange(freq_c - s_radius,
                                     freq_c + s_radius + s_step,
                                     s_step)

    # - Create Grid Search Domain
    n_cycle_c_vect_f_xx, n_cycle_r_vect_f_yy \
        = np.meshgrid(n_cycle_c_vect_f, n_cycle_r_vect_f)
    error_array_f = np.zeros([len(n_cycle_r_vect_f), len(n_cycle_c_vect_f)])

    for r_count, n_cycle_r in tqdm(enumerate(list(n_cycle_r_vect_f)),
                                   total=len(n_cycle_r_vect_f), ncols=60):
        for c_count, n_cycle_c in enumerate(list(n_cycle_c_vect_f)):
            synth_real = (2 * np.pi / n_columns) * n_cycle_c * xx_m
            synth_imag = (2 * np.pi / n_rows) * n_cycle_r * yy_m
            synth_phase_plane = synth_real + synth_imag
            synth_complex = np.exp(slope * 1j * synth_phase_plane)

            # - Compute Complex Conjugate product between the synthetic phase
            # - ramp and the input interferogram.
            dd_phase_complex_corrected \
                = np.angle(dd_phase_complex * np.conj(synth_complex))
            # - Compute the Mean Absolute value of the phase residuals
            # - > Mean Absolute Error
            error = np.abs(dd_phase_complex_corrected)
            mae = np.nansum(error) / np.nansum(raster_mask)
            error_array_f[r_count, c_count] = mae

    return{'error_array_f': error_array_f,
           'xx_m': xx_m, 'yy_m': yy_m,
           'n_cycle_c_vect_f': n_cycle_c_vect_f,
           'n_cycle_c_vect_f_xx': n_cycle_c_vect_f_xx,
           'n_cycle_r_vect_f': n_cycle_r_vect_f,
           'n_cycle_r_vect_f_yy': n_cycle_r_vect_f_yy
           }


def remove_phase_ramp(path_to_intf: str, freq_r: int, freq_c: int,
                      slope: int = 1, s_radius: float = 2,
                      s_step: float = 0.1) -> dict:
    """
    Estimate and Remove a phase ramp from the provided input interferogram
    :param path_to_intf: absolute path to input interferogram
    :param freq_r: phase ramp number of cycles along rows
    :param freq_c: phase ramp number of cycles columns
    :param slope: phase ramp slope (1: increasing phase, -1: decreasing phase)
    :param s_radius: grid-search domain radius
    :param s_step: grid-search step
    :return: Python dictionary containing estimated phase rampy and de-ramped
             interferogram.
    """
    print('# - Provided First Guess:')
    print(f'# - Num. Cycles -> Rows : {freq_r}')
    print(f'# - Num. Cycles -> Columns : {freq_c}\n')

    # - Figure Parameters - Not Editable
    fig_size1 = (10, 8)
    fig_size2 = (8, 8)
    fig_format = 'jpeg'

    # - Read Input Raster
    with rasterio.open(path_to_intf, mode="r+") as dataset_c:
        # - Read Input Raster and Binary Mask
        clipped_raster = np.array(dataset_c.read(1),
                                  dtype=dataset_c.dtypes[0])
        # - Define Valid data mask
        raster_mask = np.array(dataset_c.read_masks(1),
                               dtype=dataset_c.dtypes[0])
        raster_mask[raster_mask == 255] = 1.
        raster_mask[raster_mask == 0] = np.nan

    # - Transform the Input Phase Field into a complex array
    dd_phase_complex = np.exp(1j * clipped_raster)

    # - Generate synthetic field domain
    array_dim = dd_phase_complex.shape
    n_rows = array_dim[0]
    n_columns = array_dim[1]

    print('# - Running Grid-Search around first guess.')
    e_ramp = estimate_phase_ramp(dd_phase_complex, freq_r, freq_c, slope=slope,
                                 s_radius=s_radius, s_step=s_step)
    xx_m = e_ramp['xx_m']
    yy_m = e_ramp['yy_m']
    n_cycle_c_vect_f_xx = e_ramp['n_cycle_c_vect_f_xx']
    n_cycle_c_vect_f = e_ramp['n_cycle_c_vect_f']
    n_cycle_r_vect_f_yy = e_ramp['n_cycle_r_vect_f_yy']
    n_cycle_r_vect_f = e_ramp['n_cycle_r_vect_f']
    error_array_f = e_ramp['error_array_f']

    # - Find location of the Minimum Phase Absolute Error Value
    ind_min = np.where(error_array_f == np.nanmin(error_array_f))
    n_cycle_c_min = np.round(n_cycle_c_vect_f[ind_min[1]][0], decimals=3)
    n_cycle_r_min = np.round(n_cycle_r_vect_f[ind_min[0]][0], decimals=3)

    print('# - Minimum Found:')
    print(f'# - Num. Cycles -> Rows : {n_cycle_r_min}')
    print(f'# - Num. Cycles -> Columns : {n_cycle_c_min}')

    # - Plot Error Array
    fig_0 = plt.figure(figsize=fig_size1)
    ax_0 = fig_0.add_subplot(111)
    ax_0.set_title(r'Mean Absolute Error - '
                   rf'$\Delta Num. Cycles$ = {s_step}', weight='bold')
    ax_0.set_xlabel(r'$X - Num. Cycles$')
    ax_0.set_ylabel(r'$Y - Num. Cycles$')
    ag_0 = ax_0.pcolormesh(n_cycle_c_vect_f_xx, n_cycle_r_vect_f_yy,
                           error_array_f, cmap=plt.cm.get_cmap('jet'))
    ax_0.scatter(n_cycle_c_vect_f_xx[ind_min] + s_step / 2,
                 n_cycle_r_vect_f_yy[ind_min] + s_step / 2,
                 marker='X', color='m', s=180, label='Minimum MAE')
    cb_0 = plt.colorbar(ag_0)
    cb_0.set_label(label='Rad', weight='bold')
    cb_0.ax.set_xticks([-np.pi, 0, np.pi])
    cb_0.ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])
    ax_0.legend(loc='best', prop={'size': 16})
    ax_0.grid(color='m', linestyle='dotted', alpha=0.3)
    txt = f'Minimum Error found at: (X={n_cycle_c_min}, Y={n_cycle_r_min})'
    ax_0.annotate(txt, xy=(0.03, 0.03), xycoords="axes fraction",
                  size=12, zorder=100,
                  bbox=dict(boxstyle="square", fc="w"))
    plt.tight_layout()
    # - Save Error Map
    out_intf  \
        = path_to_intf.replace('.tiff', '_MAError_grid_search.' + fig_format)
    plt.savefig(out_intf, dpi=200, format=fig_format)
    plt.close()

    # - Compare Estimated Phase Ramp with input interferogram.
    n_cycle_r = n_cycle_r_vect_f[ind_min[0]]
    n_cycle_c = n_cycle_c_vect_f[ind_min[1]]
    synth_real = (2 * np.pi / n_columns) * n_cycle_c * xx_m
    synth_imag = (2 * np.pi / n_rows) * n_cycle_r * yy_m
    synth_phase_plane = synth_real + synth_imag
    synth_complex = np.exp(slope * 1j * synth_phase_plane)
    synth_wrapped = np.angle(synth_complex)

    fig_1 = plt.figure(figsize=fig_size2)
    ax_1 = fig_1.add_subplot(121)
    ax_1.set_title('Input Interferogram', weight='bold')
    im_1 = ax_1.pcolormesh(clipped_raster * raster_mask,
                           vmin=-np.pi, vmax=np.pi,
                           cmap=plt.cm.get_cmap('jet'))
    cb_1 = add_colorbar(fig_1, ax_1, im_1)
    cb_1.set_label(label='Rad', weight='bold')
    cb_1.ax.set_xticks([-np.pi, 0, np.pi])
    cb_1.ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])
    ax_1.grid(color='m', linestyle='dotted', alpha=0.3)

    ax_2 = fig_1.add_subplot(122)
    ax_2.set_title('Estimated Phase Ramp', weight='bold')
    im_2 = ax_2.pcolormesh(synth_wrapped * raster_mask,
                           cmap=plt.cm.get_cmap('jet'))
    cb_2 = add_colorbar(fig_1, ax_2, im_2)
    cb_2.set_label(label='Rad', weight='bold')
    cb_2.ax.set_xticks([-np.pi, 0, np.pi])
    cb_2.ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])
    ax_2.grid(color='m', linestyle='dotted', alpha=0.3)

    txt = f'Number of Cycles: \n(X={n_cycle_c_min}, Y={n_cycle_r_min})'
    ax_2.annotate(txt, xy=(0.03, 0.03), xycoords="axes fraction",
                  size=12, zorder=100,
                  bbox=dict(boxstyle="square", fc="w"))
    plt.tight_layout()
    # - save output figure
    out_intf = path_to_intf.replace('.tiff',
                                    '_input_field_vs_phase_ramp.' + fig_format)
    plt.savefig(out_intf, dpi=200, format=fig_format)
    plt.close()

    # - Remove the estimated phase ramp from the input phase field.
    # - Compute the complex conjugate product between the input phase
    # - field and the estimated ramp.
    dd_phase_complex_corrected \
        = np.angle(dd_phase_complex * np.conj(synth_complex))

    fig_3 = plt.figure(figsize=fig_size2)
    ax_3 = fig_3.add_subplot(121)
    ax_3.set_title('Input Field', weight='bold')
    im_3a = ax_3.pcolormesh(clipped_raster * raster_mask,
                            vmin=-np.pi, vmax=np.pi,
                            cmap=plt.cm.get_cmap('jet'))
    cb_3a = add_colorbar(fig_3, ax_3, im_3a)
    cb_3a.set_label(label='Rad', weight='bold')
    cb_3a.ax.set_xticks([-np.pi, 0, np.pi])
    cb_3a.ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])
    ax_3.grid(color='m', linestyle='dotted', alpha=0.3)

    ax_3 = fig_3.add_subplot(122)
    ax_3.set_title('Input Field - Phase Ramp', weight='bold')
    im_3b = ax_3.pcolormesh(dd_phase_complex_corrected * raster_mask,
                            cmap=plt.cm.get_cmap('jet'))
    cb_3b = add_colorbar(fig_3, ax_3, im_3b)
    cb_3b.set_label(label='Rad', weight='bold')
    cb_3b.ax.set_xticks([-np.pi, 0, np.pi])
    cb_3b.ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])
    ax_3.grid(color='m', linestyle='dotted', alpha=0.3)
    plt.tight_layout()
    # - save output figure
    out_fig_3 = path_to_intf.replace('.tiff', '_deramped.' + fig_format)
    plt.savefig(out_fig_3, dpi=200, format=fig_format)
    plt.close()

    # - Save the de-ramped interferogram in Geotiff format
    dd_phase_complex_corrected[np.isnan(raster_mask)] = -9999
    with rasterio.open(path_to_intf, mode="r+") as dataset_c:
        o_transform = dataset_c.transform
        o_crs = dataset_c.crs

    dd_phase_complex_corrected = np.array(dd_phase_complex_corrected,
                                          dtype=dataset_c.dtypes[0])
    out_f_name = path_to_intf.replace('.tiff', '_deramped.tiff')
    with rasterio.open(out_f_name, 'w', driver='GTiff',
                       height=dd_phase_complex_corrected.shape[0],
                       width=dd_phase_complex_corrected.shape[1],
                       count=1, dtype=rasterio.float32,
                       crs=o_crs, transform=o_transform,
                       nodata=-9999) as dst:
        dst.write(dd_phase_complex_corrected, 1)

    # -
    return{'synth_phase_ramp': synth_complex,
           'dd_phase': dd_phase_complex_corrected}


def main():
    """
    Main: Estimate and Remove Linear Phase Ramp from input Differential
    Interferogram.
    """
    # - Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Estimate and Remove Linear Phase Ramp characterizing
        the considered Differential Interferogram.
        """
    )
    # - Positional Arguments
    parser.add_argument('path_to_intf', type=str,
                        help='Absolute path to input interferogram.')

    # - Ramp Frequency - Rows-axis - FIRST GUESS
    parser.add_argument('freq_r', type=float, default=None,
                        help='Number of Cycles -> Rows-axis.')

    # - Ramp Frequency - Columns-axis - FIRST GUESS
    parser.add_argument('freq_c', type=float, default=None,
                        help='Number of Cycles -> Columns-axis.')

    # - Phase Ramp Slope: [-1, 1]
    parser.add_argument('--slope', '-S',
                        type=float, default=1, choices=[-1, 1],
                        help='Phase Ramp Slope along main axis.')
    # - Grid-Search Domain Radius
    parser.add_argument('--s_radius', '-U', type=float, default=2,
                        help='Grid-Search Radius around the provided '
                             'reference.')
    # - Grid-Search Step
    parser.add_argument('--s_step', '-T', type=float, default=0.1,
                        help='Grid-Search Step.')

    args = parser.parse_args()

    # - Estimate and remove phase ramp from input interferogram
    remove_phase_ramp(args. path_to_intf, args.freq_r, args.freq_c,
                      slope=args.slope, s_radius=args.s_radius,
                      s_step=args.s_step)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f"# - Computation Time: {end_time - start_time}")