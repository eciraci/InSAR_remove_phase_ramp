#!/usr/bin/env python
u"""
remove_phase_ramp_fft.py
Written by Enrico Ciraci' (03/2022)

Estimate and Remove the contribution of a "Linear Ramp" to the Wrapped Phase
of a Differential InSAR Interferogram.

Estimate the Linear Phase Ramp in the Frequency Domain as the maximum value
of the Power Spectrum of the Signal.

COMMAND LINE OPTIONS:
usage: remove_phase_ramp_fft.py [-h] path_to_intf

Estimate and Remove Linear Phase Ramp characterizing the considered
Differential Interferogram - Fast Fourier Based Approach.

positional arguments:
  path_to_intf  Absolute path to input interferogram.

optional arguments:
  -h, --help    show this help message and exit

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

UPDATE HISTORY:
"""
# - python dependencies
from __future__ import print_function
import argparse
import numpy as np
import rasterio
import rasterio.mask
import datetime
import matplotlib.pyplot as plt
# - program dependencies
from utils.mpl_utils import add_colorbar


def estimate_phase_ramp(igram_cpx: np.ndarray,
                        row_pad: int = 0, col_pad: int = 0) -> dict:
    """
    Estimate Phase Ramp as the maximum value of signal Power Spectrum
    :param igram_cpx: Interferogram as a complex array
    :param row_pad: padding rows - int
    :param col_pad: padding columns - int
    :return: dict
    """
    # - Use Zero-Padding to increase resolution in the frequency domain
    igram_cpx = np.pad(igram_cpx, ((int(row_pad), int(row_pad)),
                                   (int(col_pad), int(col_pad))),
                       constant_values=((0, 0), (0, 0)))
    # - Compute 2-D Fast Fourier Transform
    igram_fft = np.fft.fft2(igram_cpx)
    # - Compute Power Spectrum
    igram_pwr = np.abs(igram_fft)

    # - Find Power Spectrum Maximum Value
    igram_pwr[:, 0] = 0
    igram_pwr[0, :] = 0
    igram_pwr_nx = igram_pwr[:, :]
    index_t = np.where(igram_pwr_nx == np.max(igram_pwr_nx))
    index_r = index_t[0][0]
    index_c = index_t[1][0]
    # - Generate Synthetic Phase Ramp
    est_synth = np.zeros(igram_pwr.shape, dtype=complex)
    est_synth[index_r, index_c] = igram_fft[index_r, index_c]
    phase_ramp = np.fft.ifft2(est_synth)

    # - Remove padding from the estimated phase ramp
    if row_pad == 0 and col_pad == 0:
        phase_ramp = phase_ramp[:, :]
    elif row_pad == 0 and col_pad != 0:
        phase_ramp = phase_ramp[:, col_pad:-col_pad]
    elif row_pad != 0 and col_pad == 0:
        phase_ramp = phase_ramp[row_pad:-row_pad, :]
    else:
        phase_ramp = phase_ramp[row_pad:-row_pad, col_pad:-col_pad]

    return{'phase_ramp': phase_ramp}


def remove_phase_ramp(path_to_intf: str, row_pad: int = 0,
                      col_pad: int = 0) -> dict:
    """
    Estimate and Remove a phase ramp from the provided input interferogram
    :param path_to_intf: absolute path to input interferogram
    :param row_pad: add zero padding rows
    :param col_pad: add zero padding columns
    :return: Python dictionary containing estimated phase rampy and de-ramped
             interferogram.
    """
    fig_format = 'jpeg'     # - output figure format

    # - Read Input Raster
    with rasterio.open(path_to_intf, mode="r+") as dataset_c:
        # - Read Input Raster and Binary Mask
        intf_phase = np.array(dataset_c.read(1),
                              dtype=dataset_c.dtypes[0])
        # - Define Valid data mask
        raster_mask = np.array(dataset_c.read_masks(1),
                               dtype=dataset_c.dtypes[0])
        raster_mask[raster_mask == 255] = 1.
        raster_mask[raster_mask == 0] = np.nan

    # - Transform the Input Phase Field into a complex array
    # - Create unit-magnitude interferogram.
    dd_phase_complex = np.exp(1j * intf_phase).astype(np.complex64)
    # - Estimate Phase Ramp in the Frequency domain
    rmp = estimate_phase_ramp(dd_phase_complex,
                              row_pad=row_pad, col_pad=col_pad)
    phase_ramp = rmp['phase_ramp']
    # - Extract wrapped phase
    ingram_ramp = np.angle(phase_ramp)

    # - Remove the estimated phase ramp from the input phase field by
    # - computing the complex conjugate product between the input phase
    # - field and the estimated ramp.
    dd_phase_complex_corrected = np.angle(dd_phase_complex
                                          * np.conjugate(phase_ramp))
    fig_1 = plt.figure(figsize=(8, 8))
    ax_1 = fig_1.add_subplot(121)
    ax_1.set_title('Input Interferogram', weight='bold')
    im_1 = ax_1.pcolormesh(intf_phase * raster_mask,
                           vmin=-np.pi, vmax=np.pi,
                           cmap=plt.cm.get_cmap('jet'))
    cb_1 = add_colorbar(fig_1, ax_1, im_1)
    cb_1.set_label(label='Rad', weight='bold')
    cb_1.ax.set_xticks([-np.pi, 0, np.pi])
    cb_1.ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])
    ax_1.grid(color='m', linestyle='dotted', alpha=0.3)

    ax_2 = fig_1.add_subplot(122)
    ax_2.set_title('Estimated Phase Ramp', weight='bold')
    im_2 = ax_2.pcolormesh(ingram_ramp * raster_mask,
                           cmap=plt.cm.get_cmap('jet'))
    cb_2 = add_colorbar(fig_1, ax_2, im_2)
    cb_2.set_label(label='Rad', weight='bold')
    cb_2.ax.set_xticks([-np.pi, 0, np.pi])
    cb_2.ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])
    ax_2.grid(color='m', linestyle='dotted', alpha=0.3)
    plt.tight_layout()
    # - save output figure
    out_intf = path_to_intf.replace('.tiff',
                                    '_input_field_vs_phase_ramp_fft.'
                                    + fig_format)
    plt.savefig(out_intf, dpi=200, format=fig_format)
    plt.close()

    # - Compare Input with corrected Interferogram
    fig_3 = plt.figure(figsize=(8, 8))
    ax_3 = fig_3.add_subplot(121)
    ax_3.set_title('Input Interferogram', weight='bold')
    im_3a = ax_3.pcolormesh(intf_phase * raster_mask,
                            vmin=-np.pi, vmax=np.pi,
                            cmap=plt.cm.get_cmap('jet'))
    cb_3a = add_colorbar(fig_3, ax_3, im_3a)
    cb_3a.set_label(label='Rad', weight='bold')
    cb_3a.ax.set_xticks([-np.pi, 0, np.pi])
    cb_3a.ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])
    ax_3.grid(color='m', linestyle='dotted', alpha=0.3)

    ax_3 = fig_3.add_subplot(122)
    ax_3.set_title('Input Phase Field - Phase Ramp', weight='bold')
    im_3b = ax_3.pcolormesh(dd_phase_complex_corrected * raster_mask,
                            cmap=plt.cm.get_cmap('jet'))
    cb_3b = add_colorbar(fig_3, ax_3, im_3b)
    cb_3b.set_label(label='Rad', weight='bold')
    cb_3b.ax.set_xticks([-np.pi, 0, np.pi])
    cb_3b.ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])
    ax_3.grid(color='m', linestyle='dotted', alpha=0.3)
    plt.tight_layout()
    # - save output figure
    out_fig_3 = path_to_intf.replace('.tiff', '_deramped_fft.' + fig_format)
    plt.savefig(out_fig_3, dpi=200, format=fig_format)
    plt.close()

    # - Save the de-ramped interferogram in Geotiff format
    dd_phase_complex_corrected[np.isnan(raster_mask)] = -9999
    with rasterio.open(path_to_intf, mode="r+") as dataset_c:
        o_transform = dataset_c.transform
        o_crs = dataset_c.crs

    dd_phase_complex_corrected = np.array(dd_phase_complex_corrected,
                                          dtype=dataset_c.dtypes[0])
    out_f_name = path_to_intf.replace('.tiff', '_deramped_fft.tiff')
    with rasterio.open(out_f_name, 'w', driver='GTiff',
                       height=dd_phase_complex_corrected.shape[0],
                       width=dd_phase_complex_corrected.shape[1],
                       count=1, dtype=rasterio.float32,
                       crs=o_crs, transform=o_transform,
                       nodata=-9999) as dst:
        dst.write(dd_phase_complex_corrected, 1)

    # -
    return{'synth_phase_ramp': phase_ramp,
           'dd_phase': dd_phase_complex_corrected}


def main():
    """
    Main: Estimate and Remove Linear Phase Ramp from input Differential
    Interferogram - Fast Fourier Based Approach.
    """
    # - Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Estimate and Remove Linear Phase Ramp characterizing
            the considered Differential Interferogram - 
            Fast Fourier Based Approach.
            """
    )
    # - Positional Arguments
    parser.add_argument('path_to_intf', type=str,
                        help='Absolute path to input interferogram.')
    # - Zero Padding
    parser.add_argument('--row_pad', '-R',
                        type=int, default=0,
                        help='Zero Padding - Rows.')
    parser.add_argument('--col_pad', '-C',
                        type=int, default=0,
                        help='Zero Padding - Columns.')

    args = parser.parse_args()

    # - Processing Parameters
    path_to_intf = args.path_to_intf
    remove_phase_ramp(path_to_intf, row_pad=args.row_pad, col_pad=args.col_pad)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f"# - Computation Time: {end_time - start_time}")
