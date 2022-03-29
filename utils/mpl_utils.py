"""
Enrico Ciraci 03/2022
Set of utility functions that can be used to generate figures with matplotlib.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_colorbar(fig: plt.figure, ax: plt.Axes,
                 im: plt.pcolormesh) -> plt.colorbar:
    """
    Add colorbar to the selected plt.Axes.
    :param fig: plt.figure object
    :param ax: plt.Axes object.
    :param im: plt.pcolormesh object.
    :return: plt.colorbar
    """
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='5%', pad=0.6, pack_start=True)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    return cb
