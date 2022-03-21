#!/usr/bin/env python
"""
Enrico Ciraci 02/2022
Set of utility functions used to read/write raster data using Rasterio.
Rasterio is and alternative GDAL’s Python bindings using more idiomatic Python
types and protocols compared to the standard GDAL library.

For more info about the Rasterio project see:
https://rasterio.readthedocs.io/en/latest/

For more info about GDAL/OGR in Python see:
https://gdal.org/api/python.html
"""
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT
import rasterio.mask
import fiona
import affine
from pyproj import CRS
import numpy as np
from typing import Union


def load_raster(in_path: str) -> dict:
    """
    # - Load raster saved in GeoTiff format
    :param in_path: absolute path to input file
    :return: dictionary containing the input raster + ancillary info.
    """
    with rasterio.open(in_path, mode='r+') as src:
        # - read band #1
        raster_input = src.read(1).astype(src.dtypes[0])
        # - raster upper-left and lower-right corners
        ul_corner = src.transform * (0, 0)
        lr_corner = src.transform * (src.width, src.height)
        grid_res = src.res
        # - compute x- and y-axis coordinates
        x_coords = np.arange(ul_corner[0], lr_corner[0], grid_res[0])
        y_coords = np.arange(lr_corner[1], ul_corner[1], grid_res[1])
        # - compute raster extent - (left, right, bottom, top)
        extent = [ul_corner[0], lr_corner[0], lr_corner[1], ul_corner[1]]
        # - compute cell centroids
        x_centroids = x_coords + (grid_res[0]/2.)
        y_centroids = y_coords + (grid_res[1]/2.)
        # - rotate the output numpy array in such a way that
        # - the lower-left corner of the raster is considered
        # - the origin of the reference system.
        if src.transform.e < 0:
            raster_input = np.flipud(raster_input)
        # - Compute New Affine Transform
        transform = (Affine.translation(x_coords[0], y_coords[0])
                     * Affine.scale(src.res[0], src.res[1]))

        return{'data': raster_input, 'crs': src.crs, 'res': src.res,
               'y_coords': y_coords, 'x_coords': x_coords,
               'y_centroids': y_centroids, 'x_centroids': x_centroids,
               'transform': transform, 'src_transform': src.transform,
               'width': src.width, 'height': src.height, 'extent': extent,
               'ul_corner': ul_corner, 'lr_corner': lr_corner,
               'nodata': src.nodata, 'dtype': src.dtypes[0]}


def save_raster(raster: np.ndarray, res: int, x: np.ndarray,
                y: np.ndarray, out_path: str, crs: int) -> None:
    """
    Save the Provided Raster in GeoTiff format
    :param raster: input raster - np.ndarray
    :param res: raster resolution - integer
    :param x: x-axis - np.ndarray
    :param y: y-axis - np.ndarray
    :param crs: - coordinates reference system
    :param out_path: absolute path to output file
    :return: None
    """
    # - Calculate Affine Transformation of the output raster
    if y[1] > y[0]:
        y = np.flipud(y)
        raster = np.flipud(raster)
    transform = (Affine.translation(x[0], y[0])
                 * Affine.scale(res, -res))
    with rasterio.open(out_path, 'w', driver='GTiff',
                       height=raster.shape[0],
                       width=raster.shape[1], count=1,
                       dtype=raster.dtype, crs=crs,
                       transform=transform,
                       nodata=-9999.) as dst:
        dst.write(raster, 1)


def vrt_param(crs, res: int, bounds: list,
              resampling_alg: str, dtype: str) -> dict:
    """
    Virtual Warp Parameters
    :param crs: destination coordinate reference system
    :param res: output x/y-resolution
    :param bounds: Interpolation Domain Boundaries
    :param resampling_alg: Interpolation Algorithm
    :param dtype: the working data type for warp operation and output.
    :return: dictionary containing vrt options.
    """
    # - Re-projection Parameters
    dst_crs = CRS.from_epsg(crs)  # - Destination CRS
    # - Output image transform
    xres = yres = res
    left, bottom, right, top = bounds
    dst_width = (right - left) / xres
    dst_height = (top - bottom) / yres
    # - Affine transformation matrix
    dst_transform = affine.Affine(xres, 0.0, left,
                                  0.0, -yres, top)
    # - Virtual Warping Options
    vrt_options = {
        'resampling': Resampling[resampling_alg],
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
        'src_nodata': -9999,
        'nodata': -9999,
        'dtype': dtype,
    }
    return vrt_options


def virtual_warp_rio(src_file: str, out_file: str, res: int = 250,
                     crs: int = 3413, method: str = 'med',
                     dtype=None) -> None:
    """
    Rasterio Virtual Warp
    :param src_file: absolute path to source file
    :param out_file: absolute path to output file
    :param res: output resolution
    :param crs: output coordinate reference system
    :param method: resampling method
    :param dtype: output data type
    :return: None
    """
    # - Define output grid - with regular step equal to the
    # - selected resolution
    dem_src = load_raster(src_file)
    # - raster upper - left and lower - right corners
    ul_corner_1 = dem_src['ul_corner']
    lr_corner_1 = dem_src['lr_corner']
    minx = int((ul_corner_1[0] // res) * res) - res
    miny = int((lr_corner_1[1] // res) * res) - res
    maxx = int((lr_corner_1[0] // res) * res) + res
    maxy = int((ul_corner_1[1] // res) * res) + res
    output_bounds = [minx, miny, maxx, maxy]

    with rasterio.open(src_file) as src:
        # - virtual Warp Parameters
        if dtype is None:
            # - if not selected, source data type
            dtype = src.dtypes[0]
        vrt_options = vrt_param(crs, res,
                                output_bounds, method, dtype)
        with WarpedVRT(src, **vrt_options) as vrt:
            # Read all data into memory.
            data = vrt.read()
            # - Process the dataset in chunks.
            # - See Rasterio Documentation for more details.
            # - https://rasterio.readthedocs.io/en/latest
            # - /topics/virtual-warping.html
            for _, window in vrt.block_windows():
                data = vrt.read(window=window)

            # - Save Reprojected Data
            rio_shutil.copy(vrt, out_file, driver='GTiff')


def clip_raster(src_file: str, ref_shp: str, out_file: str) -> Union[str, None]:
    """
    Clip Input Raster Using Rasterio. Find more info here:
    https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
    :param src_file: absolute path to input raster file
    :param ref_shp: absolute path to reference shapefile
    :param out_file: absolute path to output raster file
    :return: None
    """
    # - Open Reference shapefile
    with fiona.open(ref_shp, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]

    # - Open Input Raster
    with rasterio.open(src_file) as src:
        out_raster, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    # - Define Output raster metadata
    out_meta.update({'driver': 'GTiff',
                     'height': out_raster.shape[1],
                     'width': out_raster.shape[2],
                     'transform': out_transform})
    out_raster[out_raster == src.nodata] = np.nan

    if out_raster[np.isfinite(out_raster)].shape[0] == 0:
        return None
    # - Save clipped raster
    # - [only if valid data are found within the clipped area]
    out_raster[np.isnan(out_raster)] = -9999.
    with rasterio.open(out_file, 'w', **out_meta) as dest:
        dest.write(out_raster)
