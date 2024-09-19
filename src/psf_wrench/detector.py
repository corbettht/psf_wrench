#!/usr/bin/env python3
import astropy.table as tbl
import numpy as np
import numpy.typing as npt
import sep
from typing import Optional


class SEPSourceDetect:
    """
    Class for source detection using SEP (Source Extractor in Python).

    Parameters
    ----------
    in_img_shape : tuple
        Shape of the input image.
    bg_map_shape : tuple
        Shape of the background map.
    """

    def __init__(self, in_img_shape, bg_map_shape):
        self.in_img_shape = in_img_shape
        self.bg_map_shape = bg_map_shape

    def detect(
        self, data: npt.ArrayLike, detect_sigma: float, margin: Optional[int] = 10
    ):
        """
        Detect sources in the given data.

        Parameters
        ----------
        data : numpy.ndarray
            Input image data.
        detect_sigma : float
            Detection threshold in units of background RMS.
        margin : int, optional
            Margin to exclude around the edges of the image, by default 10.

        Returns
        -------
        obj_tbl : astropy.table.Table
            Table of detected objects with their properties.
        bkg_back : numpy.ndarray
            Background map.
        bkg_rms : numpy.ndarray
            RMS of the background.
        """
        data = data.astype(np.float32, copy=False)

        bkg = sep.Background(
            data, bw=self.bg_map_shape[0], bh=self.bg_map_shape[0], fw=3, fh=3
        )
        data -= bkg.back()

        noise_image = np.sqrt(np.abs(data) / 0.5 + bkg.rms() ** 2)

        sep.set_extract_pixstack(30000000)
        objects = sep.extract(
            data,
            detect_sigma,
            err=noise_image,
            minarea=3,
            filter_type="conv",
            deblend_nthresh=24,
            deblend_cont=1,
        )
        # deblend_cont=1.0)

        flux, fluxerr, flag = sep.sum_circle(
            data, objects["x"], objects["y"], 4.5, err=noise_image
        )

        obj_tbl = tbl.Table({name: objects[name] for name in objects.dtype.names})
        r, flag = sep.flux_radius(
            data,
            obj_tbl["x"],
            obj_tbl["y"],
            6.0 * obj_tbl["a"],
            0.5,
            normflux=flux,
            subpix=5,
        )
        sig = 2.0 / 2.35 * r

        xwin, ywin, flag = sep.winpos(data, obj_tbl["x"], obj_tbl["y"], sig, subpix=0)

        obj_tbl["flux"] = flux
        obj_tbl["fluxerr"] = fluxerr
        obj_tbl["flag"] = flag
        obj_tbl["xcentroid"] = xwin
        obj_tbl["ycentroid"] = ywin
        obj_tbl = obj_tbl[obj_tbl["flux"] > 0]
        obj_tbl = obj_tbl[obj_tbl["fluxerr"] > 0]

        obj_tbl["snr"] = obj_tbl["flux"] / obj_tbl["fluxerr"]

        obj_tbl = obj_tbl[obj_tbl["flux"] > 0]
        obj_tbl["mag"] = (
            -2.5 * np.log10(obj_tbl["flux"]) + 23
        )  # 23 is an arbitrary zeropoint to get things > 0

        return obj_tbl, bkg.back(), bkg.rms()


def extract_sources(data: npt.ArrayLike, detect_sigma=5.0, bg_meshsize=128):
    """
    Extract sources from the given data.

    Parameters
    ----------
    data : numpy.ndarray
        Input image data.
    detect_sigma : float, optional
        Detection threshold in units of background RMS, by default 5.0.
    bg_meshsize : int, optional
        Size of the background mesh, by default 128.

    Returns
    -------
    obj_tbl : astropy.table.Table
        Table of detected objects with their properties.
    bkg_back : numpy.ndarray
        Background map.
    bkg_rms : numpy.ndarray
        RMS of the background.
    """
    detector = SEPSourceDetect(data.shape, (bg_meshsize, bg_meshsize))
    return detector.detect(data, detect_sigma)
