import numpy as np
from scipy.interpolate import RectBivariateSpline, SmoothBivariateSpline

from .detector import extract_sources
from .stamp_extractor import make_mono_stamp


class PSFSplineModel:
    """
    Class for modeling the Point Spread Function (PSF) using bicubic splines.

    Parameters
    ----------
    data : numpy.ndarray
        Input image data.
    oversampling : float, optional
        Oversampling factor for the PSF model, by default 10.0.
    stamp_size : int, optional
        Size of the stamp used for PSF modeling, by default 31.
    """

    def __init__(self, data, oversampling=10.0, stamp_size=31):
        self.data = data
        self.oversampling = oversampling
        self.stamp_size = stamp_size

    def fit(self):
        """
        Fit the PSF model to the data by extracting sources and fitting splines.

        Returns
        -------
        None
        """
        obj_tbl, _, _ = extract_sources(self.data)

        self.stamp_positions = np.array(
            list(
                zip(
                    obj_tbl["xcentroid"].data.astype(np.float32),
                    obj_tbl["ycentroid"].data.astype(np.float32),
                )
            ),
            dtype=np.float32,
        )
        stamps = make_mono_stamp(self.data, self.stamp_positions, size=self.stamp_size)
        self.stamps = stamps

        self.spline = self._fit_spline(
            stamps, obj_tbl["xcentroid"], obj_tbl["ycentroid"]
        )

    def _fit_spline(self, cutouts, x_positions, y_positions, spline_order=3):
        """
        Fits a PSF model using bicubic splines for the cutouts, and models the spline coefficients
        as smoothly varying functions of x and y positions across the image.

        Parameters
        ----------
        cutouts : numpy.ndarray
            3D array (n, w, h) of PSF cutouts, where n is the number of cutouts and w, h are the dimensions.
        x_positions : numpy.ndarray
            1D array of x positions (length m).
        y_positions : numpy.ndarray
            1D array of y positions (length n).
        spline_order : int, optional
            Order of splines for PSF fitting, by default 3 for bicubic.

        Returns
        -------
        function
            A function that returns the PSF at any given (x, y) in the image.
        """
        n_cutouts, w, h = cutouts.shape

        psf_splines = []
        spline_coefficients = []

        u_grid = np.linspace(0, 1, w)
        v_grid = np.linspace(0, 1, h)

        for i in range(n_cutouts):
            psf_spline = RectBivariateSpline(
                u_grid, v_grid, cutouts[i], kx=spline_order, ky=spline_order
            )
            psf_splines.append(psf_spline)
            spline_coefficients.append(psf_spline.get_coeffs())

        # Convert the list of spline coefficients to a numpy array for easier manipulation
        spline_coefficients = np.array(spline_coefficients)  # (n_cutouts, num_coeffs)

        coeff_functions = []
        num_coeffs = spline_coefficients.shape[1]

        for coeff_idx in range(num_coeffs):
            # Fit a smooth bivariate spline to the coefficients as a function of (X, Y)
            coeff_function = SmoothBivariateSpline(
                x_positions, y_positions, spline_coefficients[:, coeff_idx]
            )
            coeff_functions.append(coeff_function)

        def psf_model(x, y, u, v):
            predicted_coeffs = [f(x, y) for f in coeff_functions]

            psf_spline = RectBivariateSpline(
                u_grid,
                v_grid,
                np.array(predicted_coeffs).reshape(w, h),
                kx=spline_order,
                ky=spline_order,
            )

            return psf_spline(u, v)

        return psf_model

    def evaluate(self, x, y):
        """
        Evaluate the PSF model at given (x, y) positions.

        Parameters
        ----------
        x : float
            X-coordinate in the image.
        y : float
            Y-coordinate in the image.

        Returns
        -------
        numpy.ndarray
            Evaluated PSF at the given (x, y) positions.
        """
        u_grid = np.linspace(0, 1, self.stamp_size)
        v_grid = np.linspace(0, 1, self.stamp_size)

        return self.spline(x, y, u_grid[:, None], v_grid[None, :])

    def __call__(self, x, y):
        """
        Callable interface to evaluate the PSF model.

        Parameters
        ----------
        x : float
            X-coordinate in the image.
        y : float
            Y-coordinate in the image.

        Returns
        -------
        numpy.ndarray
            Evaluated PSF at the given (x, y) positions.
        """
        return self.evaluate(x, y)
