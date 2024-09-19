import numpy as np


def make_mono_stamp(img, candidates, size=1):
    """
    Create monochromatic stamps from an image at specified candidate positions.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.
    candidates : list or numpy.ndarray
        List of candidate positions where stamps will be extracted.
    size : int, optional
        Size of the stamp, by default 1.

    Returns
    -------
    numpy.ndarray
        Array of extracted stamps.
    """
    stamp_shape = np.array((size, size))
    stamp_half_shape = np.require(np.round(stamp_shape / 2), dtype=int)

    start = -stamp_half_shape
    end = stamp_half_shape

    sampling_grid = np.mgrid[start[0] : end[0], start[1] : end[1]]
    sampling_grid = sampling_grid.swapaxes(0, 2).swapaxes(0, 1)

    img = img.reshape((img.shape[0], img.shape[1], 1))

    positions = np.rint(np.array(candidates)).astype(int)

    img = img.transpose(2, 0, 1)

    stamp_grid = (sampling_grid[None, :, :, :] + positions[:, None, None, :]).astype(
        "int32"
    )

    X = stamp_grid[:, :, :, 0]
    Y = stamp_grid[:, :, :, 1]

    stamps = img[:, Y, X].transpose(1, 3, 2, 0)

    return np.array(stamps).astype(np.float32)[:, :, :, 0]
