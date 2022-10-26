"""
A collection of spatial analysis routines.
"""

from collections import namedtuple
import numpy as np
import bisect
import scipy.ndimage as ndimage

OccupancyMap = namedtuple('OccupancyMap',
                          ['tx', 'dt', 'mask', 'hist', 'ivls', 'ivl_masks', 'bins',
                           'smooth', 'max_dt', 'min_dx', 'min_dx_t'])
ZMap = namedtuple('ZMap', ['occupancy', 'z', 'mask', 'hist', 'ivl_masks'])


def shift_pos(xy, o, dp=-0.05):
    """
    Shifts the positions given in xy by dp along orientation given in o.
    """
    mx = xy[:, 0] - np.cos(o) * (-dp)
    my = xy[:, 1] - np.sin(o) * (-dp)
    return np.vstack((mx, my)).T


def diffx(tx):
    """
    Compute the first derivative.
    
    Parameters
    ----------
    tx : ndarray
        A two or three column numpy array containing [t, x] or [t, x, y] points.
        
    Returns
    -------
    dx : ndarray
        An N x 1 array of the first derivative at times t.
    """
    if not (tx.shape[1] in [2, 3]):
        raise IndexError('tx: Invalid number of columns')

    t_diff = (tx[:-1, 0] + tx[1:, 0]) / 2.0
    #     if tx.shape[1] == 2:
    #         dd = tx[:, 1]
    #     else:
    #         dd = np.sqrt(tx[:, 1]**2 + tx[:, 2]**2)
    # get the positional differences
    dx0 = np.diff(tx[:, 1:], axis=0)
    # now take the magnitude and scale for time
    dd = np.sqrt(np.sum(dx0 ** 2, axis=1)) / np.diff(tx[:, 0])
    return np.interp(tx[:, 0], t_diff, dd)


def find_in_intervals(ivl, v):
    """
    Find which values in v fall in between any of the intervals in pq.
    
    Parameters
    ----------
    ivl : ndarray
        A two column numpy array with rows containing [p, q] start and end
        intervals.
    v : ndarray
        A sorted vector of values to find in the intervals.
        
    Returns
    -------
    out : ndarray
        A boolean vector the same shape as v with the truth values of whether
        each element was found in any of the intervals.
    """
    out = np.zeros(v.shape[0], dtype=bool)
    out_mat = np.zeros((ivl.shape[0], v.shape[0]), dtype=bool)
    for (i, t1, t2) in zip(range(ivl.shape[0]), ivl[:, 0], ivl[:, 1]):
        i1 = (bisect.bisect_left(v, t1))
        i2 = (bisect.bisect_right(v, t2))
        out[i1:i2] = True
        out_mat[i, i1:i2] = True
    return out, out_mat


def occupancy_map(tx, bins=40, smooth=0, max_dt=None, min_dx=-1.0, min_dx_t=0.3,
                  range=None):
    """
    Constructs an occupancy map.
    
    Parameters
    ----------
    tx : ndarray
        A two or three column numpy array containing continuously sampled data.
    bins : ndarray, optional
        A tuple (H, V) containing the number of horizontal and vertical bins.
    smooth : int, optional
        The magnitude of the gaussian smoothing.
    max_dt : float, optional
        The maximum gap between samples, larger gaps are truncated to max_dt.
    min_dx : float, optional
        The minimum rate of change of the magnitude of samples in tx.
    min_dx_t : float, optional
        The minimum time required above min_dx, if min_dx is set. If max_dt
        is exceeded on samples, it also creates a discontinuity in the times
        used for min_dx_t.
        
    Returns
    -------
    occupancy : OccupancyMap
        A named tuple containing the occupancy map data along with the
        parameters used to create it.
        
        tx : ndarray
            A copy of the input data.
        dt : ndarray
            Time differences between samples
        mask : ndarray
            A boolean vector of length tx.shape[0] set True for valid samples.
        hist : ndarray
            An H x V array of the time spent in each bin.
        ivls : ndarray
        ivl_masks : ndarray
        bins : ndarray
            The number of bins used for the hist (copied from input).
        max_dt : float
            Copied from input.
        min_dx : float
            Copied from input.
        min_dx_t : float
            Copied from input.
    """

    if not (tx.shape[1] in [2, 3]):
        raise IndexError('tx: Invalid number of columns')

    # set an impossibly large max_dt if unset
    if max_dt == None:
        max_dt = np.max(tx[:, 0])

    # get dt and mask
    dt = np.diff(tx[:, 0])
    dt = np.concatenate((dt, [np.median(dt)]))
    dt_mask = dt < max_dt
    dt[~dt_mask] = max_dt

    # limit samples to those with sustained movement above min_dx
    dx = diffx(tx)
    dx_mask = dx > min_dx
    dx_mask_pad = np.concatenate(([False], dx_mask, [False]))
    dx_begin_mask = ~dx_mask_pad[:-2] & dx_mask_pad[1:-1]  # start of run
    dx_end_mask = dx_mask_pad[1:-1] & ~dx_mask_pad[2:]  # end of run
    # dt > max_dt means breaks in dx runs
    dx_break_mask = dx_mask[:-1] & dx_mask[1:] & ~dt_mask[:-1]
    dx_begin_mask[1:] = dx_begin_mask[1:] | dx_break_mask
    dx_end_mask[:-1] = dx_end_mask[:-1] | dx_break_mask

    assert np.sum(dx_begin_mask) == np.sum(dx_end_mask), \
        "Mismatch in filter interval indices."

    # get the start and end times
    ivls = np.vstack((tx[dx_begin_mask, 0], (tx[:, 0] + dt)[dx_end_mask])).T
    # filter out anything below min_dx_t
    ivls = ivls[(ivls[:, 1] - ivls[:, 0]) > min_dx_t, :]
    # find which tx samples to use (use the centre of the tx intervals)
    tx_mask, tx_ivl_masks = find_in_intervals(ivls, tx[:, 0] + dt / 2)

    # now get an occupancy map
    occ = np.histogramdd(tx[tx_mask, 1:], bins=bins, range=range, weights=dt[tx_mask])[0]

    # smooth if necessary
    if smooth > 0:
        occ = ndimage.gaussian_filter(occ, sigma=(smooth, smooth), order=0)

    return OccupancyMap(tx, dt, tx_mask, occ, ivls, tx_ivl_masks, bins, smooth, max_dt,
                        min_dx, min_dx_t)


def z_map(occ, z):
    """
    Construct a rate map of the provided point process using the same map
    variables as the provided occupancy map.
    
    Parameters
    ----------
    occ : OccupancyMap
        The occupancy map data to use (see occupancy_map).
    z : ndarray
        A vector of discrete values for the point process.
        
    Returns
    -------
    zmap : ZMap
        A named tuple containing the z map along with the occupancy map and
        parameters used to create it.
        
        occ : OccupancyMap
            The provided occupancy map.
        z : ndarray
            A copy of the provided discrete values for the point process.
        mask : ndarray
            A boolean vector of length z.shape[0] set True for valid
            samples.
        hist : ndarray
            An array of shape occ.bins of the number of z points in each
            bin.
        ivl_masks : ndarray
        
    """
    # interpolate in tx and histogram
    zx = None
    zmap = None
    z_mask = None
    z_ivl_masks = None
    z_mask, z_ivl_masks = find_in_intervals(occ.ivls, z)
    zx = [np.interp(z[z_mask], occ.tx[:, 0], x.flatten())
          for x in np.hsplit(occ.tx[:, 1:], occ.tx.shape[1] - 1)]
    zmap = np.histogramdd(np.vstack(zx).T, bins=occ.bins)[0]

    if occ.smooth > 0:
        zmap = ndimage.gaussian_filter(zmap, (occ.smooth, occ.smooth), order=0)

    return ZMap(occ, z, z_mask, zmap, z_ivl_masks)
