from __future__ import division
from __future__ import print_function

from emir_definitions import NAXIS2_EMIR


def nscan_minmax(y0_frontier_lower, y0_frontier_upper):
    """Compute valid scan range for provided y0_frontier values.

    Parameters
    ----------
    y0_frontier_lower : float
        Rectified ordinate of the lower frontier.
    y0_frontier_upper : float
        Rectified ordinate of the upper frontier.

    Returns
    -------
    nscan_min : int
        Minimum useful scan for the rectified image.
    nscan_max : int
        Maximum useful scan for the rectified image.

    """

    fraction_pixel = y0_frontier_lower - int(y0_frontier_lower)
    if fraction_pixel > 0.0:
        nscan_min = int(y0_frontier_lower) + 1
    else:
        nscan_min = int(y0_frontier_lower)
    if nscan_min < 1:
        raise ValueError("nscan_min=" + str(nscan_min) + " is < 1")

    fraction_pixel = y0_frontier_upper - int(y0_frontier_upper)
    if fraction_pixel > 0.0:
        nscan_max = int(y0_frontier_upper)
    else:
        nscan_max = int(y0_frontier_upper) - 1
    if nscan_max > NAXIS2_EMIR:
        raise ValueError("nscan_max=" + str(nscan_max) +
                         " is > NAXIS2_EMIR=" + str(NAXIS2_EMIR))

    return nscan_min, nscan_max
