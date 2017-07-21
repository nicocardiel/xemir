from __future__ import division
from __future__ import print_function

import numpy as np

from emirdrp.core import EMIR_NBARS

from emir_definitions import NAXIS1_EMIR
from emir_definitions import NAXIS2_EMIR
from emir_definitions import VALID_GRISMS
from emir_definitions import VALID_FILTERS


class Spec2DImage:
    """Emir2DImage class definition.

    This class handles 2D spectroscopic images corresponding to
    longslits built concatenating the slitlets in the interval ranging
    from 'islitmin' to 'islitmax'.

    Parameters
    ----------
    grism : string
        Grism name. It must be one in VALID_GRISMS.
    spfilter : string
        Filter name. It must be one in VALID_FILTERS.
    csu_configuration : instance of CsuConfiguration class
        CSU configuration parameters.
    dtu_configuration : instance of DtuConfiguration class
        DTU configuration parameters.
    islitmin : int
        Lower slitlet number.
    islitmax : int
        Upper slitlet number.

    Attributes
    ----------
    grism : string
        Grism name. It must be one in VALID_GRISMS.
    spfilter : string
        Filter name. It must be one in VALID_FILTERS.
    csu_configuration : instance of CsuConfiguration class
        CSU configuration parameters.
    dtu_configuration : instance of DtuConfiguration class
        DTU configuration parameters.
    islitmin : int
        Lower slitlet number.
    islitmax : int
        Upper slitlet number.
    bb_nc1_orig : int
        Minimum X coordinate (in pixel units) of the rectangle enclosing
        the 2D image.
    bb_nc2_orig : int
        Maximum X coordinate (in pixel units) of the rectangle enclosing
        the 2D image.
    bb_ns1_orig : int
        Minimum Y coordinate (in pixel units) of the rectangle enclosing
        the 2D image.
    bb_ns2_orig : int
        Maximum Y coordinate (in pixel units) of the rectangle enclosing
        the 2D image.

    """

    def __init__(self, grism, spfilter,
                 csu_configuration,
                 dtu_configuration,
                 islitmin, islitmax):

        # protections
        if grism not in VALID_GRISMS:
            raise ValueError("Grism " + str(grism) + " is not a valid option")
        if spfilter not in VALID_FILTERS:
            raise ValueError("Filter " + str(spfilter) +
                             " is not a valid  option")
        if islitmin < 1:
            raise ValueError("islitmin=" + str(islitmin) +
                             " is outside valid range")
        if islitmax > EMIR_NBARS:
            raise ValueError("islitmax=" + str(islitmax) +
                             " is outside valid range")
        if islitmin > islitmax:
            raise ValueError("islitmin=" + str(islitmin) +
                             " must be lower or equal than istlimax=" +
                             str(islitmax))

        # initial attributes
        self.grism = grism
        self.spfilter = spfilter
        self.csu_configuration = csu_configuration
        self.dtu_configuration = dtu_configuration
        self.islitmin = islitmin
        self.islitmax = islitmax

        # expected boundaries of the rectangle enclosing the 2D image
        if grism == "J" and spfilter == "J":
            poly_bb_ns1 = np.polynomial.Polynomial(
                [-8.03677111e+01,
                 3.98169266e+01,
                 -7.77949391e-02,
                 9.00823598e-04])
            delta_bb_ns2 = 84
        else:
            raise ValueError("Boundaires not yet defined for grism " +
                             str(grism) + " and filter " + str(spfilter))
        self.bb_nc1_orig = 1
        self.bb_nc2_orig = NAXIS1_EMIR
        self.bb_ns1_orig = int(poly_bb_ns1(islitmin) + 0.5)
        self.bb_ns2_orig = int(poly_bb_ns1(islitmax) + 0.5) + delta_bb_ns2
