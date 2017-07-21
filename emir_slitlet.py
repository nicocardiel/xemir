from __future__ import division
from __future__ import print_function

from datetime import datetime
from copy import deepcopy
import json
import numpy as np
from numpy.polynomial import Polynomial
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from skimage import transform
from skimage import restoration
import os
import socket
#from skimage import img_as_float

from numina.array.wavecalib.arccalibration import arccalibration
from numina.array.wavecalib.arccalibration import fit_list_of_wvfeatures
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.wavecalib.peaks_spectrum import find_peaks_spectrum
from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection
from numina.array.display.ximplot import ximplot
from numina.array.display.ximshow import ximshow
from numina.array.display.pause_debugplot import DEBUGPLOT_CODES

from emirdrp.core import EMIR_NBARS

from ccd_line import ArcLine
from ccd_line import SpectrumTrail
from numina.array.robustfit import fit_theil_sen
from overplot_bounddict import get_boundaries
from rescale_array_to_z1z2 import rescale_array_to_z1z2
from rescale_array_to_z1z2 import rescale_array_from_z1z2

from emir_definitions import NAXIS1_EMIR
from emir_definitions import NAXIS2_EMIR
from emir_definitions import VALID_GRISMS
from emir_definitions import VALID_FILTERS


class EmirSlitlet:
    """EMIR slitlet definition.

    Parameters
    ----------
    grism_name : string
        Grism name. It must be one of the following: 'J', 'H', 'K'.
    filter_name : string
        Filter name.
    rotang : float
        Rotator position angle in degrees.
    xdtu: float
        XDTU
    ydtu: float
        YDTU
    xdtu_0: float
        XDTU_0
    ydtu_0: float
        YDTU_0
    slitlet_number: int
        Number of the slitlet.
    fits_file_name : string
        File name of the original FITS file.
    date_obs : string
        Keyword DATE-OBS from FITS header.
        csu_bar_left : float
        Location of the border of the left bar (mm), measured from the
        left border.
    csu_bar_right : float
        Location of the border of the right bar (mm), measured from the
        right border.
    csu_bar_slit_center : float
        Location of the center of the slit (mm) as defined by the left
        and right bars. Note that since csup_left and csup_right have
        different origins, it is necessary to take into account that
        the separation between both origins is 341 mm. Thus:
        csu_bar_slit_center = (csu_bar_left + (341-csu_bar_right) )/ 2.
    csu_bar_slit_width : float
        Width of the slit (mm) determined by the left and right bars,
        computed as:
        csu_bar_slit_width = (341-csu_bar_right) - csu_bar_left.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. See code explanation in
        numina.array.display.pause_debugplot.py

    Attributes
    ----------
    grism_name : string
        Grism name.
    filter_name : string
        Filter name.
    rotang : float
        Rotator position angle in degrees.
    slitlet_number: int
        Number of the slitlet.
    ffits_file_name : string
        Full path and file name of the original FITS files.
    fits_file_name : string
        File name of the original FITS file.
    date_obs : string
        Keyword DATE-OBS from FITS header.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses
    deg_boundary : int
        Degree of polynomial to fit upper/lower boundaries.
    bb_nc1_orig : int
        Minimum X coordinate of the enclosing bounding box (in pixel
        units) in the original image.
    bb_nc2_orig : int
        Maximum X coordinate of the enclosing bounding box (in pixel
        units) in the original image.
    bb_ns1_orig : int
        Minimum Y coordinate of the enclosing bounding box (in pixel
        units) in the original image.
    bb_ns2_orig : int
        Maximum Y coordinate of the enclosing bounding box (in pixel
        units) in the original image.
    csu_bar_left : float
        Location of the border of the left bar (mm), measured from the
        left border.
    csu_bar_right : float
        Location of the border of the right bar (mm), measured from the
        right border.
    csu_bar_slit_center : float
        Location of the center of the slit (mm) as defined by the left
        and right bars. Note that since csup_left and csup_right have
        different origins, it is necessary to take into account that
        the separation between both origins is 341 mm. Thus:
        csu_bar_slit_center = (csu_bar_left + (341-csu_bar_right) )/ 2.
    csu_bar_slit_width : float
        Width of the slit (mm) determined by the left and right bars,
        computed as:
        csu_bar_slit_width = (341-csu_bar_right) - csu_bar_left.
    naxis1_slitlet : int
        NAXIS1 for slitlet, computed from bounding box.
    naxis2_slitlet : int
        NAXIS2 for slitlet, computed from bounding box.
    crpix1_enlarged : float
        CRPIX1 for enlarged image (i.e. the final wavelength calibrated
        image with a common sampling for all the slitlets).
    crval1_enlarged : float
        CRVAL1 for enlarged image (i.e. the final wavelength calibrated
        image with a common sampling for all the slitlets).
    cdelt1_enlarged : float
        CDELT1 for enlarged image (i.e. the final wavelength calibrated
        image with a common sampling for all the slitlets)
    naxis1_enlarged : float
        NAXIS1 for enlarged image (i.e. the final wavelength calibrated
        image with a common sampling for all the slitlets)
    crmin1_enlarged : float
        CRVAL value for first pixel in enlarged image (i.e. the final
        wavelength calibrated image with a common sampling for all the
        slitlets). Note that this value is employed as wv_ini_search
        in the automatic wavelength calibration procedure.
    crmax1_enlarged : float
        CRVAL value for last pixel in enlarged image (i.e. the final
        wavelength calibrated image with a common sampling for all the
        slitlets). Note that this value is employed as wv_end_search
        in the automatic wavelength calibration procedure.
    crpix1_slitlet : float
        CRPIX1 for slitlet using the wavelength calibration with a
        common sampling for all the slitlets.
    crval1_slitlet : float
        CRVAL1 for slitlet using the wavelength calibration with a
        common sampling for all the slitlets.
    cdelt1_slitlet : float
        CDELT1 for slitlet using the wavelength calibration with a
        common sampling for all the slitlets.
    crmin1_slitlet : float
        CRVAL value for the first slitlet pixel, using the wavelength
        calibration with a common sampling for all the slitlets.
    crmax1_slitlet : float
        CRVAL value for the last slitlet pixel, using the wavelength
        calibration with a common sampling for all the slitlets.
    xmin_lower_boundary_fit : int
        Minimum value to fit the lower boundary.
    xmax_lower_boundary_fit : int
        Maximum value to fit the lower boundary.
    xmin_upper_boundary_fit : int
        Minimum value to fit the upper boundary.
    xmax_upper_boundary_fit : int
        Maximum value to fit the upper boundary.
    list_arc_lines : list of ArcLine instances
        Identified arc lines.
    list_spectrum_trails: list of SpectrumTrail instances
        Identified spectrum trails.
    x_inter_orig : 1d numpy array, float
        X coordinates of the intersection points of arc lines with
        spectrum trails in the original image.
    y_inter_orig : 1d numpy array, float
        Y coordinates of the intersection points of arc lines with
        spectrum trails in the original image.
    x_inter_rect : 1d numpy array, float
        X coordinates of the intersection points of arc lines with
        spectrum trails in the rectified image.
    y_inter_rect : 1d numpy array, float
        Y coordinates of the intersection points of arc lines with
        spectrum trails in the rectified image.
    x0_reference : float
        X coordinate of the reference points for the rectified spectrum
        trails. The same value is used for all the available spectrum
        trails.
    y0_reference : 1d numpy array, float
        Y coordinates of the reference points for the rectified
        spectrum trails. A different value is used for each available
        spectrum trail.
    ttd : transform.PolynomialTransform instance
        Direct transformation to rectify the image.
    solution_wv : instance of SolutionArcCalibration
        Wavelength calibration solution.

    """

    def __init__(self, grism_name=None,
                 filter_name=None,
                 rotang=None,
                 xdtu=None,
                 ydtu=None,
                 xdtu_0=None,
                 ydtu_0=None,
                 slitlet_number=None,
                 fits_file_name=None,
                 date_obs=None,
                 csu_bar_left=None,
                 csu_bar_right=None,
                 csu_bar_slit_center=None,
                 csu_bar_slit_width=None,
                 debugplot=0):
        """Initialize slitlet with enclosing bounding box."""

        # protections
        if grism_name is None:
            raise ValueError("grism_name is undefined")
        if type(grism_name) is not str:
            raise ValueError("grism_name is not a valid string")
        elif grism_name not in VALID_GRISMS:
            raise ValueError("grism_name is not in valid grism list")
        if filter_name is None:
            raise ValueError("filter_name is undefined")
        if type(filter_name) is not str:
            raise ValueError("filter_name is not a valid string")
        elif filter_name not in VALID_FILTERS:
            raise ValueError("grism_name is not in valid filter list")
        if slitlet_number is None:
            raise ValueError("slitlet_number is undefined")

        if type(slitlet_number) not in [np.int, np.int64]:
            raise ValueError("slitlet_number " + str(slitlet_number) +
                             " is not a valid integer")
        if 1 <= slitlet_number <= EMIR_NBARS:
            pass
        else:
            raise ValueError("slitlet number " + str(slitlet_number) +
                             " is outside valid range for EMIR")

        if debugplot not in DEBUGPLOT_CODES:
            raise ValueError("debugplot " + str(debugplot) +
                             " is not a valid integer")

        # store initializing values
        self.grism_name = grism_name
        self.filter_name = filter_name
        self.rotang = rotang
        self.xdtu = xdtu
        self.ydtu = ydtu
        self.xdtu_0 = xdtu_0
        self.ydtu_0 = ydtu_0
        self.slitlet_number = slitlet_number
        self.ffits_file_name = fits_file_name  # full path
        self.fits_file_name = os.path.basename(fits_file_name)  # path removed
        self.date_obs = date_obs
        self.debugplot = debugplot

        # enclosing bounding box (in pixel units) in the original image
        self.bb_nc1_orig = None
        self.bb_nc2_orig = None
        self.bb_ns1_orig = None
        self.bb_ns2_orig = None

        # location of the CSU bars and related information (mm)
        self.csu_bar_left = csu_bar_left
        self.csu_bar_right = csu_bar_right
        self.csu_bar_slit_center = csu_bar_slit_center
        self.csu_bar_slit_width = csu_bar_slit_width

        # wavelength calibration parameters relative to slitlet
        # dimension
        self.crpix1_slitlet = None
        self.crval1_slitlet = None
        self.cdelt1_slitlet = None
        self.crmin1_slitlet = None
        self.crmax1_slitlet = None

        # TODO: this must updated with the new GRISM + FILTER combinations
        self.crpix1_enlarged = 1.0  # center of first pixel
        if grism_name == "J" and filter_name == "J":
            self.crval1_enlarged = 11000.000  # Angstroms
            self.cdelt1_enlarged = 0.7575  # Angstroms/pixel
            self.naxis1_enlarged = 4134  # pixels
        elif grism_name == "H" and filter_name == "H":
            self.crval1_enlarged = 14000.000  # Angstroms
            self.cdelt1_enlarged = 1.2000  # Angstroms/pixel
            self.naxis1_enlarged = 4134  # pixels
        elif grism_name == "K" and filter_name == "Ksp":
            self.crval1_enlarged = 19000.000  # Angstroms
            self.cdelt1_enlarged = 1.7000  # Angstroms/pixel
            self.naxis1_enlarged = 4134  # pixels
        elif grism_name == "LR" and filter_name == "YJ":
            self.crval1_enlarged = None  # Angstroms
            self.cdelt1_enlarged = None  # Angstroms/pixel
            self.naxis1_enlarged = None  # pixels
        elif grism_name == "LR" and filter_name == "HK":
            self.crval1_enlarged = None  # Angstroms
            self.cdelt1_enlarged = None  # Angstroms/pixel
            self.naxis1_enlarged = None  # pixels
        else:
            raise ValueError("invalid grism_name and/or filter_name")

        self.crmin1_enlarged = \
            self.crval1_enlarged + \
            (1.0 - self.crpix1_enlarged) * \
            self.cdelt1_enlarged  # Angstroms
        self.crmax1_enlarged = \
            self.crval1_enlarged + \
            (self.naxis1_enlarged - self.crpix1_enlarged) * \
            self.cdelt1_enlarged  # Angstroms

        # ToDo: remove the following definitions
        """
        self.bb_nc1_orig = 1
        self.bb_nc2_orig = NAXIS1_EMIR

        if slitlet_number == 2:
            self.bb_ns1_orig = 1
            self.bb_ns2_orig = 80
        elif slitlet_number == 3:
            self.bb_ns1_orig = 36
            self.bb_ns2_orig = 120
        elif slitlet_number == 4:
            self.bb_ns1_orig = 76
            self.bb_ns2_orig = 160
        elif slitlet_number == 5:
            self.bb_ns1_orig = 116
            self.bb_ns2_orig = 200
        elif slitlet_number == 6:
            self.bb_ns1_orig = 156
            self.bb_ns2_orig = 240
        elif slitlet_number == 7:
            self.bb_ns1_orig = 196
            self.bb_ns2_orig = 280
        elif slitlet_number == 8:
            self.bb_ns1_orig = 236
            self.bb_ns2_orig = 320
        elif slitlet_number == 9:
            self.bb_ns1_orig = 271
            self.bb_ns2_orig = 355
        elif slitlet_number == 10:
            self.bb_ns1_orig = 311
            self.bb_ns2_orig = 395
        elif slitlet_number == 11:
            self.bb_ns1_orig = 351
            self.bb_ns2_orig = 435
        elif slitlet_number == 12:
            self.bb_ns1_orig = 386
            self.bb_ns2_orig = 470
        elif slitlet_number == 13:
            self.bb_ns1_orig = 426
            self.bb_ns2_orig = 510
        elif slitlet_number == 14:
            self.bb_ns1_orig = 466
            self.bb_ns2_orig = 550
        elif slitlet_number == 15:
            self.bb_ns1_orig = 506
            self.bb_ns2_orig = 590
        elif slitlet_number == 16:
            self.bb_ns1_orig = 541
            self.bb_ns2_orig = 625
        elif slitlet_number == 17:
            self.bb_ns1_orig = 581
            self.bb_ns2_orig = 665
        elif slitlet_number == 18:
            self.bb_ns1_orig = 616
            self.bb_ns2_orig = 700
        elif slitlet_number == 19:
            self.bb_ns1_orig = 651
            self.bb_ns2_orig = 735
        elif slitlet_number == 20:
            self.bb_ns1_orig = 691
            self.bb_ns2_orig = 775
        elif slitlet_number == 21:
            self.bb_ns1_orig = 726
            self.bb_ns2_orig = 810
        elif slitlet_number == 22:
            self.bb_ns1_orig = 766
            self.bb_ns2_orig = 850
        elif slitlet_number == 23:
            self.bb_ns1_orig = 806
            self.bb_ns2_orig = 890
        elif slitlet_number == 24:
            self.bb_ns1_orig = 841
            self.bb_ns2_orig = 925
        elif slitlet_number == 25:
            self.bb_ns1_orig = 881
            self.bb_ns2_orig = 965
        elif slitlet_number == 26:
            self.bb_ns1_orig = 916
            self.bb_ns2_orig = 1000
        elif slitlet_number == 27:  # not definitive
            self.bb_ns1_orig = 956
            self.bb_ns2_orig = 1040
        elif slitlet_number == 28:  # not definitive
            self.bb_ns1_orig = 996
            self.bb_ns2_orig = 1080
        elif slitlet_number == 29:  # not definitive
            self.bb_ns1_orig = 1031
            self.bb_ns2_orig = 1115
        elif slitlet_number == 30:
            self.bb_ns1_orig = 1071
            self.bb_ns2_orig = 1155
        elif slitlet_number == 31:
            self.bb_ns1_orig = 1106
            self.bb_ns2_orig = 1190
        elif slitlet_number == 32:
            self.bb_ns1_orig = 1146
            self.bb_ns2_orig = 1230
        elif slitlet_number == 33:
            self.bb_ns1_orig = 1181
            self.bb_ns2_orig = 1265
        elif slitlet_number == 34:
            self.bb_ns1_orig = 1216
            self.bb_ns2_orig = 1300
        elif slitlet_number == 35:
            self.bb_ns1_orig = 1256
            self.bb_ns2_orig = 1340
        elif slitlet_number == 36:
            self.bb_ns1_orig = 1296
            self.bb_ns2_orig = 1380
        elif slitlet_number == 37:
            self.bb_ns1_orig = 1331
            self.bb_ns2_orig = 1415
        elif slitlet_number == 38:
            self.bb_ns1_orig = 1371
            self.bb_ns2_orig = 1455
        elif slitlet_number == 39:
            self.bb_ns1_orig = 1406
            self.bb_ns2_orig = 1490
        elif slitlet_number == 40:
            self.bb_ns1_orig = 1446
            self.bb_ns2_orig = 1530
        elif slitlet_number == 41:
            self.bb_ns1_orig = 1481
            self.bb_ns2_orig = 1565
        elif slitlet_number == 42:
            self.bb_ns1_orig = 1521
            self.bb_ns2_orig = 1605
        elif slitlet_number == 43:
            self.bb_ns1_orig = 1561
            self.bb_ns2_orig = 1645
        elif slitlet_number == 44:
            self.bb_ns1_orig = 1601
            self.bb_ns2_orig = 1685
        elif slitlet_number == 45:
            self.bb_ns1_orig = 1636
            self.bb_ns2_orig = 1720
        elif slitlet_number == 46:
            self.bb_ns1_orig = 1676
            self.bb_ns2_orig = 1760
        elif slitlet_number == 47:
            self.bb_ns1_orig = 1711
            self.bb_ns2_orig = 1795
        elif slitlet_number == 48:
            self.bb_ns1_orig = 1751
            self.bb_ns2_orig = 1835
        elif slitlet_number == 49:
            self.bb_ns1_orig = 1791
            self.bb_ns2_orig = 1875
        elif slitlet_number == 50:
            self.bb_ns1_orig = 1826
            self.bb_ns2_orig = 1910
        elif slitlet_number == 51:
            self.bb_ns1_orig = 1866
            self.bb_ns2_orig = 1950
        elif slitlet_number == 52:
            self.bb_ns1_orig = 1906
            self.bb_ns2_orig = 1990
        elif slitlet_number == 53:
            self.bb_ns1_orig = 1946
            self.bb_ns2_orig = 2030
        elif slitlet_number == 54:
            self.bb_ns1_orig = 1986
            self.bb_ns2_orig = 2048
        else:
            raise ValueError("slitlet number " + str(slitlet_number) +
                             " is not defined")
        """

        # approximate function providing scan number corresponding
        # to the lower boundary of the bounding box
        def ns1_expected(nslit):
            value = -8.03677111e+01 + 3.98169266e+01 * nslit \
                - 7.77949391e-02 * nslit ** 2 \
                + 9.00823598e-04 * nslit ** 3
            return int(value + 0.5)

        self.bb_nc1_orig = 1
        self.bb_nc2_orig = 2048
        self.bb_ns1_orig = ns1_expected(slitlet_number)
        self.bb_ns2_orig = self.bb_ns1_orig + 84
        if self.bb_ns1_orig < 1:
            self.bb_ns1_orig = 1
        if self.bb_ns2_orig < 1:
            self.bb_ns2_orig = 1
        if self.bb_ns1_orig > 2048:
            self.bb_ns1_orig = 2048
        if self.bb_ns2_orig > 2048:
            self.bb_ns2_orig = 2048

        # # read bounding boxes for all the slitlets
        # json_filename = "bbox_slitlets.json"
        # main_label = "bbox_slitlets"
        # if os.path.isfile(json_filename):
        #     bbdict = json.loads(open(json_filename).read())
        #     if bbdict.keys() != [main_label]:
        #         raise ValueError("Invalid initial key in " + json_filename)
        # else:
        #     raise ValueError("File " + json_filename + " not found!")
        # # print(json.dumps(bbdict, indent=4, sort_keys=True))
        # slitlet_label = "slitlet" + str(slitlet_number).zfill(2)
        # if slitlet_label in bbdict[main_label].keys():
        #     self.bb_nc1_orig = bbdict[main_label][slitlet_label]['bb_nc1_orig']
        #     self.bb_nc2_orig = bbdict[main_label][slitlet_label]['bb_nc2_orig']
        #     self.bb_ns1_orig = bbdict[main_label][slitlet_label]['bb_ns1_orig']
        #     self.bb_ns2_orig = bbdict[main_label][slitlet_label]['bb_ns2_orig']
        # else:
        #     raise ValueError("slitlet number " + str(slitlet_number) +
        #                      " is not defined")

        # slitlet dimensions, computed from bounding box
        self.naxis1_slitlet = self.bb_nc2_orig - self.bb_nc1_orig + 1
        self.naxis2_slitlet = self.bb_ns2_orig - self.bb_ns1_orig + 1

        # ranges to fit boundaries
        self.xmin_lower_boundary_fit = 1
        self.xmax_lower_boundary_fit = NAXIS1_EMIR
        self.xmin_upper_boundary_fit = 1
        self.xmax_upper_boundary_fit = NAXIS1_EMIR

        # specific corrections for each grism + filter combination
        if grism_name == "J" and filter_name == "J":
            self.deg_boundary = 5
            self.bb_ns1_orig += 0
            self.bb_ns2_orig += 0
            if slitlet_number == 54:
                self.xmin_upper_boundary_fit = 400
                self.xmax_upper_boundary_fit = 1750
        elif grism_name == "H" and filter_name == "H":
            self.deg_boundary = 0
            self.bb_ns1_orig += 4
            self.bb_ns2_orig += 4
            if slitlet_number == 54:
                raise ValueError("slitlet number " + str(slitlet_number) +
                                 " is not defined")
        elif grism_name == "K" and filter_name == "Ksp":
            self.deg_boundary = 0
            self.bb_ns1_orig += 0
            self.bb_ns2_orig += 0
        elif grism_name == "LR" and filter_name == "YJ":
            self.deg_boundary = 0
            self.bb_ns1_orig -= 90
            if self.bb_ns1_orig < 1:
                self.bb_ns1_orig = 1
            self.bb_ns2_orig -= 90
            if self.bb_ns2_orig < 1:
                self.bb_ns2_orig = 1
            if slitlet_number == 2:
                raise ValueError("slitlet number " + str(slitlet_number) +
                                 " is not defined")
        elif grism_name == "LR" and filter_name == "HK":
            self.deg_boundary = 2
            self.bb_ns1_orig -= 90
            self.bb_ns2_orig -= 90
            if slitlet_number == 2:
                raise ValueError("slitlet number " + str(slitlet_number) +
                                 " is not defined")
        else:
            raise ValueError("invalid grism_name and/or filter_name")

        # list of associated arc lines
        self.list_arc_lines = None
        # list of associated spectrum trails
        self.list_spectrum_trails = None
        # intersection points of arc lines with spectrum trails in the
        # original image
        self.x_inter_orig = None
        self.y_inter_orig = None
        # intersection points of arc lines with spectrum trails in the
        # rectified image
        self.x_inter_rect = None
        self.y_inter_rect = None
        # reference points for rectified spectrum trails
        self.x0_reference = float(NAXIS1_EMIR) / 2.0 + 0.5  # single float
        self.y0_reference = None  # different value for each spectrum trail
        # transformations to rectify slitlet
        self.ttd = None  # direct
        # wavelength calibration solution
        self.solution_wv = None

    def __repr__(self):
        """Define printable representation of a EmirSlitlet instance."""

        # list of associated arc lines
        if self.list_arc_lines is None:
            number_arc_lines = None
        else:
            number_arc_lines = len(self.list_arc_lines)

        # list of associated spectrum trails
        if self.list_spectrum_trails is None:
            number_spectrum_trails = None
        else:
            number_spectrum_trails = len(self.list_spectrum_trails)

        # intersection points (original image)
        if self.x_inter_orig is None:
            str_x_inter_orig = "None"
        else:
            str_x_inter_orig = "[" + str(self.x_inter_orig[0]) + ",..."
        if self.y_inter_orig is None:
            str_y_inter_orig = "None"
        else:
            str_y_inter_orig = "[" + str(self.y_inter_orig[0]) + ",..."

        # intersection points (rectified image)
        if self.x_inter_rect is None:
            str_x_inter_rect = "None"
        else:
            str_x_inter_rect = "[" + str(self.x_inter_rect[0]) + ",..."
        if self.y_inter_rect is None:
            str_y_inter_rect = "None"
        else:
            str_y_inter_rect = "[" + str(self.y_inter_rect[0]) + ",..."

        # transformation to rectify image
        if self.ttd is None:
            str_ttd_params_0 = "None"
            str_ttd_params_1 = "None"
        else:
            str_ttd_params_0 = str(self.ttd.params[0])
            str_ttd_params_1 = str(self.ttd.params[1])

        # return string with all the information
        return "<EmirSlilet instance>\n" + \
            "- FITS file name.....: " + self.fits_file_name + "\n" + \
            "- DATE_OBS...........: " + self.date_obs + "\n" + \
            "- GRISM..............: " + str(self.grism_name) + "\n" + \
            "- FILTER.............: " + str(self.filter_name) + "\n" + \
            "- ROTANG.............: " + str(self.rotang) + "\n" + \
            "- XDTU...............: " + str(self.xdtu) + "\n" + \
            "- YDTU...............: " + str(self.ydtu) + "\n" + \
            "- XDTU_0.............: " + str(self.xdtu_0) + "\n" + \
            "- YDTU_0.............: " + str(self.ydtu_0) + "\n" + \
            "- slitlet number.....: " + str(self.slitlet_number) + "\n" + \
            "- CRPIX1_slitlet.....: " + str(self.crpix1_slitlet) + "\n" + \
            "- CRVAL1_slitlet.....: " + str(self.crval1_slitlet) + "\n" + \
            "- CDELT1_slitlet.....: " + str(self.cdelt1_slitlet) + "\n" + \
            "- CRMIN1_slitlet.....: " + str(self.crmin1_slitlet) + "\n"  \
            "- CRMAX1_slitlet.....: " + str(self.crmax1_slitlet) + "\n" + \
            "- CRPIX1_enlarged....: " + str(self.crpix1_enlarged) + "\n" + \
            "- CRVAL1_enlarged....: " + str(self.crval1_enlarged) + "\n" + \
            "- CDELT1_enlarged....: " + str(self.cdelt1_enlarged) + "\n" + \
            "- NAXIS1_enlarged....: " + str(self.naxis1_enlarged) + "\n" + \
            "- CRMIN1_enlarged....: " + str(self.crmin1_enlarged) + "\n" + \
            "- CRMAX1_enlarged....: " + str(self.crmax1_enlarged) + "\n" + \
            "- deb_boundary.......: " + str(self.deg_boundary) + "\n" + \
            "- xmin_lower_boundary_fit: " + \
               str(self.xmin_lower_boundary_fit) + "\n" + \
            "- xmax_lower_boundary_fit: " + \
               str(self.xmax_lower_boundary_fit) + "\n" + \
            "- xmin_upper_boundary_fit: " + \
               str(self.xmin_upper_boundary_fit) + "\n" + \
            "- xmax_upper_boundary_fit: " + \
               str(self.xmax_upper_boundary_fit) + "\n" + \
            "- bb_nc1_orig........: " + str(self.bb_nc1_orig) + "\n" + \
            "- bb_nc2_orig........: " + str(self.bb_nc2_orig) + "\n" + \
            "- bb_ns1_orig........: " + str(self.bb_ns1_orig) + "\n" + \
            "- bb_ns2_orig........: " + str(self.bb_ns2_orig) + "\n" + \
            "- csu_bar_left.......: " + str(self.csu_bar_left) + "\n" + \
            "- csu_bar_right......: " + str(self.csu_bar_right) + "\n" + \
            "- csu_bar_slit_center: " + str(self.csu_bar_slit_center) + \
               "\n" + \
            "- csu_bar_slit_width.: " + str(self.csu_bar_slit_width) + "\n" + \
            "- naxis1_slitlet.....: " + str(self.naxis1_slitlet) + "\n" + \
            "- naxis2_slitlet.....: " + str(self.naxis2_slitlet) + "\n" + \
            "- number of associated arc lines......: " + \
               str(number_arc_lines) + "\n" + \
            "- number of associated spectrum trails: " + \
               str(number_spectrum_trails) + "\n" + \
            "- x0_reference.......: " + str(self.x0_reference) + "\n" + \
            "- y0_reference.......: " + str(self.y0_reference) + "\n" + \
            "- x_inter_orig.......: " + str_x_inter_orig + "\n" + \
            "- y_inter_orig.......: " + str_y_inter_orig + "\n" + \
            "- x_inter_rect.......: " + str_x_inter_rect + "\n" + \
            "- y_inter_rect.......: " + str_y_inter_rect + "\n" + \
            "- ttd_params[0]......:\n\t" + str_ttd_params_0 + "\n" + \
            "- ttd_params[1]......:\n\t" + str_ttd_params_1 + "\n" + \
            "- Wavelength cal.....: " + str(self.solution_wv)

    def _check_2k2k(self, image_2k2k):
        """Check that the image corresponds to the 2k x 2k EMIR format.

        In addition, this function verifies that the slitlet bounding
        box fits within the original image dimensions.

        Parameters
        ----------
        image_2k2k : 2d numpy array, float
            Image to be displayed (dimensions NAXIS1 * NAXIS2)

        """

        if type(image_2k2k) is not np.ndarray:
            raise ValueError("image_2k2k=" + str(image_2k2k) +
                             " must be a numpy.ndarray")
        elif image_2k2k.ndim is not 2:
            raise ValueError("image_2k2k.ndim=" + str(image_2k2k.dim) +
                             " must be 2")

        naxis2, naxis1 = image_2k2k.shape

        if naxis1 != NAXIS1_EMIR:
            raise ValueError("NAXIS1=" + str(naxis1) +
                             " of image_2k2k is not " + str(NAXIS1_EMIR))
        if naxis2 != NAXIS2_EMIR:
            raise ValueError("NAXIS2=" + str(naxis2) +
                             " of image_2k2k is not " + str(NAXIS2_EMIR))

        # duplicate variables with shorter names
        nc1 = self.bb_nc1_orig
        nc2 = self.bb_nc2_orig
        ns1 = self.bb_ns1_orig
        ns2 = self.bb_ns2_orig

        # check that the slitlet bounding box fit within the original
        # EMIR image
        if 1 <= nc1 <= nc2 <= naxis1:
            pass
        else:
            raise ValueError("slitlet bounding box outside valid NAXIS1 range")
        if 1 <= ns1 <= ns2 <= naxis2:
            pass
        else:
            raise ValueError("slitlet bounding box outside valid NAXIS2 range")

    def _check_slitlet2d(self, slitlet2d):
        """Check that slitlet2d is a valid image, with the expected size.

        Parameters
        ----------
        slitlet2d : 2d numpy array, float
            Image corresponding to the slitlet bounding box dimensions.

        """

        if type(slitlet2d) is not np.ndarray:
            raise ValueError("slitlet2d=" + str(slitlet2d) +
                             " must be a numpy.ndarray")
        elif slitlet2d.ndim is not 2:
            raise ValueError("slitlet2d.ndim=" + str(slitlet2d.dim) +
                             " must be 2")

        # duplicate variables with shorter names
        nc1 = self.bb_nc1_orig
        nc2 = self.bb_nc2_orig
        ns1 = self.bb_ns1_orig
        ns2 = self.bb_ns2_orig

        # check that the slitlet dimensions correspond to the expected
        # values
        naxis2_, naxis1_ = slitlet2d.shape
        if naxis1_ != nc2 - nc1 + 1:
            raise ValueError("invalid slitlet2d dimensions")
        if naxis2_ != ns2 - ns1 + 1:
            raise ValueError("invalid slitlet2d dimensions")

    def extract_slitlet2d(self, image_2k2k):
        """Extract slitlet 2d image from image with original EMIR dimensions.

        Parameters
        ----------
        image_2k2k : 2d numpy array, float
            Original image (dimensions NAXIS1 * NAXIS2)

        Returns
        -------
        slitlet2d : 2d numpy array, float
            Image corresponding to the slitlet region defined by its
            bounding box.

        """

        # protections
        self._check_2k2k(image_2k2k)

        # extract slitlet region
        slitlet2d = image_2k2k[(self.bb_ns1_orig - 1):self.bb_ns2_orig,
                               (self.bb_nc1_orig - 1):self.bb_nc2_orig]

        # transform to float
        slitlet2d = slitlet2d.astype(np.float)

        # return slitlet image
        return slitlet2d

    def define_spectrails_from_boundaries(self, slitlet2d):
        """Define middle, lower and upper spectrum trails from boundaries.

        Parameters
        ----------
        slitlet2d : 2d numpy array, float
            Image containing the slitlet bounding box.

        """

        # get boundaries
        pol_lower_boundary, pol_upper_boundary = get_boundaries(
            self.grism_name, self.filter_name, self.slitlet_number,
            self.csu_bar_slit_center,
            nsampling=100, deg_boundary=4)
        if pol_lower_boundary is None:
            raise ValueError("Unexpected pol_lower_boundary=None")
        if pol_upper_boundary is None:
            raise ValueError("Unexpected pol_upper_boundary=None")

        # polynomial degree employed in boundaries
        poldeg_lower = len(pol_lower_boundary.coef) - 1
        poldeg_upper = len(pol_upper_boundary.coef) - 1
        if poldeg_lower != poldeg_upper:
            raise ValueError("Unexepcted poldeg_lower != poldeg_upper")
        else:
            poldeg = poldeg_lower

        # evaluate polynomials
        xp = np.linspace(start=1, stop=NAXIS1_EMIR, num=1000)
        yp_lower = pol_lower_boundary(xp)
        yp_upper = pol_upper_boundary(xp)
        yp_middle = (yp_lower + yp_upper) / 2.0

        # define middle spectrum trail
        middle_spectrail = SpectrumTrail()  # declare SpectrumTrail instance
        middle_spectrail.fit(x=xp, y=yp_middle, deg=poldeg)
        # initialize list with spectrum trails, being the first element
        # of that list the final middle spectrum trail
        self.list_spectrum_trails = [middle_spectrail]
        # evaluate the middle spectrum trail at a predefined
        # abscissa (self.x0_reference); note that this value is stored
        # in a numpy array (self.y0_reference) initially with only one
        # element since additional values, corresponding to additional
        # spectrum trails, will be appended to it
        middle_spectrail.y_rectified = \
            middle_spectrail.poly_funct(self.x0_reference)
        self.y0_reference = np.array([middle_spectrail.y_rectified])

        # add lower boundary as an additional spectrum trail
        lower_spectrail = SpectrumTrail()  # declare SpectrumTrail instance
        lower_spectrail.fit(x=xp, y=yp_lower, deg=poldeg)
        self.list_spectrum_trails.append(lower_spectrail)
        # update list of spectrum trail y0_reference values
        y_rectified = lower_spectrail.poly_funct(self.x0_reference)
        self.y0_reference = np.append(self.y0_reference, y_rectified)
        # update y0_rectified in the new spectrum trail itself
        lower_spectrail.y_rectified = y_rectified

        # add upper boundary as an additional spectrum trail
        upper_spectrail = SpectrumTrail()  # declare SpectrumTrail instance
        upper_spectrail.fit(x=xp, y=yp_upper, deg=poldeg)
        self.list_spectrum_trails.append(upper_spectrail)
        # update list of spectrum trail y0_reference values
        y_rectified = upper_spectrail.poly_funct(self.x0_reference)
        self.y0_reference = np.append(self.y0_reference, y_rectified)
        # update y0_rectified in the new spectrum trail itself
        upper_spectrail.y_rectified = y_rectified

        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            # display image with zscale cuts
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) +"]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (define_spectrails_from_boundaries)"
            ax = ximshow(slitlet2d, title=title,
                         image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                     self.bb_ns1_orig, self.bb_ns2_orig),
                         show=False)
            # overplot boundaries for current slitlet
            ax.plot(xp, yp_lower, 'g-')
            ax.plot(xp, yp_upper, 'b-')
            ax.plot(xp, yp_middle, 'r-')
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

    def locate_unknown_arc_lines(self, slitlet2d,
                                 times_sigma_threshold=4,
                                 minimum_threshold=None,
                                 delta_x_max=30,
                                 delta_y_min=37,
                                 deg_middle_spectrail=2,
                                 dist_middle_min=15,
                                 dist_middle_max=28):
        """Determine the location of unknown arc lines in slitlet.

        This function also computes the middle spectrum trail.

        Parameters
        ----------
        slitlet2d : 2d numpy array, float
            Image containing the slitlet bounding box.
        times_sigma_threshold : float
            Times (robust) sigma above the median of the image to look
            for arc lines.
        minimum_threshold : float or None
            Minimum threshold to look for arc lines.
        delta_x_max : float
            Maximum size of potential arc line in the X direction.
        delta_y_min : float
            Minimum size of potential arc line in the Y direction.
        deg_middle_spectrail : int
            Degree of the polynomial describing the middle spectrum
            trail.
        dist_middle_min : float
            Minimum Y distance from the middle spectrum trail to the
            extreme of the potential arc line.
        dist_middle_max : float
            Maximum Y distance from the middle spectrum trail to the
            extreme of the potential arc line.

        """

        # protections
        self._check_slitlet2d(slitlet2d)

        # smooth denoising of slitlet2d
        slitlet2d_rs, coef_rs = rescale_array_to_z1z2(slitlet2d, z1z2=(-1, 1))
        slitlet2d_dn = restoration.denoise_nl_means(slitlet2d_rs,
                                                    patch_size=3,
                                                    patch_distance=2)
        slitlet2d_dn = rescale_array_from_z1z2(slitlet2d_dn, coef_rs)

        # compute basic statistics
        q25, q50, q75 = np.percentile(slitlet2d_dn, q=[25.0, 50.0, 75.0])
        sigmag = 0.7413 * (q75 - q25)  # robust standard deviation
        if self.debugplot >= 10:
            q16, q84 = np.percentile(slitlet2d_dn, q=[15.87, 84.13])
            print('>>> q16...:', q16)
            print('>>> q25...:', q25)
            print('>>> q50...:', q50)
            print('>>> q75...:', q75)
            print('>>> q84...:', q84)
            print('>>> sigmaG:', sigmag)
        if self.debugplot % 10 != 0:
            # display initial image with zscale cuts
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (locate_unknown_arc_lines #1)"
            ximshow(slitlet2d, title=title,
                    image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                self.bb_ns1_orig, self.bb_ns2_orig),
                    debugplot=self.debugplot)
            # display denoised image with zscale cuts
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (locate_unknown_arc_lines #2)"
            ximshow(slitlet2d_dn, title=title,
                    image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                self.bb_ns1_orig, self.bb_ns2_orig),
                    debugplot=self.debugplot)
            # display image with different cuts
            z1z2 = (q50 + times_sigma_threshold * sigmag,
                    q50 + 2 * times_sigma_threshold * sigmag)
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (locate_unknown_arc_lines #3)"
            ximshow(slitlet2d_dn, title=title, z1z2=z1z2,
                    image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                self.bb_ns1_orig, self.bb_ns2_orig),
                    debugplot=self.debugplot)

        # determine threshold (using the maximum of q50 + t *sigmag or
        # minimum_threshold)
        threshold = q50 + times_sigma_threshold * sigmag
        if minimum_threshold is not None:
            if minimum_threshold > threshold:
                threshold = minimum_threshold

        # identify objects in slitlet2d above threshold
        labels2d_objects, no_objects = ndimage.label(slitlet2d_dn > threshold)
        if self.debugplot >= 10:
            print("Number of objects initially found:", no_objects)
        if self.debugplot % 10 != 0:
            # display all objects identified in the image
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (locate_unknown_arc_lines #4)"
            z1z2 = (labels2d_objects.min(), labels2d_objects.max())
            ximshow(labels2d_objects, title=title,
                    cbar_label="Object number",
                    z1z2=z1z2, cmap="nipy_spectral",
                    image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                self.bb_ns1_orig, self.bb_ns2_orig),
                    debugplot=self.debugplot)

        # select tentative arc lines by imposing the criteria based
        # on the dimensions of the detected objects
        slices_possible_arc_lines = ndimage.find_objects(labels2d_objects)
        slices_ok = np.repeat([False], no_objects)  # flag
        for i in range(no_objects):
            if self.debugplot >= 10:
                print('slice:', i, slices_possible_arc_lines[i])
            slice_x = slices_possible_arc_lines[i][1]
            slice_y = slices_possible_arc_lines[i][0]
            delta_x = slice_x.stop - slice_x.start + 1
            delta_y = slice_y.stop - slice_y.start + 1
            if delta_x <= delta_x_max and delta_y >= delta_y_min:
                slices_ok[i] = True

        # generate list with ID of tentative arc lines
        list_slices_ok = []
        for i in range(no_objects):
            if slices_ok[i]:
                list_slices_ok.append(i+1)

        number_tentative_arc_lines = len(list_slices_ok)

        if self.debugplot >= 10:
            print("\nNumber of tentative arc lines finally identified is:",
                  number_tentative_arc_lines)
            print("Slice ID of lines passing the selection:\n",
                  list_slices_ok)

        if number_tentative_arc_lines == 0:
            raise ValueError("Number of tentative arc lines identified is 0")

        # generate mask with all the tentative arc-line points passing
        # the selection
        mask_tentative_arc_lines = np.zeros_like(slitlet2d_dn)
        for k in list_slices_ok:
            mask_tentative_arc_lines[labels2d_objects == k] = 1

        # select all data points passing the selection
        xy_tmp = np.where(mask_tentative_arc_lines == 1)
        x_tmp = xy_tmp[1] + self.bb_nc1_orig
        y_tmp = xy_tmp[0] + self.bb_ns1_orig
        w_tmp = slitlet2d_dn[xy_tmp]

        # weighted fit for tentative middle spectrum trail
        spectrail = SpectrumTrail()   # declare SpectrumTrail instance
        spectrail.fit(x=x_tmp, y=y_tmp, deg=deg_middle_spectrail, w=w_tmp)
        if self.debugplot >= 10:
            print("Tentative middle spectrum trail:\n" + str(spectrail))

        # display tentative arc lines and middle spectrum trail
        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            # display all objects identified in the image
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (locate_unknown_arc_lines #5)"
            z1z2 = (labels2d_objects.min(),
                    labels2d_objects.max())
            ax = ximshow(labels2d_objects, show=False, title=title,
                         cbar_label="Object number",
                         z1z2=z1z2, cmap="nipy_spectral",
                         image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                     self.bb_ns1_orig, self.bb_ns2_orig),
                         debugplot=self.debugplot)
            # tentative arc lines
            for i in range(no_objects):
                if slices_ok[i]:
                    slice_x = slices_possible_arc_lines[i][1]
                    slice_y = slices_possible_arc_lines[i][0]
                    xini_slice = slice_x.start + self.bb_nc1_orig
                    yini_slice = slice_y.start + self.bb_ns1_orig
                    xwidth_slice = slice_x.stop - slice_x.start + 1
                    ywidth_slice = slice_y.stop - slice_y.start + 1
                    rect = plt.Rectangle((xini_slice, yini_slice),
                                         xwidth_slice, ywidth_slice,
                                         edgecolor='w', facecolor='none')
                    ax.add_patch(rect)
            # tentative middle spectrum trail
            xpol, ypol = spectrail.linspace_pix(
                start=self.bb_nc1_orig, stop=self.bb_nc2_orig)
            ax.plot(xpol, ypol, 'r--')  # weighted fit
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

        # accept arc line only if the extreme points are within a given
        # distance (in the Y direction) from the tentative middle
        # spectrum trail
        for i in range(no_objects):
            if slices_ok[i]:
                slice_x = slices_possible_arc_lines[i][1]
                slice_y = slices_possible_arc_lines[i][0]
                # central X coordinate of the slice (pixel units)
                xmean_slice = (slice_x.start + slice_x.stop)/2. + \
                              self.bb_nc1_orig
                # extreme points of the slice (pixel units)
                ymin_slice = slice_y.start + self.bb_ns1_orig
                ymax_slice = slice_y.stop + self.bb_ns1_orig
                # central Y coordinate of the slice (pixel units)
                # computed from the tentative middle spectrum trail
                ymean_slice = spectrail.poly_funct(xmean_slice)
                # distance to the extreme points
                dist_lower = ymean_slice - ymin_slice
                dist_upper = ymax_slice - ymean_slice
                if (dist_middle_min <= dist_lower <= dist_middle_max) and \
                        (dist_middle_min <= dist_upper <= dist_middle_max):
                    pass  # the tentative arc line seems ok
                else:
                    slices_ok[i] = False  # remove tentative arc line

        # update list with ID of final arc lines
        list_slices_ok = []
        for i in range(no_objects):
            if slices_ok[i]:
                list_slices_ok.append(i+1)

        number_final_arc_lines = len(list_slices_ok)

        if self.debugplot >= 10:
            print(
                "\nNumber of final arc lines finally identified is:",
                number_final_arc_lines)
            print("Slice ID of lines passing the selection:\n",
                  list_slices_ok)

        if number_final_arc_lines == 0:
            raise ValueError(
                "Number of final arc lines identified is 0")

        # generate mask with all the final arc-line points passing
        # the selection
        mask_final_arc_lines = np.zeros_like(slitlet2d_dn)
        for k in list_slices_ok:
            mask_final_arc_lines[labels2d_objects == k] = 1

        # select all data points passing the selection
        xy_tmp = np.where(mask_final_arc_lines == 1)
        x_tmp = xy_tmp[1] + self.bb_nc1_orig
        y_tmp = xy_tmp[0] + self.bb_ns1_orig
        w_tmp = slitlet2d_dn[xy_tmp]

        # weighted fit for the final middle spectrum trail
        spectrail = SpectrumTrail()  # declare SpectrumTrail instance
        spectrail.fit(x=x_tmp, y=y_tmp, deg=deg_middle_spectrail, w=w_tmp)

        # initialize list with spectrum trails, being the first element
        # of that list the final middle spectrum trail
        self.list_spectrum_trails = [spectrail]

        # evaluate the final middle spectrum trail at a predefined
        # abscissa (self.x0_reference); note that this value is stored
        # in a numpy array (self.y0_reference) initially with only one
        # element since additional values, corresponding to auxiliary
        # spectrum trails, will be appended to it
        spectrail.y_rectified = spectrail.poly_funct(self.x0_reference)
        self.y0_reference = np.array([spectrail.y_rectified])

        if self.debugplot >= 10:
            print("Final middle spectrum trail:\n" + str(spectrail))

        # display final arc lines and middle spectrum trail
        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            # display all objects identified in the image
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (locate_unknown_arc_lines #6)"
            z1z2 = (labels2d_objects.min(),
                    labels2d_objects.max())
            ax = ximshow(labels2d_objects, show=False, title=title,
                         cbar_label="Object number",
                         z1z2=z1z2, cmap="nipy_spectral",
                         image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                     self.bb_ns1_orig, self.bb_ns2_orig))
            # final arc lines
            for i in range(no_objects):
                if slices_ok[i]:
                    slice_x = slices_possible_arc_lines[i][1]
                    slice_y = slices_possible_arc_lines[i][0]
                    xini_slice = slice_x.start + self.bb_nc1_orig
                    yini_slice = slice_y.start + self.bb_ns1_orig
                    xwidth_slice = slice_x.stop - slice_x.start + 1
                    ywidth_slice = slice_y.stop - slice_y.start + 1
                    rect = plt.Rectangle((xini_slice, yini_slice),
                                         xwidth_slice, ywidth_slice,
                                         edgecolor='w', facecolor='none')
                    ax.add_patch(rect)
            # final middle spectrum trail
            xpol, ypol = spectrail.linspace_pix(
                start=self.bb_nc1_orig, stop=self.bb_nc2_orig)
            ax.plot(xpol, ypol, 'r--')
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

        # adjust individual arc lines passing the selection
        self.list_arc_lines = []  # list of ArcLines
        for k in range(number_final_arc_lines):  # fit each arc line
            # select points to be fitted for a particular arc line
            xy_tmp = np.where(labels2d_objects == list_slices_ok[k])
            x_tmp = xy_tmp[1] + self.bb_nc1_orig
            y_tmp = xy_tmp[0] + self.bb_ns1_orig
            w_tmp = slitlet2d_dn[xy_tmp]
            # declare new ArcLine instance
            arc_line = ArcLine()
            # define new ArcLine using a weighted fit
            # (note that it must be X vs Y)
            arc_line.fit(x=x_tmp, y=y_tmp, deg=1, w=w_tmp, y_vs_x=False)
            # update list with identified ArcLines
            self.list_arc_lines.append(arc_line)

        number_arc_lines = len(self.list_arc_lines)
        if number_arc_lines < deg_middle_spectrail - 1:
            raise ValueError("Insufficient number of arc lines found!")

        if self.debugplot >= 10:
            # print list of arc lines
            print('\nlist_arc_lines:')
            for k in range(number_arc_lines):
                print(k, '->', self.list_arc_lines[k], '\n')

        # display results
        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            # compute image with only the arc lines passing the selection
            labels2d_arc_lines = labels2d_objects * mask_final_arc_lines
            # display background image with filtered arc lines
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (locate_unknown_arc_lines #7)"
            z1z2 = (labels2d_arc_lines.min(),
                    labels2d_arc_lines.max())
            ax = ximshow(labels2d_arc_lines, show=False,
                         cbar_label="Object number",
                         title=title, z1z2=z1z2, cmap="nipy_spectral",
                         image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                     self.bb_ns1_orig, self.bb_ns2_orig),
                         debugplot=self.debugplot)
            # plot weighted fit for each arc line (note that the fit is
            # X vs Y)
            for k in range(number_arc_lines):
                xpol, ypol = self.list_arc_lines[k].linspace_pix()
                ax.plot(xpol, ypol, 'g--')
            # display lower and upper points of each arc line
            x_tmp = [arc_line.xlower_line for arc_line in self.list_arc_lines]
            y_tmp = [arc_line.ylower_line for arc_line in self.list_arc_lines]
            ax.plot(x_tmp, y_tmp, 'ro')
            x_tmp = [arc_line.xupper_line for arc_line in self.list_arc_lines]
            y_tmp = [arc_line.yupper_line for arc_line in self.list_arc_lines]
            ax.plot(x_tmp, y_tmp, 'go')
            # display global fit
            xpol, ypol = self.list_spectrum_trails[0].linspace_pix(
                start=self.bb_nc1_orig, stop=self.bb_nc2_orig)
            ax.plot(xpol, ypol, 'r--')
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

    def locate_known_arc_lines(self, slitlet2d,
                               below_middle_spectrail,
                               above_middle_spectrail,
                               nwidth, nsearch):
        """Determine the location of known arc lines in slitlet.

        Parameters
        ----------
        slitlet2d : 2d numpy array, float
            Image containing the slitlet bounding box.
        below_middle_spectrail : float
            Y distance (pixels) below the middle spectrum trail from
            which the location of the arc lines is fitted.
        above_middle_spectrail : float
            Y distance (pixels) above the middle spectrum trail to
            which the location of the arc lines is fitted.
        nwidth : integer
            Width of the running window employed to estimate the
            median location of the arc line in order to extrapolate the
            expected position of that line when moving up and down from
            the middle spectrum trail.
        nsearch: integer
            Number of pixels at each side of the expected arc line
            peak employed to fit the line and obtain a more precise
            estimation of the line center, i.e., a total of
            (2 * nsearch + 1) pixels are employed in the fit.

        """

        # protections
        self._check_slitlet2d(slitlet2d)
        if nwidth % 2 == 0:
            raise ValueError("nwidth=" + str(nwidth) +
                             " must be an odd integer")
        if self.solution_wv is None:
            raise ValueError("Missing wavelength calibration")

        # slitlet dimensions
        naxis2_, naxis1_ = slitlet2d.shape

        if naxis1_ != self.naxis1_slitlet:
            raise ValueError("Unexpected NAXIS1 value for slitlet")
        if naxis2_ != self.naxis2_slitlet:
            raise ValueError("Unexpected NAXIS2 value for slitlet")

        # middle spectrum trail
        middle_spectrail = self.list_spectrum_trails[0]

        # prepare member of class to store results
        self.list_arc_lines = []

        # fit each individual known line
        for iline in range(self.solution_wv.nlines_arc):
            # starting point
            x0 = self.solution_wv.xpos[iline]
            # note that y0 must be computed in the original coordinate
            # system (before computing x0 -= self.bb_nc1_orig)
            y0 = middle_spectrail.poly_funct(x0)
            x0 -= self.bb_nc1_orig
            y0 -= self.bb_ns1_orig
            # compute minimum and maximum image array indices in
            # the y direction corresponding to the arc line region
            # which will be employed to fit the line
            imin = int(y0 - below_middle_spectrail + 0.5)
            if imin < 0:
                imin = 0
            imax = int(y0 + above_middle_spectrail + 0.5)
            if imax > naxis2_ - 1:
                imax = naxis2_ - 1
            # compute minimum and maximum image array indices of
            # the running y-window where the median location of
            # the line is computed to extrapolate the expected
            # position when moving up and down
            i1 = int(y0 - (nwidth - 1) / 2 + 0.5)
            if i1 < imin:
                i1 = imin
            i2 = int(y0 + (nwidth - 1) / 2 + 0.5)
            if i2 > imax:
                i2 = imax
            # fit line peak in each initial spectrum around the middle
            # spectrum trail
            x_fit_center = np.zeros(i2 - i1 + 1)
            y_fit_center = np.zeros(i2 - i1 + 1)
            for i in range(i1, i2 + 1):
                xdum, sdum = refine_peaks_spectrum(
                    sx=slitlet2d[i, :], ixpeaks=[int(x0 + 0.5)],
                    nwinwidth=2*nsearch+1, method="gaussian")
                x_fit_center[i - i1] = xdum[0]
                y_fit_center[i - i1] = float(i)
            # store result in arrays that should grow as the moving
            # window progresses
            x_fit_full_line = np.copy(x_fit_center)
            y_fit_full_line = np.copy(y_fit_center)
            # define arrays to store fitted line positions around
            # the middle spectrum trail, in the moving window
            x_fit_running = np.copy(x_fit_center)
            y_fit_running = np.copy(y_fit_center)
            # move upwards to fit the line peak in each spectrum above
            # the region around middle spectrum trail
            for i in range(i2 + 1, imax + 1):
                predicted_peak = int(np.median(x_fit_running) + 0.5)
                xdum, sdum = refine_peaks_spectrum(
                    sx=slitlet2d[i, :], ixpeaks=[predicted_peak],
                    nwinwidth=2*nsearch+1, method="gaussian")
                x_fit_running = np.append(x_fit_running[1:], xdum)
                y_fit_running = np.append(y_fit_running[1:],
                                          np.array([i]))
                x_fit_full_line = np.append(x_fit_full_line, xdum)
                y_fit_full_line = np.append(y_fit_full_line, np.array([i]))
            # reverse arrays employed to store the fitted line
            # position in each spectrum around the middle spectrum trail,
            x_fit_running = np.copy(x_fit_center[::-1])
            y_fit_running = np.copy(y_fit_center[::-1])
            # move downwards to fit the line peak in each spectrum
            # below the region around the middle spectrum trail
            for i in range(i1 - 1, imin - 1, -1):
                predicted_peak = int(np.median(x_fit_running) + 0.5)
                xdum, sdum = refine_peaks_spectrum(
                    sx=slitlet2d[i, :], ixpeaks=[predicted_peak],
                    nwinwidth=2*nsearch+1, method="gaussian")
                x_fit_running = np.append(x_fit_running[1:], xdum)
                y_fit_running = np.append(y_fit_running[1:],
                                          np.array([i]))
                x_fit_full_line = np.append(x_fit_full_line, xdum)
                y_fit_full_line = np.append(y_fit_full_line, np.array([i]))
            # declare new ArcLine instance
            arc_line = ArcLine()
            # define new ArcLine using a fit to the detected line peaks
            # (note that it must be X vs Y), remembering that the
            # coordinates must correspond to pixels (and not indices of
            # the image array)
            x_fit_full_line += self.bb_nc1_orig
            y_fit_full_line += self.bb_ns1_orig
            arc_line.fit(x=x_fit_full_line, y=y_fit_full_line, deg=1,
                         y_vs_x=False,
                         times_sigma_reject=10)  #, debugplot=12)
            # update list with identified ArcLines
            self.list_arc_lines.append(arc_line)

        # plot results
        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            # display image with zscale cuts
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (locate_known_arc_lines)"
            ax = ximshow(slitlet2d, title=title, show=False,
                         image_bbox=(0, 0, 0, 0))
            # middle spectrum trail
            xpol, ypol = middle_spectrail.linspace_pix(start=self.bb_nc1_orig,
                                                       stop=self.bb_nc2_orig)
            xpol -= self.bb_nc1_orig
            ypol -= self.bb_ns1_orig
            ax.plot(xpol, ypol, 'w--')
            # lower and upper spectrum trails
            lower_spectrail = middle_spectrail.offset(-below_middle_spectrail)
            xpol, ypol = lower_spectrail.linspace_pix(start=self.bb_nc1_orig,
                                                       stop=self.bb_nc2_orig)
            xpol -= self.bb_nc1_orig
            ypol -= self.bb_ns1_orig
            ax.plot(xpol, ypol, 'c--')
            upper_spectrail = middle_spectrail.offset(above_middle_spectrail)
            xpol, ypol = upper_spectrail.linspace_pix(start=self.bb_nc1_orig,
                                                       stop=self.bb_nc2_orig)
            xpol -= self.bb_nc1_orig
            ypol -= self.bb_ns1_orig
            ax.plot(xpol, ypol, 'c--')
            # location of known arc lines over middle spectrum trail
            ax.plot(self.solution_wv.xpos - self.bb_nc1_orig,
                    middle_spectrail.poly_funct(self.solution_wv.xpos)
                    - self.bb_ns1_orig,
                    "co")
            for iline in range(self.solution_wv.nlines_arc):
                arc_line = self.list_arc_lines[iline]
                xpol, ypol = arc_line.linspace_pix()
                xpol -= self.bb_nc1_orig
                ypol -= self.bb_ns1_orig
                ax.plot(xpol, ypol, 'm-')
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

    def additional_spectrail_from_middle_spectrail(self, v_offsets):
        """Estimate additional spectrum trails by shifting middle one.

        Updates the list of spectrum trails by adding new trails
        computed from the middle trail shifted according to the
        specified offsets. Note that the class member
        self.list_spectrum_trails will append the additional trails in
        the order specified by 'v_offsets'. By default, the first trail
        in that list will be considered as the middle one.

        Parameters
        ----------
        v_offsets : list of floats
            Vertical offsets to be applied to the middle spectrum trail.

        """

        # protections
        if type(v_offsets) != list:
            raise ValueError("v_offsets=" + str(v_offsets) +
                             " must be a list")
        if self.list_spectrum_trails is None:
            raise ValueError("There is no middle spectrum trail defined")

        for value in v_offsets:
            new_spectrail = self.list_spectrum_trails[0].offset(value)
            self.list_spectrum_trails.append(new_spectrail)
            # update list of spectrum trail y0_reference values
            y_rectified = new_spectrail.poly_funct(self.x0_reference)
            self.y0_reference = np.append(self.y0_reference, y_rectified)
            # update y0_rectified in the new spectrum trail itself
            new_spectrail.y_rectified = y_rectified

        if self.debugplot >= 10:
            print("Coefficients for all the spectrum trails:")
            for spectrail in self.list_spectrum_trails:
                print(spectrail.poly_funct.coef)

    def xy_spectrail_arc_intersections(self, slitlet2d=None,
                                      apply_wv_calibration=False):
        """Compute intersection points of spectrum trails with arc lines.

        The member list_arc_lines is updated with new keyword:keyval
        values for each arc line.

        Parameters
        ----------
        slitlet2d : 2d numpy array, float (optional)
            Slitlet image to be displayed with the computed boundaries
            and intersecting points overplotted. Note that the image is
            displayed only if self.debugplot % 10 != 0.
        apply_wv_calibration : bool
            If True, apply wavelength calibration in order to compute
            abscissae for the rectified arc lines that guarantee a
            linear dispersion in the wavelength direction.

        """

        # protections
        number_spectrum_trails = len(self.list_spectrum_trails)
        if number_spectrum_trails == 0:
            raise ValueError("Number of spectrum trails is 0")
        if self.list_arc_lines is None:
            raise ValueError("Arc lines not sought")
        number_arc_lines = len(self.list_arc_lines)
        if number_arc_lines == 0:
            raise ValueError("Number of available arc lines is 0")
        if slitlet2d is not None:
            self._check_slitlet2d(slitlet2d)
        if apply_wv_calibration:
            if self.solution_wv is None:
                raise ValueError("Missing wavelength calibration")

        # intersection of the arc lines with the spectrum trails
        # (note: the coordinates are computed using pixel values,
        #  ranging from 1 to NAXIS1_EMIR, as given in the original
        #  image reference system ---not in the slitlet image reference
        #  system---)
        self.x_inter_orig = np.array([])  # original image coordinates
        self.y_inter_orig = np.array([])  # original image coordinates
        self.x_inter_rect = np.array([])  # rectified image coordinates
        self.y_inter_rect = np.array([])  # rectified image coordinates

        # when applying wavelength calibration, it is necessary to
        # update some important data members
        if apply_wv_calibration:
            # expected wavelength at pixel (integer transformed to
            # float) where self.x0_reference is found, using for that
            # purpose the linear prediction of the wavelength
            # calibration
            pixel_x0 = float(int(self.x0_reference + 0.5))
            wv_at_x0 = self.solution_wv.crval1_linear + \
                       (pixel_x0 - self.solution_wv.crpix1_linear) * \
                       self.solution_wv.cdelt1_linear
            # pixel coordinate within the enlarged image where the
            # previous wavelength must appear
            pixel_wv_enlarged = (wv_at_x0 - self.crval1_enlarged) / \
                                self.cdelt1_enlarged + self.crpix1_enlarged
            # fraction of previous pixel
            # (note: that this fraction of pixel will be employed to
            # guarantee that all the wavelength calibrated slitlets
            # will share a common wavelength calibration scale, where
            # the different slitlets can overlap after applying
            # translations involving an integer number of pixels, with
            # no need of computing interpolations using fractions of
            # pixels)
            frac_pixel = pixel_wv_enlarged - int(pixel_wv_enlarged)
            # estimate CRVAL1 for a spectrum of the same length as the
            # slitlet, but using the CDELT1 value of the enlarged image
            self.crpix1_slitlet = 1.0
            self.cdelt1_slitlet = self.cdelt1_enlarged
            self.crval1_slitlet = \
                wv_at_x0 - \
                (pixel_x0 - (self.bb_nc1_orig - 1) -
                 self.crpix1_slitlet + frac_pixel) * self.cdelt1_slitlet
            self.crmin1_slitlet = self.crval1_slitlet
            self.crmax1_slitlet = self.crval1_slitlet + \
                                  (self.bb_nc2_orig - self.bb_nc1_orig) * \
                                  self.cdelt1_slitlet

        # loop in arc lines
        for i in range(len(self.list_arc_lines)):
            arcline = self.list_arc_lines[i]
            # approximate location of the solution
            expected_x = (arcline.xlower_line + arcline.xupper_line) / 2.0
            # loop in spectrum trails
            first_spectrail = True
            for spectrail in self.list_spectrum_trails:
                # composition of polynomials to find intersection as
                # one of the roots of a new polynomial
                rootfunct = arcline.poly_funct(spectrail.poly_funct)
                rootfunct.coef[1] -= 1
                # compute roots to find solution
                tmp_xroots = rootfunct.roots()
                # take the nearest root to the expected location
                xroot = tmp_xroots[np.abs(tmp_xroots - expected_x).argmin()]
                if np.isreal(xroot):
                    xroot = xroot.real
                else:
                    raise ValueError("xroot=" + str(xroot) +
                                     " is a complex number")
                yroot = spectrail.poly_funct(xroot)
                self.x_inter_orig = np.append(self.x_inter_orig, xroot)
                self.y_inter_orig = np.append(self.y_inter_orig, yroot)
                if first_spectrail:  # the middle spectrum trail
                    # the abscissae for the intersection of the
                    # rectified arc line with all the spectrum trails
                    # are identical
                    if apply_wv_calibration:
                        # wavelength of the corresponding arc line
                        wv_arcline = self.solution_wv.reference[i]
                        # expected pixel (relative to the actual
                        # slitlet size)
                        xpos_linear = (wv_arcline - self.crval1_slitlet) / \
                                      self.cdelt1_slitlet + self.crpix1_slitlet
                        # since crval1_slitlet has been computed
                        # relative to the actual slitlet size, it is
                        # neccessary to transform the previous value
                        # to the absolute pixel location in the
                        # original image
                        xpos_linear += (self.bb_nc1_orig - 1)
                        # store information
                        arcline.x_rectified = xpos_linear
                        self.x_inter_rect = np.append(
                            self.x_inter_rect,
                            [xpos_linear] * number_spectrum_trails)
                    else:
                        arcline.x_rectified = xroot
                        self.x_inter_rect = np.append(
                            self.x_inter_rect,
                            [xroot]*number_spectrum_trails)
                    first_spectrail = False
            # ordinates of rectified intersection points
            self.y_inter_rect = np.append(self.y_inter_rect, self.y0_reference)

        # display slitlet with intersection points
        if self.debugplot % 10 != 0 and slitlet2d is not None:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            # display image with zscale cuts
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (xy_spectrail_arc_intersections #1)"
            ax = ximshow(slitlet2d, show=False, title=title,
                         image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                     self.bb_ns1_orig, self.bb_ns2_orig),
                         debugplot=self.debugplot)
            # spectrum trails
            for spectrail in self.list_spectrum_trails:
                xdum, ydum = spectrail.linspace_pix(start=self.bb_nc1_orig,
                                                    stop=self.bb_nc2_orig)
                ax.plot(xdum, ydum, 'g')
            # arc lines
            for arcline in self.list_arc_lines:
                xdum, ydum = arcline.linspace_pix(start=self.bb_ns1_orig,
                                                  stop=self.bb_ns2_orig)
                ax.plot(xdum, ydum, 'g')
            # intersection points
            ax.plot(self.x_inter_orig, self.y_inter_orig, 'co')
            ax.plot(self.x_inter_rect, self.y_inter_rect, 'bo')
            # show plot
            pause_debugplot(self.debugplot,pltshow=True)
            # same image but using array index coordinates
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (xy_spectrail_arc_intersections #2)"
            ax = ximshow(slitlet2d, show=False, title=title,
                         image_bbox=(0, 0, 0, 0),
                         debugplot=self.debugplot)
            # intersection points
            ax.plot(self.x_inter_orig - self.bb_nc1_orig,
                    self.y_inter_orig - self.bb_ns1_orig, 'co')
            ax.plot(self.x_inter_rect - self.bb_nc1_orig,
                    self.y_inter_rect - self.bb_ns1_orig, 'bo')
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

    def estimate_tt_to_rectify(self, order=2, slitlet2d=None):
        """Estimate the transformation to rectify slitlet.

        Parameters
        ----------
        order : int
            Order to be employed in the transformation estimated to
            rectify the image. See the documentation of scikit-image
            function transform.PolynomialTransform.estimate().
        slitlet2d : 2d numpy array, float (optional)
            Slitlet image to be displayed with the computed boundaries
            and intersecting points overplotted. Note that the image is
            displayed only if self.debugplot % 10 != 0.

        """

        # protections
        if len(self.list_spectrum_trails) == 0:
            raise ValueError("Number of spectrum trails is zero")
        if len(self.list_arc_lines) == 0:
            raise ValueError("Number of detected arc lines is zero")
        if self.x_inter_orig is None:
            raise ValueError("self.x_inter_orig is None")
        if self.y_inter_orig is None:
            raise ValueError("self.y_inter_orig is None")
        if self.x_inter_rect is None:
            raise ValueError("self.x_inter_rect is None")
        if self.y_inter_rect is None:
            raise ValueError("self.y_inter_rect is None")
        if slitlet2d is not None:
            self._check_slitlet2d(slitlet2d)

        # correct coordinates from origin in order to manipulate
        # coordinates corresponding to image indices
        x_inter_orig_shifted = self.x_inter_orig - self.bb_nc1_orig
        y_inter_orig_shifted = self.y_inter_orig - self.bb_ns1_orig
        x_inter_rect_shifted = self.x_inter_rect - self.bb_nc1_orig
        y_inter_rect_shifted = self.y_inter_rect - self.bb_ns1_orig

        # normalize ranges dividing by the maximum, so the
        # transformation fit will be computed with data points with
        # coordinates in the range [0,1]
        x_scale = 1.0 / np.concatenate((x_inter_orig_shifted,
                                        x_inter_rect_shifted)).max()
        y_scale = 1.0 / np.concatenate((y_inter_orig_shifted,
                                        y_inter_rect_shifted)).max()
        if self.debugplot >= 10:
            print("x_scale:", x_scale)
            print("y_scale:", y_scale)
        x_inter_orig_scaled = x_inter_orig_shifted * x_scale
        y_inter_orig_scaled = y_inter_orig_shifted * y_scale
        x_inter_rect_scaled = x_inter_rect_shifted * x_scale
        y_inter_rect_scaled = y_inter_rect_shifted * y_scale

        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.autoscale(False)
            ax.plot(x_inter_orig_scaled, y_inter_orig_scaled, 'co',
                    label="original")
            dum = zip(x_inter_orig_scaled, y_inter_orig_scaled)
            for idum in range(len(dum)):
                ax.text(dum[idum][0], dum[idum][1], str(idum+1), fontsize=10,
                        horizontalalignment='center',
                        verticalalignment='bottom', color='green')
            ax.plot(x_inter_rect_scaled, y_inter_rect_scaled, 'bo',
                    label="rectified")
            dum = zip(x_inter_rect_scaled, y_inter_rect_scaled)
            for idum in range(len(dum)):
                ax.text(dum[idum][0], dum[idum][1], str(idum+1), fontsize=10,
                        horizontalalignment='center',
                        verticalalignment='bottom', color='blue')
            xmin = np.concatenate((x_inter_orig_scaled,
                                   x_inter_rect_scaled)).min()
            xmax = np.concatenate((x_inter_orig_scaled,
                                   x_inter_rect_scaled)).max()
            ymin = np.concatenate((y_inter_orig_scaled,
                                   y_inter_rect_scaled)).min()
            ymax = np.concatenate((y_inter_orig_scaled,
                                   y_inter_rect_scaled)).max()
            dx = xmax - xmin
            xmin -= dx/20
            xmax += dx/20
            dy = ymax - ymin
            ymin -= dy/20
            ymax += dy/20
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_xlabel("pixel (normalized coordinate)")
            ax.set_ylabel("pixel (normalized coordinate)")
            ax.set_title("(estimate_tt_to_rectify #1)\n\n")
            # shrink current axis and put a legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.92])
            ax.legend(loc=3, bbox_to_anchor=(0., 1.02, 1., 0.07),
                      mode="expand", borderaxespad=0., ncol=4,
                      numpoints=1)
            pause_debugplot(self.debugplot, pltshow=True)

        # build data sets to fit transformation
        orig_points = np.array(zip(x_inter_orig_scaled, y_inter_orig_scaled))
        rect_points = np.array(zip(x_inter_rect_scaled, y_inter_rect_scaled))

        # TODO: is this really necessary?
        # use only a fraction of the data points (this step was
        # necessary when computing the transformation without
        # normalizing the data ranges to the [0,1] interval; using too
        # many points lead to numerical errors
        # (note: use n_step_remove = 2, 3,... to select a subsample)
        n_step_remove = 1
        if n_step_remove > 1:
            orig_points = orig_points[1:len(orig_points):n_step_remove]
            rect_points = rect_points[1:len(rect_points):n_step_remove]

        # estimate the transformation
        """
        self.ttd = transform.PolynomialTransform()
        ok_estimate = self.ttd.estimate(rect_points, orig_points, order=order)
        if not ok_estimate:
            print(self.__repr__())
            raise ValueError("Polynomial transformation ttd estimate failed!")
        # estimate the inverse transformation
        self.tti = transform.PolynomialTransform()
        ok_estimate = self.tti.estimate(orig_points, rect_points, order=order)
        if not ok_estimate:
            print(self.__repr__())
            raise ValueError("Polynomial transformation tti estimate failed!")
        """
        # In order to avoid the need of solving a large system of
        # equations (as shown in the previous commented-out code), next
        # we prefer to solve 2 systems of equations with half number
        # of unknowns each.
        if order == 1:
            A = np.vstack([np.ones(len(x_inter_rect_scaled)),
                           x_inter_rect_scaled,
                           y_inter_rect_scaled]).T
        elif order == 2:
            A = np.vstack([np.ones(len(x_inter_rect_scaled)),
                           x_inter_rect_scaled,
                           y_inter_rect_scaled,
                           x_inter_rect_scaled ** 2,
                           x_inter_rect_scaled * y_inter_orig_scaled,
                           y_inter_rect_scaled ** 2]).T
        elif order == 3:
            A = np.vstack([np.ones(len(x_inter_rect_scaled)),
                           x_inter_rect_scaled,
                           y_inter_rect_scaled,
                           x_inter_rect_scaled**2,
                           x_inter_rect_scaled*y_inter_orig_scaled,
                           y_inter_rect_scaled**2,
                           x_inter_rect_scaled**3,
                           x_inter_rect_scaled**2 * y_inter_rect_scaled,
                           x_inter_rect_scaled * y_inter_rect_scaled**2,
                           y_inter_rect_scaled**3]).T
        elif order == 4:
            A = np.vstack([np.ones(len(x_inter_rect_scaled)),
                           x_inter_rect_scaled,
                           y_inter_rect_scaled,
                           x_inter_rect_scaled**2,
                           x_inter_rect_scaled*y_inter_orig_scaled,
                           y_inter_rect_scaled**2,
                           x_inter_rect_scaled**3,
                           x_inter_rect_scaled**2 * y_inter_rect_scaled,
                           x_inter_rect_scaled * y_inter_rect_scaled**2,
                           y_inter_rect_scaled**3,
                           x_inter_rect_scaled**4,
                           x_inter_rect_scaled**3 * y_inter_rect_scaled**1,
                           x_inter_rect_scaled**2 * y_inter_rect_scaled**2,
                           x_inter_rect_scaled**1 * y_inter_rect_scaled**3,
                           y_inter_rect_scaled**4]).T
        else:
            raise ValueError("Invalid order=" + str(order))
        self.ttd = transform.PolynomialTransform(
            np.vstack(
                [np.linalg.lstsq(A, x_inter_orig_scaled)[0],
                 np.linalg.lstsq(A, y_inter_orig_scaled)[0]]
            )
        )
        #self.ttd.params[0] = np.linalg.lstsq(A, x_inter_orig_scaled)[0]
        #self.ttd.params[1] = np.linalg.lstsq(A, y_inter_orig_scaled)[0]
        #print(np.linalg.lstsq(A, x_inter_orig_scaled)[0])
        #print(np.linalg.lstsq(A, y_inter_orig_scaled)[0])

        # reverse normalization to recover coefficients of the
        # transformation in the correct system
        factor = np.zeros_like(self.ttd.params[0])
        k = 0
        for i in range(order + 1):
            for j in range(i + 1):
                factor[k] = (x_scale**(i-j)) * (y_scale**j)
                k += 1
        self.ttd.params[0] *= factor/x_scale
        self.ttd.params[1] *= factor/y_scale
        if self.debugplot >= 10:
            print("ttd.params X:\n", self.ttd.params[0])
            print("ttd.params Y:\n", self.ttd.params[1])

        # display slitlet with intersection points and grid indicating
        # the fitted transformation
        if self.debugplot % 10 != 0 and slitlet2d is not None:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            # display image with zscale cuts
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (estimate_tt_to_rectify #2)"
            ax = ximshow(slitlet2d, show=False, title=title,
                         image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                     self.bb_ns1_orig, self.bb_ns2_orig),
                         debugplot=self.debugplot)
            # intersection points
            ax.plot(self.x_inter_orig, self.y_inter_orig, 'co')
            # grid with fitted transformation: horizontal boundaries
            xx = np.arange(0, self.bb_nc2_orig - self.bb_nc1_orig + 1,
                           dtype=np.float)
            for spectrail in self.list_spectrum_trails:
                yy0 = spectrail.y_rectified
                yy = np.tile([yy0 - self.bb_ns1_orig], xx.size)
                ax.plot(xx + self.bb_nc1_orig, yy + self.bb_ns1_orig, "b")
                xxx = np.zeros_like(xx)
                yyy = np.zeros_like(yy)
                k = 0
                for i in range(order + 1):
                    for j in range(i + 1):
                        xxx += self.ttd.params[0][k] * \
                               (xx ** (i - j)) * (yy ** j)
                        yyy += self.ttd.params[1][k] * \
                               (xx ** (i - j)) * (yy ** j)
                        k += 1
                ax.plot(xxx + self.bb_nc1_orig, yyy + self.bb_ns1_orig, "g")
            # grid with fitted transformation: arc lines
            ylower_line = self.y0_reference.min()
            yupper_line = self.y0_reference.max()
            n_points = int(yupper_line - ylower_line + 0.5) + 1
            yy = np.linspace(ylower_line - self.bb_ns1_orig,
                             yupper_line - self.bb_ns1_orig,
                             num=n_points,
                             dtype=np.float)
            for arc_line in self.list_arc_lines:
                xline = arc_line.x_rectified - self.bb_nc1_orig
                xx = np.array([xline]*n_points)
                ax.plot(xx + self.bb_nc1_orig, yy + self.bb_ns1_orig, "b")
                xxx = np.zeros_like(xx)
                yyy = np.zeros_like(yy)
                k = 0
                for i in range(order + 1):
                    for j in range(i + 1):
                        xxx += self.ttd.params[0][k] * \
                               (xx ** (i - j)) * (yy ** j)
                        yyy += self.ttd.params[1][k] * \
                               (xx ** (i - j)) * (yy ** j)
                        k += 1
                ax.plot(xxx + self.bb_nc1_orig,
                        yyy + self.bb_ns1_orig, "c")
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

    def rectify(self, slitlet2d, order=1):
        """Rectify slitlet image.

        Parameters
        ----------
        slitlet2d : 2d numpy array
            Slitlet image to be rectified.
        order : int
            The order of interpolation. The order has to be in the
            range 0-5:
             - 0: Nearest-neighbor
             - 1: Bi-linear (default)
             - 2: Bi-quadratic
             - 3: Bi-cubic
             - 4: Bi-quartic
             - 5: Bi-quintic

        Returns
        -------
        slitlet2d_rect : 2d numpy array
            Rectified slitlet image.

        """

        # protections
        if self.ttd is None:
            raise ValueError("self.ttd is None")
        self._check_slitlet2d(slitlet2d)

        # rescale image flux to [-1,1]
        slitlet2d_rs, coef_rs = rescale_array_to_z1z2(slitlet2d, z1z2=(-1, 1))

        # rectify image
        slitlet2d_rect = transform.warp(slitlet2d_rs, self.ttd,
                                        order=order, cval=-coef_rs[1])

        # rescale image flux to original scale
        slitlet2d_rect = rescale_array_from_z1z2(slitlet2d_rect, coef_rs)

        # display rectified slitlet
        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) +"]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (rectify)"
            ax = ximshow(slitlet2d_rect, show=False, title=title,
                         image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                     self.bb_ns1_orig, self.bb_ns2_orig),
                         debugplot=self.debugplot)
            ax.plot(self.x_inter_rect, self.y_inter_rect, 'bo')
            for y0 in self.y0_reference:  # loop in horizontal boundaries
                ax.plot([self.bb_nc1_orig, self.bb_nc2_orig], [y0, y0], 'b')
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

        # return result
        return slitlet2d_rect

    def median_spectrum_around_middle_spectrail(self, slitlet2d,
                                                below_middle_spectrail,
                                                above_middle_spectrail,
                                                sigma_gaussian_filtering,
                                                nwinwidth_initial,
                                                nwinwidth_refined,
                                                times_sigma_threshold,
                                                minimum_threshold_factor=None,
                                                npix_avoid_border=0):
        """Compute median spectrum around the distorted middle spectrum trail.

        The median spectrum is computed using pixels above and below
        the middle spectrum trail, ignoring that the arc lines can be
        slighted tilted. Thus, the number of pixels above and below
        should not be very large.

        In addition, this function determines the location of the arc
        line peaks.

        Parameters
        ----------
        slitlet2d : 2d numpy array, float
            Slitlet image.
        below_middle_spectrail : float
            Y distance (pixels) below the middle spectrum trail from
            which the median spectrum will be computed.
        above_middle_spectrail : float
            Y distance (pixels) above the middle spectrum trail to
            which the median spectrum will be computed.
        sigma_gaussian_filtering : int
            Sigma of the gaussian filter to be applied to the median
            spectrum in order to avoid problems with saturared lines.
            This filtering is skipped if this parameter is zero.
        nwinwidth_initial : int
            Width of the window where each peak must be found using
            the initial method (approximate)
        nwinwidth_refined : int
            Width of the window where each peak location will be
            refined.
        times_sigma_threshold : float
            Times (robust) sigma above the median of the image to set
            the minimum threshold when searching for line peaks.
        minimum threshold_factor : float or None
            The maximum of the median spectrum divided by this factor
            is employed as an additional threshold.
        npix_avoid_border : int
            Number of pixels at the borders of the spectrum where peaks
            are not considered. If zero, the actual number will be
            given by nwinwidth_initial.

        Returns
        -------
        median_spectrum : 1d numpy array, float
            Median spectrum.
        fxpeaks : 1d numpy array, float
            Refined location of arc lines (in array index scale).
        sxpeaks : 1d numpy array, float
            When fitting Gaussians, this array stores the fitted line
            widths (sigma). Otherwise, this array returns zeros.

        """

        # protections
        if len(self.list_spectrum_trails) == 0:
            raise ValueError("Number of spectrum trails is zero.")

        # define middle spectrum trail
        middle_spectrail = self.list_spectrum_trails[0]

        # slitled2d dimensions
        naxis2_, naxis1_ = slitlet2d.shape

        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            # display image with zscale cuts
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) +"]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (median_spectrum_around_middle_spectrail #1)"
            ax = ximshow(slitlet2d, title=title,
                         image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                     self.bb_ns1_orig, self.bb_ns2_orig),
                         show=False)
            # overplot middle spectrum trail
            xp, yp_middle = middle_spectrail.linspace_pix(
                start=self.bb_nc1_orig,
                stop=self.bb_nc2_orig)
            yp_lower = yp_middle - below_middle_spectrail
            yp_upper = yp_middle + above_middle_spectrail
            ax.plot(xp, yp_middle, 'r-')
            ax.plot(xp, yp_lower, 'r--')
            ax.plot(xp, yp_upper, 'r--')
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

        # compute median spectrum
        median_spectrum = np.zeros(naxis1_)
        for j in range(naxis1_):
            xdum = self.bb_nc1_orig + float(j)   # image pixel
            ydum = middle_spectrail.poly_funct(xdum)  # image pixel
            idum = int(ydum + 0.5) - self.bb_ns1_orig # array index
            idum_min = idum - below_middle_spectrail
            if idum_min < 0:
                idum_min = 0
            idum_max = idum + above_middle_spectrail
            if idum_max > naxis2_ - 1:
                idum_max = naxis2_ -1
            if idum_min < idum_max:
                median_spectrum[j] = \
                    np.median(slitlet2d[idum_min:(idum_max+1),j], axis=0)

        # gaussian filtering
        if sigma_gaussian_filtering > 0:
            median_spectrum = gaussian_filter(median_spectrum,
                                              sigma=sigma_gaussian_filtering)

        # initial location of the peaks (integer values)
        q25, q50, q75 = np.percentile(median_spectrum, q=[25.0, 50.0, 75.0])
        sigma_g = 0.7413 * (q75 - q25)  # robust standard deviation
        threshold = q50 + times_sigma_threshold * sigma_g
        if self.debugplot >= 10:
            print("median...........:", q50)
            print("robuts std.......:", sigma_g)
            print("threshold........:", threshold)
        if minimum_threshold_factor is not None:
            minimum_threshold = median_spectrum.max()/minimum_threshold_factor
            if minimum_threshold > threshold:
                threshold = minimum_threshold
                if self.debugplot >= 10:
                    print("minimum threshold:", minimum_threshold)
                    print("final threshold..:", threshold)

        ixpeaks = find_peaks_spectrum(median_spectrum,
                                      nwinwidth=nwinwidth_initial,
                                      threshold=threshold)

        # remove peaks too close to any of the borders of the spectrum
        if npix_avoid_border > 0:
            lok_ini = ixpeaks >= npix_avoid_border
            lok_end = ixpeaks <= len(median_spectrum) - 1 - npix_avoid_border
            ixpeaks = ixpeaks[lok_ini * lok_end]

        # refined location of the peaks (float values)
        fxpeaks, sxpeaks = refine_peaks_spectrum(median_spectrum, ixpeaks,
                                                 nwinwidth=nwinwidth_refined,
                                                 method="gaussian")

        # display median spectrum and peaks
        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (median_spectrum_around_middle_spectrail #2)"
            ax = ximplot(median_spectrum, title=title, show=False,
                         plot_bbox=(self.bb_nc1_orig, self.bb_nc2_orig))
            ymin = median_spectrum.min()
            ymax = median_spectrum.max()
            dy = ymax-ymin
            ymin -= dy/20.
            ymax += dy/20.
            ax.set_ylim([ymin, ymax])
            # display threshold
            plt.axhline(y=threshold, color="black", linestyle="dotted",
                        label="detection threshold")
            # mark peak location
            plt.plot(ixpeaks + self.bb_nc1_orig,
                     median_spectrum[ixpeaks], 'bo', label="initial location")
            plt.plot(fxpeaks + self.bb_nc1_orig,
                     median_spectrum[ixpeaks], 'go', label="refined location")
            # legend
            plt.legend(numpoints=1)
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

        # return result
        return median_spectrum, fxpeaks, sxpeaks

    def median_spectrum_from_rectified_image(self, slitlet2d_rect,
                                             below_middle_spectrail,
                                             above_middle_spectrail,
                                             nwinwidth_initial,
                                             nwinwidth_refined,
                                             times_sigma_threshold,
                                             minimum_threshold=None,
                                             npix_avoid_border=0):
        """Compute median spectrum around the rectified middle spectrum trail.

        The median spectrum is computed using spectra above and below
        the horizontal (i.e. rectified) middle spectrum trail.

        In addition, this function determines the location of the arc
        line peaks.

        Parameters
        ----------
        slitlet2d_rect : 2d numpy array, float
            Rectified slitlet image.
        below_middle_spectrail : float
            Y distance (pixels) below the middle spectrum trail from
            which the median spectrum will be computed.
        above_middle_spectrail : float
            Y distance (pixels) above the middle spectrum trail to
            which the median spectrum will be computed.
        nwinwidth_initial : int
            Width of the window where each peak must be found using
            the initial method (approximate)
        nwinwidth_refined : int
            Width of the window where each peak location will be
            refined.
        times_sigma_threshold : float
            Times (robust) sigma above the median of the image to set
            the minimum threshold when searching for line peaks.
        minimum threshold : float or None
            Minimum value of the threshold.
        npix_avoid_border : int
            Number of pixels at the borders of the spectrum where peaks
            are not considered. If zero, the actual number will be
            given by nwinwidth_initial.

        Returns
        -------
        median_spectrum : 1d numpy array, float
            Median spectrum.
        fxpeaks : 1d numpy array, float
            Refined location of arc lines (in array index scale).
        sxpeaks : 1d numpy array, float
            When fitting Gaussians, this array stores the fitted line
            widths (sigma). Otherwise, this array returns zeros.

        """

        # protections
        if len(self.list_spectrum_trails) == 0:
            raise ValueError("Number of spectrum trails is zero.")

        # rectangle where the median spectrum is going to be computed
        # in pixel coordinates
        xini_rect = self.bb_nc1_orig
        xwidth_rect = self.bb_nc2_orig - self.bb_nc1_orig + 1
        yini_rect = self.y0_reference[0] - below_middle_spectrail
        ywidth_rect = below_middle_spectrail + above_middle_spectrail
        if self.debugplot >= 10:
            print("yini_rect:", yini_rect)
            print("ycenter..:", self.y0_reference[0])
            print("yend_rect:", yini_rect + ywidth_rect)

        # idem in array coordinates
        naxis2_, naxis1_ = slitlet2d_rect.shape
        imin = int(yini_rect + 0.5) - self.bb_ns1_orig
        if imin < 0:
            raise ValueError("Lower limit < 0")
        imax = int(yini_rect + ywidth_rect + 0.5) - self.bb_ns1_orig
        if imax >= naxis2_:
            raise ValueError("Upper limit beyond image size in Y direction")
        if self.debugplot >= 10:
            print('Extracting median from image rows:', imin, imax)

        # display image with region around middle spectrum trail to be
        # used to extract the median spectrum
        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            # display image with zscale cuts
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (median_spectrum_from_rectified_image #1)"
            ax = ximshow(slitlet2d_rect, show=False, title=title,
                         image_bbox=(self.bb_nc1_orig, self.bb_nc2_orig,
                                     self.bb_ns1_orig, self.bb_ns2_orig))
            # display middle spectrum trail
            ax.plot([self.bb_nc1_orig, self.bb_nc2_orig],
                    [self.y0_reference[0], self.y0_reference[0]], 'b-')
            # display region around middle spectrum trail
            rect = plt.Rectangle((xini_rect, yini_rect),
                                 xwidth_rect, ywidth_rect,
                                 edgecolor='c', facecolor='none')
            ax.add_patch(rect)
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

        # compute median spectrum
        median_spectrum = np.median(slitlet2d_rect[imin:(imax+1), :], axis=0)

        # initial location of the peaks (integer values)
        q25, q50, q75 = np.percentile(median_spectrum, q=[25.0, 50.0, 75.0])
        sigma_g = 0.7413 * (q75 - q25)  # robust standard deviation
        threshold = q50 + times_sigma_threshold * sigma_g
        if self.debugplot >= 10:
            print("median...........:", q50)
            print("robuts std.......:", sigma_g)
            print("threshold........:", threshold)
        if minimum_threshold > threshold:
            threshold = minimum_threshold
            if self.debugplot >= 10:
                print("minimum threshold:", minimum_threshold)
                print("final threshold..:", threshold)

        ixpeaks = find_peaks_spectrum(median_spectrum,
                                      nwinwidth=nwinwidth_initial,
                                      threshold=threshold)

        # remove peaks too close to any of the borders of the spectrum
        if npix_avoid_border > 0:
            lok_ini = ixpeaks >= npix_avoid_border
            lok_end = ixpeaks <= len(median_spectrum) - 1 - npix_avoid_border
            ixpeaks = ixpeaks[lok_ini * lok_end]

        # refined location of the peaks (float values)
        fxpeaks, sxpeaks = refine_peaks_spectrum(median_spectrum, ixpeaks,
                                                 nwinwidth=nwinwidth_refined,
                                                 method="gaussian")

        # display median spectrum and peaks
        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (median_spectrum_from_rectified_image #2)"
            ax = ximplot(median_spectrum, title=title, show=False,
                         plot_bbox=(self.bb_nc1_orig, self.bb_nc2_orig))
            ymin = median_spectrum.min()
            ymax = median_spectrum.max()
            dy = ymax-ymin
            ymin -= dy/20.
            ymax += dy/20.
            ax.set_ylim([ymin, ymax])
            # display threshold
            plt.axhline(y=threshold, color="black", linestyle="dotted",
                        label="detection threshold")
            # mark peak location
            plt.plot(ixpeaks + self.bb_nc1_orig,
                     median_spectrum[ixpeaks], 'bo', label="initial location")
            plt.plot(fxpeaks + self.bb_nc1_orig,
                     median_spectrum[ixpeaks], 'go', label="refined location")
            # legend
            plt.legend(numpoints=1)
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

        # return result
        return median_spectrum, fxpeaks, sxpeaks

    def wavelength_calibration(self, fxpeaks,
                               lamp,
                               crpix1,
                               error_xpos_arc,
                               times_sigma_r,
                               frac_triplets_for_sum,
                               times_sigma_theil_sen,
                               poly_degree_wfit,
                               times_sigma_polfilt,
                               times_sigma_cook,
                               times_sigma_inclusion,
                               weighted,
                               plot_residuals=False):
        """Wavelength calibration of arc lines located at xfpeaks.

        Parameters
        ----------
        fxpeaks : 1d numpy array, floats
            Refined location of arc lines (in array index scale).
        lamp : string
            Lamp identification.
        crpix1 : float
            CRPIX1 value to be employed in the wavelength calibration.
        error_xpos_arc : float
            Error in arc line position (pixels).
        times_sigma_r : float
            Times sigma to search for valid line position ratios.
        frac_triplets_for_sum : float
            Fraction of distances to different triplets to sum when
            computing the cost function.
        times_sigma_theil_sen : float
            Number of times the (robust) standard deviation around the
            linear fit (using the Theil-Sen method) to reject points.
        poly_degree_wfit : int
            Degree for polynomial fit to wavelength calibration.
        times_sigma_polfilt : float
            Number of times the (robust) standard deviation around the
            polynomial fit to reject points.
        times_sigma_cook : float
            Number of times the standard deviation of Cook's distances
            to detect outliers. If zero, this method of outlier
            detection is ignored.
        times_sigma_inclusion : float
            Number of times the (robust) standard deviation around the
            polynomial fit to include a new line in the set of
            identified lines.
        weighted : bool
            Determines whether the polynomial fit of the wavelength
            calibration is weighted or not, using as weights the values
            of the cost function obtained in the line identification.
        plot_residuals : bool
            If True, plot residuals (independently of the current value
            of debugplot).

        """

        # TODO: include protections here

        # change scale from image index to channel number
        xchannel = fxpeaks + self.bb_nc1_orig

        # read arc line wavelengths from external file
        arc_line_filename = lamp + ".dat"  # "_" + self.grism_name + "band.dat"
        master_table = np.genfromtxt(arc_line_filename)
        wv_master = master_table[:, 0]
        if self.debugplot >= 10:
            print("Reading master table: " + arc_line_filename)
            print("wv_master:\n", wv_master)

        # clip master arc lines to expected wavelength range
        lok1 = self.crmin1_enlarged <= wv_master
        lok2 = wv_master <= self.crmax1_enlarged
        lok = lok1 * lok2
        wv_master = wv_master[lok]
        if self.debugplot >= 10:
            print("clipped wv_master:\n", wv_master)

        # wavelength calibration
        list_of_wvfeatures = arccalibration(
            wv_master=wv_master,
            xpos_arc=xchannel,
            naxis1_arc=NAXIS1_EMIR,
            crpix1=crpix1,
            wv_ini_search=self.crmin1_enlarged,
            wv_end_search=self.crmax1_enlarged,
            error_xpos_arc=error_xpos_arc,
            times_sigma_r=times_sigma_r,
            frac_triplets_for_sum=frac_triplets_for_sum,
            times_sigma_theil_sen=times_sigma_theil_sen,
            poly_degree_wfit=poly_degree_wfit,
            times_sigma_polfilt=times_sigma_polfilt,
            times_sigma_cook=times_sigma_cook,
            times_sigma_inclusion=times_sigma_inclusion,
            debugplot=self.debugplot
        )

        debugplot_dum = self.debugplot
        if plot_residuals:
            if self.debugplot == 0 or self.debugplot == 10:
                debugplot_dum += 1

        self.solution_wv = fit_list_of_wvfeatures(
                list_of_wvfeatures=list_of_wvfeatures,
                naxis1_arc=NAXIS1_EMIR,
                crpix1=crpix1,
                poly_degree_wfit=poly_degree_wfit,
                weighted=weighted,
                debugplot=debugplot_dum,
                plot_title=self.fits_file_name
            )

    def overplot_wavelength_calibration(self, median_spectrum,
                                        fxpeaks=None):
        """Overplot wavelength calibration over median spectrum.

        Parameters
        ----------
        median_spectrum : numpy array, floats
            Median spectrum employed to determine line peaks and obtain
            the wavelength calibration.
        fxpeaks : 1d numpy array, float (optional)
            Refined location of arc line peaks (in array index scale).
            If this parameter is None, these peaks are not plotted.

        """
        if self.debugplot % 10 != 0:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            title = self.fits_file_name + \
                    " [slit #" + str(self.slitlet_number) + "]" + \
                    "\ngrism=" + self.grism_name + \
                    ", filter=" + self.filter_name + \
                    " (median spectrum)"
            ax = ximplot(median_spectrum, title=title, show=False,
                         plot_bbox=(self.bb_nc1_orig, self.bb_nc2_orig))
            ymin = median_spectrum.min()
            ymax = median_spectrum.max()
            dy = ymax-ymin
            ymin -= dy/20.
            ymax += dy/20.
            ax.set_ylim([ymin, ymax])
            # plot arc line peaks
            if fxpeaks is not None:
                plt.plot(fxpeaks + self.bb_nc1_orig,
                         median_spectrum[(fxpeaks+0.5).astype(int)], 'ro')
            # plot wavelength of each identified line
            for xpos, reference in zip(self.solution_wv.xpos,
                                self.solution_wv.reference):
                ax.text(xpos, median_spectrum[int(xpos+0.5)-1]+dy*0.01,
                        str(reference), fontsize=8,
                        horizontalalignment='center')
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

    def dict_arc_lines_slitlet(self):
        """Store identified arc lines information into dictionary.

        Determine and store the location of each identified arc line:
        location in the middle spectrum trail, slope of the fitted
        arc line, corresponding wavelength, as well as the
        intersection of the arc lines with the lower and upper
        boundaries.

        """

        # protections
        number_spectrum_trails = len(self.list_spectrum_trails)
        if number_spectrum_trails == 0:
            raise ValueError("Number of spectrum trails is 0")
        if self.list_arc_lines is None:
            raise ValueError("Arc lines not sought")
        number_arc_lines = len(self.list_arc_lines)
        if number_arc_lines == 0:
            raise ValueError("Number of available arc lines is 0")
        if self.solution_wv is None:
            raise ValueError("Missing wavelength calibration")
        if number_arc_lines != self.solution_wv.nlines_arc:
            print(self)
            raise ValueError("Number of arc lines is different to " +
                             "the number of wavelength calibrated lines")

        # intersection of each arc line with middle spectrum trail,
        # as well as the intersections with the lower and upper
        # boundaries
        max_error_in_pixels = 2.0
        tmp_dict = {}
        i_eff = -1
        for i in range(number_arc_lines):
            arcline = self.list_arc_lines[i]
            # approximate location of the solution
            expected_x = (arcline.xlower_line + arcline.xupper_line) / 2.0
            middle_spectrail = self.list_spectrum_trails[0]
            # composition of polynomials to find intersection as one of
            # the roots of a new polynomial
            rootfunct = arcline.poly_funct(middle_spectrail.poly_funct)
            rootfunct.coef[1] -= 1
            # compute roots to find solution
            tmp_xroots = rootfunct.roots()
            # take the nearest root to the expected location
            xroot = tmp_xroots[np.abs(tmp_xroots - expected_x).argmin()]
            if np.isreal(xroot):
                xroot = xroot.real
            else:
                raise ValueError("xroot=" + str(xroot) +
                                 " is a complex number")
            yroot = middle_spectrail.poly_funct(xroot)
            # check that the intersection point is very close to the
            # location of the line employed to determine the wavelength
            # calibration
            if abs(self.solution_wv.xpos[i] - xroot) > max_error_in_pixels:
                print("line number, from 0 to (number_arc_lines-1):", i)
                print("xpos.:", self.solution_wv.xpos[i])
                print("xroot:", xroot)
                print("WARNING: xpos and xroot are too different." +
                      " Ignoring arc line.")
            else:
                # compute intersection with lower boundary
                spectrail = self.list_spectrum_trails[1]
                rootfunct = arcline.poly_funct(spectrail.poly_funct)
                rootfunct.coef[1] -= 1
                tmp_xroots = rootfunct.roots()
                xroot1 = tmp_xroots[np.abs(tmp_xroots - expected_x).argmin()]
                if np.isreal(xroot1):
                    xroot1 = xroot1.real
                else:
                    raise ValueError("xroot1=" + str(xroot1) +
                                     " is a complex number")
                yroot1 = spectrail.poly_funct(xroot1)
                # compute intersection with upper boundary
                spectrail = self.list_spectrum_trails[2]
                rootfunct = arcline.poly_funct(spectrail.poly_funct)
                rootfunct.coef[1] -= 1
                tmp_xroots = rootfunct.roots()
                xroot2 = tmp_xroots[np.abs(tmp_xroots - expected_x).argmin()]
                if np.isreal(xroot2):
                    xroot2 = xroot2.real
                else:
                    raise ValueError("xroot2=" + str(xroot2) +
                                     " is a complex number")
                yroot2 = spectrail.poly_funct(xroot2)
                # store results to be saved into dictionary
                if self.debugplot >= 10:
                    print(i, self.grism_name, self.slitlet_number,
                          self.date_obs,
                          self.csu_bar_slit_center,
                          self.solution_wv.reference[i],
                          xroot, yroot, arcline.poly_funct.coef[1])
                i_eff += 1
                tmp_dict['arcline' + str(i_eff).zfill(4)] = {
                    'reference' : self.solution_wv.reference[i],
                    'slope' : arcline.poly_funct.coef[1],
                    # intersection with middle spectrail
                    'xpos_0' : xroot,
                    'ypos_0' : yroot,
                    # intersection with lower boundary
                    'xpos_1' : xroot1,
                    'ypos_1' : yroot1,
                    # intersection with upper boundary
                    'xpos_2' : xroot2,
                    'ypos_2' : yroot2
                }
        tmp_dict['csu_bar_slit_center'] = self.csu_bar_slit_center
        tmp_dict['csu_bar_slit_width'] = self.csu_bar_slit_width
        tmp_dict['rotang'] = self.rotang
        tmp_dict['number_arc_lines'] = i_eff + 1
        tmp_dict['wcal_crpix1'] = self.solution_wv.crpix1_linear
        tmp_dict['wcal_crval1'] = self.solution_wv.crval1_linear
        tmp_dict['wcal_crmin1'] = self.solution_wv.crmin1_linear
        tmp_dict['wcal_crmax1'] = self.solution_wv.crmax1_linear
        tmp_dict['wcal_cdelt1'] = self.solution_wv.cdelt1_linear
        # the numpy array must be converted into a list to be exported
        # within a json file
        tmp_dict['wcal_poly_coeff'] = self.solution_wv.coeff.tolist()
        tmp_dict['z_info1'] = os.getlogin() + '@' + socket.gethostname()
        tmp_dict['z_info2'] = datetime.now().isoformat()

        return tmp_dict



