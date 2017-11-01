from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
from copy import deepcopy
import json
from matplotlib.patches import Rectangle
import numpy as np
from scipy import ndimage
from scipy.signal import medfilt
from skimage import restoration
from skimage import transform
import sys

from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection
from numina.array.display.ximplotxy import ximplotxy
from numina.array.display.ximshow import ximshow
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.wavecalib.__main__ import read_wv_master_file
from numina.array.wavecalib.__main__ import wvcal_spectrum
from numina.array.wavecalib.arccalibration import refine_arccalibration
from numina.array.wavecalib.peaks_spectrum import find_peaks_spectrum
from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum

from emirdrp.core import EMIR_NBARS

from ccd_line import ArcLine
from ccd_line import intersection_spectrail_arcline
from csu_configuration import CsuConfiguration
from csu_configuration import merge_odd_even_csu_configurations
from dtu_configuration import DtuConfiguration
from emir_definitions import NAXIS1_EMIR
from emir_definitions import NAXIS2_EMIR
from fit_boundaries import bound_params_from_dict
from fit_boundaries import expected_distorted_boundaries
from fit_boundaries import expected_distorted_frontiers
from rescale_array_to_z1z2 import rescale_array_to_z1z2
from rescale_array_to_z1z2 import rescale_array_from_z1z2
from save_ndarray_to_fits import save_ndarray_to_fits

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


class Slitlet2D(object):
    """Slitlet2D class definition.

    It is important to distinguish between boundaries (the slitlet
    region when useful information is available) and frontiers (which
    define the real separation between consecutive slitlets when no gap
    between them is considered).

    Parameters
    ----------
    islitlet : int
        Slitlet number.
    params : :class:`~lmfit.parameter.Parameters`
        Parameters to be employed in the prediction of the distorted
        boundaries.
    parmodel : str
        Model to be assumed. Allowed values are 'longslit' and
        'multislit'.
    csu_conf : CsuConfiguration object
        Instance of CsuConfiguration.
    ymargin : int
        Extra number of pixels above and below the enclosing rectangle
        that defines the slitlet bounding box.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Attributes
    ----------
    islitlet : int
        Slitlet number.
    csu_bar_left : float
        Location (mm) of the left bar for each slitlet.
    csu_bar_right : list of floats
        Location (mm) of the right bar for each slitlet, using the
        same origin employed for csu_bar_left (which is not the
        value stored in the FITS keywords.
    csu_bar_slit_center : list of floats
        Middle point (mm) in between the two bars defining a slitlet.
    csu_bar_slit_width : list of floats
        Slitlet width (mm), computed as the distance between the two
        bars defining the slitlet.
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
    x0_reference : float
        X coordinate where the rectified y0_reference_middle is computed
        as the Y coordinate of the spectrum trails. The same value
        is used for all the available spectrum trails.
    y0_reference_lower: float
        Y coordinate corresponding to the lower spectrum trail computed
        at x0_reference. This value is employed as the Y coordinate of
        the lower spectrum trail of the rectified slitlet.
    y0_reference_middle: float
        Y coordinate corresponding to the middle spectrum trail computed
        at x0_reference. This value is employed as the Y coordinate of
        the middle spectrum trail of the rectified slitlet.
    y0_reference_upper: float
        Y coordinate corresponding to the upper spectrum trail computed
        at x0_reference. This value is employed as the Y coordinate of
        the upper spectrum trail of the rectified slitlet.
    y0_frontier_lower: float
        Y coordinate corresponding to the lower frontier computed at
        x0_reference.
    y0_frontier_upper: float
        Y coordinate corresponding to the upper frontier computed at
        x0_reference.
    list_spectrails: list of SpectrumTrail instances
        List of spectrum trails defined.
    list_frontiers: list of SpectrumTrail instances
        List of spectrum trails defining the slitlet frontiers.
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
    ttd_order : int or None
        Polynomial order corresponding to the rectification
        transformation. It is None until the coefficients ttd_aij and
        ttd_bij have been computed.
    ttd_aij : numpy array
        Polynomial coefficents corresponding to the rectification
        transformation coefficients a_ij.
    ttd_bij : numpy array
        Polynomial coefficents corresponding to the rectification
        transformation coefficients b_ij.
    ttd_order_smoothed : int or None
        Polynomial order corresponding to the smoothed rectification
        transformation. It is None until the coefficients
        ttd_aij_smoothed and ttd_bij_smoothed have been computed.
    ttd_aij_smoothed : numpy array
        Polynomial coefficents corresponding to the rectification
        transformation coefficients a_ij interpolated with a smooth
        polynomial variation as a function of y0_reference_middle.
    ttd_bij_smoothed : numpy array
        Polynomial coefficents corresponding to the rectification
        transformation coefficients b_ij interpolated with a smooth
        polynomial variation as a function of y0_reference_middle.
    wpoly_initial : Polynomial instance
        Initial wavelength calibration polynomial, providing the
        wavelength as a function of pixel number (running from 1 to
        NAXIS1).
    wpoly_refined : Polynomial instance
        Refined wavelength calibration polynomial, providing the
        wavelength as a function of pixel number (running from 1 to
        NAXIS1), or None (when the fit cannot be obtained).
    wpoly_refined_smoothed : Polynomial instance
        Refined and smoothed wavelength calibration polynomial,
        providing the wavelength as a function of pixel number (running
        from 1 to NAXIS1), or None (when the fit cannot be obtained).
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    """

    def __init__(self, islitlet, params, parmodel, csu_conf, ymargin=10,
                 debugplot=0):

        # slitlet number
        self.islitlet = islitlet
        self.csu_bar_left = csu_conf.csu_bar_left[islitlet - 1]
        self.csu_bar_right = csu_conf.csu_bar_right[islitlet - 1]
        self.csu_bar_slit_center = csu_conf.csu_bar_slit_center[islitlet - 1]
        self.csu_bar_slit_width = csu_conf.csu_bar_slit_width[islitlet - 1]

        # horizontal bounding box
        self.bb_nc1_orig = 1
        self.bb_nc2_orig = NAXIS1_EMIR

        # reference abscissa
        self.x0_reference = float(NAXIS1_EMIR) / 2.0 + 0.5  # single float

        # compute spectrum trails and store in which order they are computed
        self.i_lower_spectrail = 0
        self.i_middle_spectrail = 1
        self.i_upper_spectrail = 2
        self.list_spectrails = expected_distorted_boundaries(
                islitlet, self.csu_bar_slit_center,
                [0, 0.5, 1], params, parmodel,
                numpts=101, deg=5, debugplot=0
            )
        # update y_rectified computed at x0_reference
        for spectrail in self.list_spectrails:
            spectrail.y_rectified = spectrail.poly_funct(self.x0_reference)

        # define reference ordinates using lower, middle and upper spectrails
        # evaluated at x0_reference
        self.y0_reference_lower = \
            self.list_spectrails[self.i_lower_spectrail].y_rectified
        self.y0_reference_middle = \
            self.list_spectrails[self.i_middle_spectrail].y_rectified
        self.y0_reference_upper = \
            self.list_spectrails[self.i_upper_spectrail].y_rectified

        # compute frontiers (lower and upper)
        self.list_frontiers = expected_distorted_frontiers(
            islitlet, self.csu_bar_left,
            params, parmodel,
            numpts=101, deg=5, debugplot=0
        )
        # update y_rectified computed at x0_reference
        for spectrail in self.list_frontiers:
            spectrail.y_rectified = spectrail.poly_funct(self.x0_reference)

        # define frontier ordinates at x0_reference
        self.y0_frontier_lower = self.list_frontiers[0].y_rectified
        self.y0_frontier_upper = self.list_frontiers[1].y_rectified

        # determine vertical bounding box
        xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
        ylower = self.list_spectrails[self.i_lower_spectrail].poly_funct(xdum)
        yupper = self.list_spectrails[self.i_upper_spectrail].poly_funct(xdum)
        self.bb_ns1_orig = int(ylower.min() + 0.5) - ymargin
        if self.bb_ns1_orig < 1:
            self.bb_ns1_orig = 1
        self.bb_ns2_orig = int(yupper.max() + 0.5) + ymargin
        if self.bb_ns2_orig > NAXIS2_EMIR:
            self.bb_ns2_orig = NAXIS2_EMIR

        # place holder for still undefined members
        self.list_arc_lines = None
        self.x_inter_orig = None
        self.y_inter_orig = None
        self.x_inter_rect = None
        self.y_inter_rect = None
        self.ttd_order = None
        self.ttd_aij = None
        self.ttd_bij = None
        self.ttd_order_smoothed = None
        self.ttd_aij_smoothed = None
        self.ttd_bij_smoothed = None
        self.wpoly_initial = None
        self.wpoly_refined = None
        self.wpoly_refined_smoothed = None

        # debugplot
        self.debugplot = debugplot

    def __repr__(self):
        """Define printable representation of a Slitlet2D instance."""

        # string with all the information
        output = "<Slilet2D instance>\n" + \
            "- islitlet....................: " + \
                 str(self.islitlet) + "\n" + \
            "- csu_bar_left................: " + \
                 str(self.csu_bar_left) + "\n" + \
            "- csu_bar_right...............: " + \
                 str(self.csu_bar_right) + "\n" + \
            "- csu_bar_slit_center.........: " + \
                 str(self.csu_bar_slit_center) + "\n" + \
            "- csu_bar_slit_width..........: " + \
                 str(self.csu_bar_slit_width) + "\n" + \
            "- x0_reference................: " + \
                 str(self.x0_reference) + "\n" + \
            "- y0_reference_lower..........: " + \
                str(self.y0_reference_lower) + "\n" + \
            "- y0_reference_middle.........: " + \
                 str(self.y0_reference_middle) + "\n" + \
            "- y0_reference_upper..........: " + \
                 str(self.y0_reference_upper) + "\n" + \
            "- y0_frontier_lower..........: " + \
                 str(self.y0_frontier_lower) + "\n" + \
            "- y0_frontier_upper..........: " + \
                 str(self.y0_frontier_upper) + "\n" + \
            "- bb_nc1_orig.................: " + \
                 str(self.bb_nc1_orig) + "\n" + \
            "- bb_nc2_orig.................: " + \
                 str(self.bb_nc2_orig) + "\n" + \
            "- bb_ns1_orig.................: " + \
                 str(self.bb_ns1_orig) + "\n" + \
            "- bb_ns2_orig.................: " + \
                 str(self.bb_ns2_orig) + "\n" + \
            "- lower spectrail.poly_funct..:\n\t" + \
                 str(self.list_spectrails[self.i_lower_spectrail].poly_funct)\
                 + "\n" + \
            "- middle spectrail.poly_funct.:\n\t" + \
                 str(self.list_spectrails[self.i_middle_spectrail].poly_funct)\
                 + "\n" + \
            "- upper spectrail.poly_funct..:\n\t" + \
                 str(self.list_spectrails[self.i_upper_spectrail].poly_funct)\
                 + "\n" + \
            "- lower frontier.poly_funct...:\n\t" + \
                str(self.list_frontiers[0].poly_funct) + "\n" + \
            "- upper frontier.poly_funct...:\n\t" + \
                str(self.list_frontiers[1].poly_funct) + "\n"

        if self.list_arc_lines is None:
            number_arc_lines = None
        else:
            number_arc_lines = len(self.list_arc_lines)

        output += "- num. of associated arc lines: " + \
                  str(number_arc_lines) + "\n"

        for dumval, dumlab in zip(
                [self.x_inter_orig, self.y_inter_orig,
                 self.x_inter_rect, self.y_inter_rect],
                ["x_inter_orig", "y_inter_orig",
                 "x_inter_rect", "y_inter_rect"]
        ):
            if dumval is None:
                output += "- " + dumlab + "................: None\n"
            else:
                output += "- " + dumlab + "................: " + \
                          str(len(dumval)) + " values defined\n\t[" + \
                          str(dumval[0]) + ", ..., " + \
                          str(dumval[-1]) + "]\n"
        output += \
            "- ttd_order..........: " + str(self.ttd_order) + "\n" + \
            "- ttd_aij............:\n\t" + str(self.ttd_aij) + "\n" + \
            "- ttd_bij............:\n\t" + str(self.ttd_bij) + "\n" + \
            "- ttd_order_smoothed.: " + \
            str(self.ttd_order_smoothed) + "\n" + \
            "- ttd_aij_smoothed...:\n\t" + \
            str(self.ttd_aij_smoothed) + "\n" + \
            "- ttd_bij_smoothed...:\n\t" + \
            str(self.ttd_bij_smoothed) + "\n" + \
            "- wpoly_initial.........:\n\t" + \
            str(self.wpoly_initial) + "\n" + \
            "- wpoly_refined.........:\n\t" + \
            str(self.wpoly_refined) + "\n" + \
            "- wpoly_refined_smoothed:\n\t" + \
            str(self.wpoly_refined_smoothed) + "\n" + \
            "- debugplot...................: " + \
            str(self.debugplot)

        return output

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

        # extract slitlet region
        slitlet2d = image_2k2k[(self.bb_ns1_orig - 1):self.bb_ns2_orig,
                               (self.bb_nc1_orig - 1):self.bb_nc2_orig]

        # transform to float
        slitlet2d = slitlet2d.astype(np.float)

        # display slitlet2d with boundaries and middle spectrum trail
        if abs(self.debugplot) in [21, 22]:
            ax = ximshow(slitlet2d, title="Slitlet#" + str(self.islitlet),
                         first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                         show=False)
            xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
            ylower = \
                self.list_spectrails[self.i_lower_spectrail].poly_funct(xdum)
            ax.plot(xdum, ylower, 'b-')
            ymiddle = \
                self.list_spectrails[self.i_middle_spectrail].poly_funct(xdum)
            ax.plot(xdum, ymiddle, 'b--')
            yupper = \
                self.list_spectrails[self.i_upper_spectrail].poly_funct(xdum)
            ax.plot(xdum, yupper, 'b-')
            ylower_frontier = self.list_frontiers[0].poly_funct(xdum)
            ax.plot(xdum, ylower_frontier, 'b:')
            yupper_frontier = self.list_frontiers[1].poly_funct(xdum)
            ax.plot(xdum, yupper_frontier, 'b:')
            pause_debugplot(debugplot=self.debugplot, pltshow=True)

        # return slitlet image
        return slitlet2d

    def locate_unknown_arc_lines(self, slitlet2d,
                                 times_sigma_threshold=15,
                                 minimum_threshold=None,
                                 delta_x_max=30,
                                 delta_y_min=30,
                                 min_dist_from_middle=15):
        """Determine the location of known arc lines in slitlet.

        Parameters
        ----------
        slitlet2d : 2d numpy array, float
            Image containing the 2d slitlet image.
        times_sigma_threshold : float
            Times (robust) sigma above the median of the image to look
            for arc lines.
        minimum_threshold : float or None
            Minimum threshold to look for arc lines.
        delta_x_max : float
            Maximum size of potential arc line in the X direction.
        delta_y_min : float
            Minimum size of potential arc line in the Y direction.
        min_dist_from_middle : float
            Minimum Y distance from the middle spectrum trail to the
            extreme of the potential arc line. This constraint avoid
            detecting arc line reflections as bone fide arc lines.

        """

        # smooth denoising of slitlet2d
        slitlet2d_rs, coef_rs = rescale_array_to_z1z2(slitlet2d, z1z2=(-1, 1))
        slitlet2d_dn = restoration.denoise_nl_means(slitlet2d_rs,
                                                    patch_size=3,
                                                    patch_distance=2,
                                                    multichannel=False)
        slitlet2d_dn = rescale_array_from_z1z2(slitlet2d_dn, coef_rs)

        # compute basic statistics
        q25, q50, q75 = np.percentile(slitlet2d_dn, q=[25.0, 50.0, 75.0])
        sigmag = 0.7413 * (q75 - q25)  # robust standard deviation
        if abs(self.debugplot) >= 10:
            q16, q84 = np.percentile(slitlet2d_dn, q=[15.87, 84.13])
            print('>>> q16...:', q16)
            print('>>> q25...:', q25)
            print('>>> q50...:', q50)
            print('>>> q75...:', q75)
            print('>>> q84...:', q84)
            print('>>> sigmaG:', sigmag)
        if abs(self.debugplot) in [21, 22]:
            # display initial image with zscale cuts
            title = "Slitlet#" + str(self.islitlet) + \
                    " (locate_unknown_arc_lines, step #1)"
            ximshow(slitlet2d, title=title,
                    first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                    debugplot=self.debugplot)
            # display denoised image with zscale cuts
            title = "Slitlet#" + str(self.islitlet) + \
                    " (locate_unknown_arc_lines, step #2)"
            ximshow(slitlet2d_dn, title=title,
                    first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                    debugplot=self.debugplot)
            # display image with different cuts
            z1z2 = (q50 + times_sigma_threshold * sigmag,
                    q50 + 2 * times_sigma_threshold * sigmag)
            title = "Slitlet#" + str(self.islitlet) + \
                    " (locate_unknown_arc_lines, step #3)"
            ximshow(slitlet2d_dn, title=title, z1z2=z1z2,
                    first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                    debugplot=self.debugplot)

        # determine threshold (using the maximum of q50 + t *sigmag or
        # minimum_threshold)
        threshold = q50 + times_sigma_threshold * sigmag
        if minimum_threshold is not None:
            if minimum_threshold > threshold:
                threshold = minimum_threshold

        # identify objects in slitlet2d above threshold
        labels2d_objects, no_objects = ndimage.label(slitlet2d_dn > threshold)
        if abs(self.debugplot) >= 10:
            print("Number of objects initially found:", no_objects)
        if abs(self.debugplot) in [21, 22]:
            # display all objects identified in the image
            title = "Slitlet#" + str(self.islitlet) + \
                    " (locate_unknown_arc_lines, step #4)"
            z1z2 = (labels2d_objects.min(), labels2d_objects.max())
            ximshow(labels2d_objects, title=title,
                    first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                    cbar_label="Object number",
                    z1z2=z1z2, cmap="nipy_spectral",
                    debugplot=self.debugplot)

        # select arc lines by imposing the criteria based on the
        # dimensions of the detected objects and the intersection with
        # the middle spectrum trail
        slices_possible_arc_lines = ndimage.find_objects(labels2d_objects)
        slices_ok = np.repeat([False], no_objects)  # flag
        for i in range(no_objects):
            if abs(self.debugplot) >= 10:
                print('object', i + 1,
                      '[in np.array coordinates]:',
                      slices_possible_arc_lines[i])
            slice_x = slices_possible_arc_lines[i][1]
            slice_y = slices_possible_arc_lines[i][0]
            # note that the width computation doesn't require to
            # add +1 since slice_x.stop (and slice_y.stop) is
            # already the upper limit +1 (in np.array coordinates)
            delta_x = slice_x.stop - slice_x.start
            delta_y = slice_y.stop - slice_y.start
            # dimensions criterion
            if delta_x <= delta_x_max and delta_y >= delta_y_min:
                # intersection with middle spectrum trail criterion;
                # note that slice_x and slice_y are given in np.array
                # coordinates and are transformed into image coordinates;
                # in addition, -0.5 shift the origin to the lower left
                # corner of the pixel
                xini_slice = slice_x.start + self.bb_nc1_orig - 0.5
                xmiddle_slice = xini_slice + delta_x / 2
                polydum = \
                    self.list_spectrails[self.i_middle_spectrail].poly_funct
                ymiddle_slice = polydum(xmiddle_slice)
                yini_slice = slice_y.start + self.bb_ns1_orig - 0.5
                yend_slice = yini_slice + delta_y
                if yini_slice + min_dist_from_middle <= ymiddle_slice <= \
                        yend_slice - min_dist_from_middle:
                    slices_ok[i] = True

        # generate list with ID of arc lines (note that first object is
        # number 0 and not 1)
        list_slices_ok = []
        for i in range(no_objects):
            if slices_ok[i]:
                list_slices_ok.append(i + 1)
        number_arc_lines = len(list_slices_ok)
        if abs(self.debugplot) >= 10:
            print("\nNumber of arc lines initially identified is:",
                  number_arc_lines)
            if number_arc_lines > 0:
                print("Slice ID of lines passing the selection:\n",
                      list_slices_ok)
        if number_arc_lines == 0:
            return

        # display arc lines
        if abs(self.debugplot) in [21, 22]:
            # display all objects identified in the image
            title = "Slitlet#" + str(self.islitlet) + \
                    " (locate_unknown_arc_lines, step #5)"
            z1z2 = (labels2d_objects.min(),
                    labels2d_objects.max())
            ax = ximshow(labels2d_objects, show=False, title=title,
                         first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                         cbar_label="Object number",
                         z1z2=z1z2, cmap="nipy_spectral",
                         debugplot=self.debugplot)
            # plot rectangle around identified arc lines
            for i in range(no_objects):
                if slices_ok[i]:
                    slice_x = slices_possible_arc_lines[i][1]
                    slice_y = slices_possible_arc_lines[i][0]
                    # note that slice_x and slice_y are given in np.array
                    # coordinates and are transformed into image coordinates;
                    # in addition, -0.5 shift the origin to the lower left
                    # corner of the pixel
                    xini_slice = slice_x.start + self.bb_nc1_orig - 0.5
                    yini_slice = slice_y.start + self.bb_ns1_orig - 0.5
                    # note that the width computation doesn't require to
                    # add +1 since slice_x.stop (and slice_y.stop) is
                    # already the upper limit +1 (in np.array coordinates)
                    xwidth_slice = slice_x.stop - slice_x.start
                    ywidth_slice = slice_y.stop - slice_y.start
                    rect = Rectangle((xini_slice, yini_slice),
                                     xwidth_slice, ywidth_slice,
                                     edgecolor='w', facecolor='none')
                    ax.add_patch(rect)
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

        # adjust individual arc lines passing the initial selection
        self.list_arc_lines = []  # list of ArcLines
        for k in range(number_arc_lines):  # fit each arc line
            # select points to be fitted for a particular arc line
            xy_tmp = np.where(labels2d_objects == list_slices_ok[k])
            x_tmp = xy_tmp[1] + self.bb_nc1_orig  # use image coordinates
            y_tmp = xy_tmp[0] + self.bb_ns1_orig  # use image coordinates
            w_tmp = slitlet2d_dn[xy_tmp]
            # declare new ArcLine instance
            arc_line = ArcLine()
            # define new ArcLine using a weighted fit
            # (note that it must be X vs Y)
            arc_line.fit(x=x_tmp, y=y_tmp, deg=1, w=w_tmp, y_vs_x=False)
            # update list with identified ArcLines
            self.list_arc_lines.append(arc_line)

        # remove arc lines with unexpected slopes
        yfit = np.array([self.list_arc_lines[k].poly_funct.coef[1]
                         for k in range(number_arc_lines)])
        xfit = np.zeros(number_arc_lines)
        # intersection between middle spectrum trail and arc line
        for k in range(number_arc_lines):
            arcline = self.list_arc_lines[k]
            xfit[k], ydum = intersection_spectrail_arcline(
                self.list_spectrails[self.i_middle_spectrail], arcline
            )

        # fit slope versus x-coordinate of the intersection of the arc line
        # with the middle spectrum trail
        polydum, residum, rejected = polfit_residuals_with_sigma_rejection(
            x=xfit, y=yfit, deg=5, times_sigma_reject=4.0,
            xlabel='arc line center (islitlet #' + str(self.islitlet) + ')',
            ylabel='arc line slope', debugplot=0
        )
        # remove rejected arc lines
        if len(rejected) > 0:
            if abs(self.debugplot) >= 10:
                print('Rejecting', sum(rejected),
                      'arc lines with suspicious slopes: Slice ID',
                      [list_slices_ok[k] for k in range(number_arc_lines)
                       if rejected[k]])
            self.list_arc_lines = \
                [self.list_arc_lines[k] for k in range(number_arc_lines)
                 if not rejected[k]]
            # recompute number of arc lines
            number_arc_lines = len(self.list_arc_lines)
            if abs(self.debugplot) >= 10:
                print("\nNumber of arc lines finally identified is:",
                      number_arc_lines)

        if abs(self.debugplot) >= 20:
            # print list of arc lines
            print('\nlist_arc_lines:')
            for k in range(number_arc_lines):
                print(k, '->', self.list_arc_lines[k], '\n')

        # display results
        if abs(self.debugplot) in [21, 22]:
            # generate mask with all the arc-line points passing the selection
            mask_arc_lines = np.zeros_like(slitlet2d_dn)
            for k in list_slices_ok:
                mask_arc_lines[labels2d_objects == k] = 1
            # compute image with only the arc lines passing the selection
            labels2d_arc_lines = labels2d_objects * mask_arc_lines
            # display background image with filtered arc lines
            title = "Slitlet#" + str(self.islitlet) + \
                    " (locate_unknown_arc_lines, step #6)"
            z1z2 = (labels2d_arc_lines.min(),
                    labels2d_arc_lines.max())
            ax = ximshow(labels2d_arc_lines, show=False,
                         first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                         cbar_label="Object number",
                         title=title, z1z2=z1z2, cmap="nipy_spectral",
                         debugplot=self.debugplot)
            # plot weighted fit for each arc line (note that the fit is
            # X vs Y)
            for k in range(number_arc_lines):
                xpol, ypol = self.list_arc_lines[k].linspace_pix()
                ax.plot(xpol, ypol, 'g--')
            # display lower and upper points of each arc line
            x_tmp = [arc_line.xlower_line for arc_line in self.list_arc_lines]
            y_tmp = [arc_line.ylower_line for arc_line in self.list_arc_lines]
            ax.plot(x_tmp, y_tmp, 'w+')
            x_tmp = [arc_line.xupper_line for arc_line in self.list_arc_lines]
            y_tmp = [arc_line.yupper_line for arc_line in self.list_arc_lines]
            ax.plot(x_tmp, y_tmp, 'w+')
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

    def xy_spectrail_arc_intersections(self, slitlet2d=None):
        """Compute intersection points of spectrum trails with arc lines.

        The member list_arc_lines is updated with new keyword:keyval
        values for each arc line.

        Parameters
        ----------
        slitlet2d : 2d numpy array
            Slitlet image to be displayed with the computed boundaries
            and intersecting points overplotted. This argument is
            optional.

        """

        # protections
        if self.list_arc_lines is None:
            raise ValueError("Arc lines not sought")
        number_spectrum_trails = len(self.list_spectrails)
        if number_spectrum_trails == 0:
            raise ValueError("Number of available spectrum trails is 0")
        number_arc_lines = len(self.list_arc_lines)
        if number_arc_lines == 0:
            raise ValueError("Number of available arc lines is 0")

        # intersection of the arc lines with the spectrum trails
        # (note: the coordinates are computed using pixel values,
        #  ranging from 1 to NAXIS1_EMIR, as given in the original
        #  image reference system ---not in the slitlet image reference
        #  system---)
        self.x_inter_rect = np.array([])  # rectified image coordinates
        self.y_inter_rect = np.array([])  # rectified image coordinates
        for arcline in self.list_arc_lines:
            # middle spectrum trail
            spectrail = self.list_spectrails[self.i_middle_spectrail]
            xroot, yroot = intersection_spectrail_arcline(
                spectrail=spectrail, arcline=arcline
            )
            arcline.x_rectified = xroot
            self.x_inter_rect = np.append(
                self.x_inter_rect, [xroot] * number_spectrum_trails
            )
            for spectrail in self.list_spectrails:
                self.y_inter_rect = np.append(
                    self.y_inter_rect, spectrail.y_rectified)
        #
        self.x_inter_orig = np.array([])  # original image coordinates
        self.y_inter_orig = np.array([])  # original image coordinates
        for arcline in self.list_arc_lines:
            for spectrail in self.list_spectrails:
                xroot, yroot = intersection_spectrail_arcline(
                    spectrail=spectrail, arcline=arcline
                )
                self.x_inter_orig = np.append(self.x_inter_orig, xroot)
                self.y_inter_orig = np.append(self.y_inter_orig, yroot)

        # display intersection points
        if abs(self.debugplot % 10) != 0 and slitlet2d is not None:
            # display image with zscale cuts
            title = "Slitlet#" + str(self.islitlet) + \
                    " (xy_spectrail_arc_intersections, step #1)"
            ax = ximshow(slitlet2d, title=title,
                         first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                         show=False)
            # spectrum trails
            for spectrail in self.list_spectrails:
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
            pause_debugplot(self.debugplot, pltshow=True)

    def estimate_tt_to_rectify(self, order, slitlet2d=None):
        """Estimate the polynomial transformation to rectify the image.

        Parameters
        ----------
        order : int
            Order of the polynomial transformation.
        slitlet2d : 2d numpy array
            Slitlet image to be displayed with the computed boundaries
            and intersecting points overplotted. This argument is
            optional.

        """

        # protections
        if self.x_inter_orig is None or \
                        self.y_inter_orig is None or \
                        self.x_inter_rect is None or \
                        self.y_inter_rect is None:
            raise ValueError('Intersection points not computed')

        npoints = len(self.x_inter_orig)
        if len(self.y_inter_orig) != npoints or \
            len(self.x_inter_rect) != npoints or \
            len(self.y_inter_rect) != npoints:
            raise ValueError('Unexpected different number of points')

        # IMPORTANT: correct coordinates from origin in order to manipulate
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
        if abs(self.debugplot) >= 10:
            print("x_scale:", x_scale)
            print("y_scale:", y_scale)
        x_inter_orig_scaled = x_inter_orig_shifted * x_scale
        y_inter_orig_scaled = y_inter_orig_shifted * y_scale
        x_inter_rect_scaled = x_inter_rect_shifted * x_scale
        y_inter_rect_scaled = y_inter_rect_shifted * y_scale

        if abs(self.debugplot) % 10 != 0:
            ax = ximplotxy(x_inter_orig_scaled, y_inter_orig_scaled,
                           show=False,
                           **{'marker': 'o', 'color': 'cyan',
                              'label': 'original', 'linestyle': ''})
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

        # solve 2 systems of equations with half number of unknowns each
        if order == 1:
            A = np.vstack([np.ones(npoints),
                           x_inter_rect_scaled,
                           y_inter_rect_scaled]).T
        elif order == 2:
            A = np.vstack([np.ones(npoints),
                           x_inter_rect_scaled,
                           y_inter_rect_scaled,
                           x_inter_rect_scaled ** 2,
                           x_inter_rect_scaled * y_inter_orig_scaled,
                           y_inter_rect_scaled ** 2]).T
        elif order == 3:
            A = np.vstack([np.ones(npoints),
                           x_inter_rect_scaled,
                           y_inter_rect_scaled,
                           x_inter_rect_scaled ** 2,
                           x_inter_rect_scaled * y_inter_orig_scaled,
                           y_inter_rect_scaled ** 2,
                           x_inter_rect_scaled ** 3,
                           x_inter_rect_scaled ** 2 * y_inter_rect_scaled,
                           x_inter_rect_scaled * y_inter_rect_scaled ** 2,
                           y_inter_rect_scaled ** 3]).T
        elif order == 4:
            A = np.vstack([np.ones(npoints),
                           x_inter_rect_scaled,
                           y_inter_rect_scaled,
                           x_inter_rect_scaled ** 2,
                           x_inter_rect_scaled * y_inter_orig_scaled,
                           y_inter_rect_scaled ** 2,
                           x_inter_rect_scaled ** 3,
                           x_inter_rect_scaled ** 2 * y_inter_rect_scaled,
                           x_inter_rect_scaled * y_inter_rect_scaled ** 2,
                           y_inter_rect_scaled ** 3,
                           x_inter_rect_scaled ** 4,
                           x_inter_rect_scaled ** 3 * y_inter_rect_scaled ** 1,
                           x_inter_rect_scaled ** 2 * y_inter_rect_scaled ** 2,
                           x_inter_rect_scaled ** 1 * y_inter_rect_scaled ** 3,
                           y_inter_rect_scaled ** 4]).T
        else:
            raise ValueError("Invalid order=" + str(order))
        ttd = transform.PolynomialTransform(
            np.vstack(
                [np.linalg.lstsq(A, x_inter_orig_scaled)[0],
                 np.linalg.lstsq(A, y_inter_orig_scaled)[0]]
            )
        )

        # reverse normalization to recover coefficients of the
        # transformation in the correct system
        factor = np.zeros_like(ttd.params[0])
        k = 0
        for i in range(order + 1):
            for j in range(i + 1):
                factor[k] = (x_scale**(i-j)) * (y_scale**j)
                k += 1
        self.ttd_order = order
        self.ttd_aij = ttd.params[0] * factor/x_scale
        self.ttd_bij = ttd.params[1] * factor/y_scale
        if self.debugplot >= 10:
            print("ttd_aij X:\n", self.ttd_aij)
            print("ttd_bij Y:\n", self.ttd_bij)

        # display slitlet with intersection points and grid indicating
        # the fitted transformation
        if abs(self.debugplot % 10) != 0 and slitlet2d is not None:
            # display image with zscale cuts
            title = "Slitlet#" + str(self.islitlet) + \
                    " (estimate_tt_to_rectify)"
            ax = ximshow(slitlet2d, title=title,
                         first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                         show=False)
            # intersection points
            ax.plot(self.x_inter_orig, self.y_inter_orig, 'co')
            ax.plot(self.x_inter_rect, self.y_inter_rect, 'bo')
            # grid with fitted transformation: spectrum trails
            xx = np.arange(0, self.bb_nc2_orig - self.bb_nc1_orig + 1,
                           dtype=np.float)
            for spectrail in self.list_spectrails:
                yy0 = spectrail.y_rectified
                yy = np.tile([yy0 - self.bb_ns1_orig], xx.size)
                ax.plot(xx + self.bb_nc1_orig, yy + self.bb_ns1_orig, "b")
                xxx, yyy = fmap(self.ttd_order, self.ttd_aij, self.ttd_bij,
                                xx, yy)
                ax.plot(xxx + self.bb_nc1_orig, yyy + self.bb_ns1_orig, "g")
            # grid with fitted transformation: arc lines
            ylower_line = \
                self.list_spectrails[self.i_lower_spectrail].y_rectified
            yupper_line = \
                self.list_spectrails[self.i_upper_spectrail].y_rectified
            n_points = int(yupper_line - ylower_line + 0.5) + 1
            yy = np.linspace(ylower_line - self.bb_ns1_orig,
                             yupper_line - self.bb_ns1_orig,
                             num=n_points,
                             dtype=np.float)
            for arc_line in self.list_arc_lines:
                xline = arc_line.x_rectified - self.bb_nc1_orig
                xx = np.array([xline] * n_points)
                ax.plot(xx + self.bb_nc1_orig, yy + self.bb_ns1_orig, "b")
                xxx, yyy = fmap(self.ttd_order, self.ttd_aij, self.ttd_bij,
                                xx, yy)
                ax.plot(xxx + self.bb_nc1_orig, yyy + self.bb_ns1_orig, "c")
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

    def rectify(self, slitlet2d, resampling, transformation):
        """Rectify slitlet using computed transformation.

        Parameters
        ----------
        slitlet2d : 2d numpy array, float
            Image containing the 2d slitlet image.
        resampling : int
            1: nearest neighbour, 2: flux preserving interpolation.
        transformation : int
            1: initial, 2: smoothed

        Returns
        -------
        slitlet2d_rect : 2d numpy array
            Rectified slitlet image.

        """

        if resampling not in [1, 2]:
            raise ValueError("Unexpected resampling value=" + str(resampling))

        if transformation not in [1, 2]:
            raise ValueError("Unexpected transformation value=" +
                             str(transformation))

        # initialize output array
        slitlet2d_rect = np.zeros_like(slitlet2d)
        naxis2, naxis1 = slitlet2d_rect.shape
        if naxis1 != self.bb_nc2_orig - self.bb_nc1_orig + 1:
            raise ValueError("Unexpected slitlet2d_rect naxis1")
        if naxis2 != self.bb_ns2_orig - self.bb_ns1_orig + 1:
            raise ValueError("Unexpected slitlet2d_rect naxis2")

        if resampling == 1:
            # pixel coordinates (rectified image); since the fmap function
            # below requires floats, these arrays must use dtype=np.float
            j = np.arange(0, naxis1, dtype=np.float)
            i = np.arange(0, naxis2, dtype=np.float)
            # the cartesian product of the previous 1D arrays could be stored
            # as np.transpose([xx,yy]), where xx and yy are computed as follows
            xx = np.tile(j, (len(i),))
            yy = np.repeat(i, len(j))
            # compute pixel coordinates in original (distorted) image
            if transformation == 1:
                xxx, yyy = fmap(self.ttd_order,
                                self.ttd_aij,
                                self.ttd_bij,
                                xx, yy)
            else:
                xxx, yyy = fmap(self.ttd_order_smoothed,
                                self.ttd_aij_smoothed,
                                self.ttd_bij_smoothed,
                                xx, yy)
            # round to nearest integer and cast to integer; note that the
            # rounding still provides a float, so the casting is required
            ixxx = np.rint(xxx).astype(np.int)
            iyyy = np.rint(yyy).astype(np.int)
            # determine pixel coordinates within available image
            lxxx = np.logical_and(ixxx >= 0, ixxx < naxis1)
            lyyy = np.logical_and(iyyy >= 0, iyyy < naxis2)
            lok = np.logical_and(lxxx, lyyy)
            # assign pixel values to rectified image
            ixx = xx.astype(np.int)[lok]
            iyy = yy.astype(np.int)[lok]
            ixxx = ixxx[lok]
            iyyy = iyyy[lok]
            slitlet2d_rect[iyy, ixx] = slitlet2d[iyyy, ixxx]
        else:
            raise ValueError("Sorry, this resampling method has not been"
                             " implemented yet!")

        if abs(self.debugplot % 10) != 0:
            title = "Slitlet#" + str(self.islitlet) + " (rectify)"
            ax = ximshow(slitlet2d_rect, title=title,
                         first_pixel=(self.bb_nc1_orig, self.bb_ns1_orig),
                         show=False)
            if self.list_arc_lines is not None:
                # intersection points
                ax.plot(self.x_inter_rect, self.y_inter_rect, 'bo')
            # grid with fitted transformation: spectrum trails
            xx = np.arange(0, self.bb_nc2_orig - self.bb_nc1_orig + 1,
                           dtype=np.float)
            for spectrail in self.list_spectrails:
                yy0 = spectrail.y_rectified
                yy = np.tile([yy0 - self.bb_ns1_orig], xx.size)
                ax.plot(xx + self.bb_nc1_orig, yy + self.bb_ns1_orig, "b")
            for spectrail in self.list_frontiers:
                yy0 = spectrail.y_rectified
                yy = np.tile([yy0 - self.bb_ns1_orig], xx.size)
                ax.plot(xx + self.bb_nc1_orig, yy + self.bb_ns1_orig, "b:")
            # grid with fitted transformation: arc lines
            ylower_line = \
                self.list_spectrails[self.i_lower_spectrail].y_rectified
            yupper_line = \
                self.list_spectrails[self.i_upper_spectrail].y_rectified
            n_points = int(yupper_line - ylower_line + 0.5) + 1
            yy = np.linspace(ylower_line - self.bb_ns1_orig,
                             yupper_line - self.bb_ns1_orig,
                             num=n_points,
                             dtype=np.float)
            if self.list_arc_lines is not None:
                for arc_line in self.list_arc_lines:
                    xline = arc_line.x_rectified - self.bb_nc1_orig
                    xx = np.array([xline] * n_points)
                    ax.plot(xx + self.bb_nc1_orig, yy + self.bb_ns1_orig, "b")
            # show plot
            pause_debugplot(self.debugplot, pltshow=True)

        return slitlet2d_rect

    def median_spectrum_from_rectified_image(self, slitlet2d_rect,
                                             sigma_gaussian_filtering=0,
                                             nwinwidth_initial=5,
                                             nwinwidth_refined=7,
                                             times_sigma_threshold=5,
                                             minimum_threshold=None,
                                             npix_avoid_border=0,
                                             nbrightlines=[0]):
        """Median spectrum and line peaks from rectified image.

        In order to avoid the line ghosts, the line peaks are identified
        independently in the upper and lower halves of the rectified
        image. The final peaks correspond to lines that appear in both
        spectra.

        Parameters
        ----------
        slitlet2d_rect : 2d numpy array
            Rectified slitlet image.
        sigma_gaussian_filtering : float
            Sigma of the gaussian filter to be applied to the spectrum
            in order to avoid problems with saturated lines. This
            filtering is skipped when this parameter is <= 0.
        nwinwidth_initial : int
            Width of the window where each peak must be found using
            the initial method (approximate)
        nwinwidth_refined : int
            Width of the window where each peak location will be
            refined.
        times_sigma_threshold : float
            Times (robust) sigma above the median of the image to set
            the minimum threshold when searching for line peaks.
        minimum_threshold : float or None
            Minimum value of the threshold.
        npix_avoid_border : int
            Number of pixels at the borders of the spectrum where peaks
            are not considered. If zero, the actual number will be
            given by nwinwidth_initial.
        nbrightlines : int or list of integers
            Maximum number of brightest lines to be employed in the
            wavelength calibration. If this value is 0, all the detected
            lines will be employed.

        Returns
        -------
        sp0 : 1d numpy array
            Median spectrum.
        fxpeaks : 1d numpy array
            Refined location of arc lines (in array index scale).

        """

        # protections
        naxis2, naxis1 = slitlet2d_rect.shape
        if naxis1 != self.bb_nc2_orig - self.bb_nc1_orig + 1:
            raise ValueError("Unexpected slitlet2d_rect naxis1")
        if naxis2 != self.bb_ns2_orig - self.bb_ns1_orig + 1:
            raise ValueError("Unexpected slitlet2d_rect naxis2")

        # lower, middle and upper spectrum trails
        ylower_line = \
            self.list_spectrails[self.i_lower_spectrail].y_rectified
        ymiddle_line = \
            self.list_spectrails[self.i_middle_spectrail].y_rectified
        yupper_line = \
            self.list_spectrails[self.i_upper_spectrail].y_rectified

        ilower = int(ylower_line + 0.5) - self.bb_ns1_orig
        imiddle = int(ymiddle_line + 0.5) - self.bb_ns1_orig
        iupper = int(yupper_line + 0.5) - self.bb_ns1_orig

        # median spectra using different image regions
        sp0_ini = np.median(slitlet2d_rect[ilower:(iupper + 1), :], axis=0)
        sp1_ini = np.median(slitlet2d_rect[ilower:(imiddle + 1), :], axis=0)
        sp2_ini = np.median(slitlet2d_rect[imiddle:(iupper + 1), :], axis=0)

        # gaussian filtering when requested (to avoid line saturation)
        if sigma_gaussian_filtering > 0:
            sp0 = ndimage.filters.gaussian_filter(
                sp0_ini,
                sigma=sigma_gaussian_filtering
            )
            sp1 = ndimage.filters.gaussian_filter(
                sp1_ini,
                sigma=sigma_gaussian_filtering
            )
            sp2 = ndimage.filters.gaussian_filter(
                sp2_ini,
                sigma=sigma_gaussian_filtering
            )
        else:
            sp0 = np.copy(sp0_ini)
            sp1 = np.copy(sp1_ini)
            sp2 = np.copy(sp2_ini)

        # compute threshold
        q25, q50, q75 = np.percentile(sp0, q=[25.0, 50.0, 75.0])
        sigma_g = 0.7413 * (q75 - q25)  # robust standard deviation
        threshold = q50 + times_sigma_threshold * sigma_g
        if abs(self.debugplot) >= 10:
            print("median...........:", q50)
            print("robuts std.......:", sigma_g)
            print("threshold........:", threshold)
        if minimum_threshold > threshold:
            threshold = minimum_threshold
        if abs(self.debugplot) >= 10:
            print("minimum threshold:", minimum_threshold)
            print("final threshold..:", threshold)

        # initial location of the peaks (integer values)
        ixpeaks0 = find_peaks_spectrum(sp0, nwinwidth=nwinwidth_initial,
                                       threshold=threshold,
                                       debugplot=self.debugplot)

        # peaks in the lower and upper regions
        ixpeaks1 = find_peaks_spectrum(sp1, nwinwidth=nwinwidth_initial,
                                       threshold=threshold,
                                       debugplot=self.debugplot)
        ixpeaks2 = find_peaks_spectrum(sp2, nwinwidth=nwinwidth_initial,
                                       threshold=threshold,
                                       debugplot=self.debugplot)

        # the peaks are valid if the are also found in the lower and
        # upper regions (with a tolerance of +1 or -1 pixel)
        ixpeaks = []
        for ixpeak in ixpeaks0:
            l1 = ixpeak in np.concatenate((ixpeaks1, ixpeaks1+1, ixpeaks1-1))
            l2 = ixpeak in np.concatenate((ixpeaks2, ixpeaks2+1, ixpeaks2-1))
            if l1 and l2:
                ixpeaks.append(ixpeak)
        ixpeaks = np.array(ixpeaks)
        if abs(self.debugplot) >= 10:
            print("Merged initial list of peaks:\n", ixpeaks)

        # remove peaks too close to any of the borders of the spectrum
        if npix_avoid_border > 0:
            lok_ini = ixpeaks >= npix_avoid_border
            lok_end = ixpeaks <= len(sp0) - 1 - npix_avoid_border
            ixpeaks = ixpeaks[lok_ini * lok_end]

        # select a maximum number of brightest lines in each region
        if len(nbrightlines) == 1 and nbrightlines[0] == 0:
            pass
        else:
            if abs(self.debugplot) >= 10:
                print('nbrightlines =', nbrightlines)
                print('ixpeaks in whole spectrum:\n', ixpeaks)
            region_size = (naxis1-1)/len(nbrightlines)
            ixpeaks_filtered = np.array([], dtype=int)
            for iregion, nlines_in_region in enumerate(nbrightlines):
                if nlines_in_region > 0:
                    imin = int(iregion * region_size)
                    imax = int((iregion + 1) * region_size)
                    if iregion > 0:
                        imin += 1
                    ixpeaks_region = \
                        ixpeaks[np.logical_and(ixpeaks >= imin,
                                               ixpeaks <= imax)]
                    if len(ixpeaks_region) > 0:
                        peak_fluxes = sp0[ixpeaks_region]
                        spos = peak_fluxes.argsort()
                        ixpeaks_tmp = ixpeaks_region[spos[-nlines_in_region:]]
                        ixpeaks_tmp.sort()  # in-place sort
                        if abs(self.debugplot) >= 10:
                            print('ixpeaks in region........:\n', ixpeaks_tmp)
                        ixpeaks_filtered = np.concatenate((ixpeaks_filtered,
                                                           ixpeaks_tmp))
            ixpeaks = ixpeaks_filtered
            if abs(self.debugplot) >= 10:
                print('ixpeaks filtered.........:\n', ixpeaks)

        # refined location of the peaks (float values)
        fxpeaks, sxpeaks = refine_peaks_spectrum(sp0, ixpeaks,
                                                 nwinwidth=nwinwidth_refined,
                                                 method="gaussian")

        if abs(self.debugplot) % 10 != 0:
            x = np.arange(self.bb_nc1_orig, self.bb_nc2_orig + 1)
            title = "Slitlet#" + str(self.islitlet) + " (median spectrum)"
            ax = ximplotxy(x, sp1, show=False, title=title,
                           xlabel='pixel coordinate (from 1 to NAXIS1)',
                           ylabel='number of counts',
                           **{'marker': ' ', 'label': 'lower region'})
            ax.plot(x, sp2, label='upper region')
            ax.plot(x, sp0, label='whole region')
            # mark peak location
            ax.plot(ixpeaks + self.bb_nc1_orig,
                    sp0[ixpeaks], 'o', label="initial location")
            ax.plot(fxpeaks + self.bb_nc1_orig,
                    sp0[ixpeaks], 'o', label="refined location")
            ax.legend()
            pause_debugplot(self.debugplot, pltshow=True, tight_layout=False)

        # return median spectrum and refined peak location
        return sp0, fxpeaks


def ncoef_fmap(order):
    """Expected number of coefficients for the 2D polynomial transformation.

    Parameters
    ----------
    order : int
        Order of the 2D polynomial transformation.

    Returns
    -------
    ncoef : int
        Expected number of coefficients.

    """

    ncoef = 0
    for i in range(order + 1):
        for j in range(i + 1):
            ncoef += 1
    return ncoef


def fmap(order, aij, bij, x, y):
    """Evaluate the 2D polynomial transformation.

    Parameters
    ----------
    order : int
        Order of the polynomial transformation.
    aij : numpy array
        Polynomial coefficents corresponding to a_ij.
    bij : numpy array
        Polynomial coefficents corresponding to b_ij.
    x : numpy array or float
        X coordinate values where the transformation is computed. Note
        that these values correspond to array indices.
    y : numpy array or float
        Y coordinate values where the transformation is computed. Note
        that these values correspond to array indices.

    Returns
    -------
    u : numpy array or float
        U coordinate values.
    v : numpy array or float
        V coordinate values.

    """

    u = np.zeros_like(x)
    v = np.zeros_like(y)

    k = 0
    for i in range(order + 1):
        for j in range(i + 1):
            u += aij[k] * (x ** (i - j)) * (y ** j)
            v += bij[k] * (x ** (i - j)) * (y ** j)
            k += 1

    return u, v


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='test')
    # required arguments
    parser.add_argument("fitsfile_odd",
                        help="FITS file with odd-numbered slitlets",
                        type=argparse.FileType('r'))
    parser.add_argument("fitsfile_even",
                        help="FITS file with even-numbered slitlets",
                        type=argparse.FileType('r'))
    parser.add_argument("--tuple_slit_numbers", required=True,
                        help="Tuple n1[,n2[,step]] to define slitlet numbers")
    parser.add_argument("--fitted_bound_param", required=True,
                        help="Input JSON with fitted boundary parameters",
                        type=argparse.FileType('r'))
    parser.add_argument("--order_fmap", required=True,
                        help="Order of the 2D rectification transformation "
                             "(default=2)",
                        default=2, type=int)
    parser.add_argument("--wv_master_file", required=True,
                        help="TXT file containing wavelengths")
    parser.add_argument("--poldeg_initial", required=True,
                        help="Polynomial degree for initial calibration",
                        type=int)
    parser.add_argument("--poldeg_refined", required=True,
                        help="Polynomial degree for refined calibration",
                        type=int)
    parser.add_argument("--out_rect", required=True,
                        help="Rectified but not wavelength calibrated output "
                             "FITS file",
                        type=argparse.FileType('w'))
    # optional arguments
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting & debugging options"
                             " (default=0)",
                        default=0, type=int,
                        choices=DEBUGPLOT_CODES)
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")
    args = parser.parse_args(args)

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    # ------------------------------------------------------------------------

    # read slitlet numbers to be computed
    tmp_str = args.tuple_slit_numbers.split(",")
    if len(tmp_str) == 3:
        n1 = int(tmp_str[0])
        n2 = int(tmp_str[1])
        step = int(tmp_str[2])
    elif len(tmp_str) == 2:
        n1 = int(tmp_str[0])
        n2 = int(tmp_str[1])
        step = 1
    elif len(tmp_str) == 1:
        n1 = int(tmp_str[0])
        n2 = n1
        step = 1
    else:
        raise ValueError("Invalid tuple for slitlet numbers")
    if n1 < 1:
        raise ValueError("Invalid slitlet number < 1")
    if n2 > EMIR_NBARS:
        raise ValueError("Invalid slitlet number > EMIR_NBARS")
    if step <= 0:
        raise ValueError("Invalid step <= 0")
    list_slitlets = range(n1, n2 + 1, step)

    # read the CSU configuration from the two initial FITS files and merge
    # the corresponding configurations in a single one containing the
    # the information corresponding to the odd- and even-numbered slitlets
    # accordingly
    csu_conf_odd = CsuConfiguration()
    csu_conf_odd.define_from_fits(args.fitsfile_odd)
    csu_conf_even = CsuConfiguration()
    csu_conf_even.define_from_fits(args.fitsfile_even)
    csu_conf = merge_odd_even_csu_configurations(csu_conf_odd, csu_conf_even)
    if abs(args.debugplot) >= 10:
        print(csu_conf)
        pause_debugplot(args.debugplot)

    # read the DTU configuration from the two initial FITS files and check
    # that both configurations are identical
    dtu_conf_odd = DtuConfiguration()
    dtu_conf_odd.define_from_fits(args.fitsfile_odd)
    dtu_conf_even = DtuConfiguration()
    dtu_conf_even.define_from_fits(args.fitsfile_even)
    if dtu_conf_odd == dtu_conf_even:
        dtu_conf = deepcopy(dtu_conf_odd)
        if abs(args.debugplot) >= 10:
            print(dtu_conf)
            pause_debugplot(args.debugplot)
    else:
        print("--> DTU config. odd-numbered slitlets:\n", dtu_conf_odd)
        print("--> DTU config. even-numbered slitlets:\n", dtu_conf_even)
        raise ValueError("DTU configuration from odd-numbered and "
                         "even-numbered slitlets FITS files are different")

    fitted_bound_param = json.loads(open(args.fitted_bound_param.name).read())
    parmodel = fitted_bound_param['meta-info']['parmodel']
    params = bound_params_from_dict(fitted_bound_param)
    print('-' * 79)
    print('* FITTED BOUND PARAMETERS')
    params.pretty_print()
    pause_debugplot(args.debugplot)

    # read FITS image with odd-numbered slitlets
    hdulist_odd = fits.open(args.fitsfile_odd)
    image2d_odd = hdulist_odd[0].data
    hdulist_odd.close()

    # read FITS image with even-numbered slitlets
    hdulist_even = fits.open(args.fitsfile_even)
    image2d_even = hdulist_even[0].data
    hdulist_even.close()

    # determine parameters according to grism+filter combination
    grism_name = fitted_bound_param['tags']['grism']
    filter_name = fitted_bound_param['tags']['filter']
    crpix1_enlarged = 1.0  # center of first pixel
    if grism_name == "J" and filter_name == "J":
        crval1_enlarged = 11000.000  # Angstroms
        cdelt1_enlarged = 0.7575  # Angstroms/pixel
        naxis1_enlarged = 4134  # pixels
        nbrightlines = [18]
    elif grism_name == "H" and filter_name == "H":
        crval1_enlarged = 14000.000  # Angstroms
        cdelt1_enlarged = 1.2000  # Angstroms/pixel
        naxis1_enlarged = 4134  # pixels
        nbrightlines = [0]
    elif grism_name == "K" and filter_name == "Ksp":
        crval1_enlarged = 19000.000  # Angstroms
        cdelt1_enlarged = 1.7000  # Angstroms/pixel
        naxis1_enlarged = 4134  # pixels
        nbrightlines = [0]
    elif grism_name == "LR" and filter_name == "YJ":
        crval1_enlarged = None  # Angstroms
        cdelt1_enlarged = None  # Angstroms/pixel
        naxis1_enlarged = None  # pixels
        nbrightlines = None
    elif grism_name == "LR" and filter_name == "HK":
        crval1_enlarged = None  # Angstroms
        cdelt1_enlarged = None  # Angstroms/pixel
        naxis1_enlarged = None  # pixels
        nbrightlines = None
    else:
        raise ValueError("invalid grism_name and/or filter_name")
    crmin1_enlarged = \
        crval1_enlarged + \
        (1.0 - crpix1_enlarged) * \
        cdelt1_enlarged  # Angstroms
    crmax1_enlarged = \
        crval1_enlarged + \
        (naxis1_enlarged - crpix1_enlarged) * \
        cdelt1_enlarged  # Angstroms

    # read master arc line wavelengths (only brightest lines)
    wv_master = read_wv_master_file(
        wv_master_file=args.wv_master_file,
        lines='brightest',
        debugplot=args.debugplot
    )
    # clip master arc line list to expected wavelength range
    lok1 = crmin1_enlarged <= wv_master
    lok2 = wv_master <= crmax1_enlarged
    lok = lok1 * lok2
    wv_master = wv_master[lok]
    if abs(args.debugplot) >= 10:
        print("clipped wv_master:\n", wv_master)

    # read master arc line wavelengths (whole data set)
    wv_master_all = read_wv_master_file(
        wv_master_file=args.wv_master_file,
        lines='all',
        debugplot=args.debugplot
    )
    # clip master arc line list to expected wavelength range
    lok1 = crmin1_enlarged <= wv_master_all
    lok2 = wv_master_all <= crmax1_enlarged
    lok = lok1 * lok2
    wv_master_all = wv_master_all[lok]

    # ------------------------------------------------------------------------

    measured_slitlets = []

    for islitlet in list_slitlets:

        # define Slitlet2D object
        slt = Slitlet2D(islitlet=islitlet,
                        params=params, parmodel=parmodel,
                        csu_conf=csu_conf,
                        debugplot=args.debugplot)

        # extract 2D image corresponding to the selected slitlet
        if islitlet % 2 == 0:
            slitlet2d = slt.extract_slitlet2d(image2d_even)
        else:
            slitlet2d = slt.extract_slitlet2d(image2d_odd)

        # subtract smooth background computed as follows:
        # - median collapsed spectrum of the whole slitlet2d
        # - independent median filtering of the previous spectrum in the
        #   two halves in the spectral direction
        spmedian = np.median(slitlet2d, axis=0)
        sp1 = medfilt(spmedian[:int(NAXIS1_EMIR/2)], 201)
        sp2 = medfilt(spmedian[int(NAXIS1_EMIR/2):], 201)
        spbackground = np.concatenate((sp1, sp2))
        slitlet2d -= spbackground

        # locate unknown arc lines
        slt.locate_unknown_arc_lines(slitlet2d=slitlet2d)

        # continue working with current slitlet only if arc lines have
        # been detected
        if slt.list_arc_lines is not None:

            # compute intersections between spectrum trails and arc lines
            slt.xy_spectrail_arc_intersections(slitlet2d=slitlet2d)

            # compute rectification transformation
            slt.estimate_tt_to_rectify(order=args.order_fmap,
                                       slitlet2d=slitlet2d)

            # rectify image
            slitlet2d_rect = slt.rectify(slitlet2d,
                                         resampling=1,
                                         transformation=1)

            # median spectrum and line peaks from rectified image
            sp_median, fxpeaks = slt.median_spectrum_from_rectified_image(
                slitlet2d_rect,
                sigma_gaussian_filtering=2,
                nwinwidth_initial=5,
                nwinwidth_refined=5,
                times_sigma_threshold=5,
                npix_avoid_border=6,
                nbrightlines=nbrightlines
            )

            # perform initial wavelength calibration
            solution_wv = wvcal_spectrum(
                sp=sp_median,
                fxpeaks=fxpeaks,
                poly_degree_wfit=args.poldeg_initial,
                wv_master=wv_master,
                debugplot=slt.debugplot
            )
            # store initial wavelength calibration polynomial in current
            # slitlet instance
            slt.wpoly_initial = np.polynomial.Polynomial(solution_wv.coeff)
            pause_debugplot(args.debugplot)

            # refine wavelength calibration
            poly_refined, npoints_eff, residual_std = refine_arccalibration(
                sp=sp_median,
                poly_initial=slt.wpoly_initial,
                wv_master=wv_master_all,
                poldeg=args.poldeg_refined,
                npix=1,
                debugplot=slt.debugplot
            )
            # store refined wavelength calibration polynomial in current
            # slitlet instance
            slt.wpoly_refined = poly_refined

        # store current slitlet in list of measured slitlets
        measured_slitlets.append(slt)

        if args.debugplot == 0:
            if islitlet % 10 == 0:
                cout = str(islitlet // 10)
            else:
                cout = '.'
            sys.stdout.write(cout)
            # print(slt)
            if islitlet == list_slitlets[-1]:
                sys.stdout.write('\n')
            sys.stdout.flush()
        else:
            pause_debugplot(args.debugplot)

    # ------------------------------------------------------------------------

    # polynomial coefficients corresponding to the wavelength calibration
    # step 1: compute variation of each coefficient as a function of
    # y0_reference_middle of each slitlet
    list_poly = []
    for i in range(args.poldeg_refined + 1):
        xp = []
        yp = []
        for slt in measured_slitlets:
            if slt.wpoly_refined is not None:
                xp.append(slt.y0_reference_middle)
                yp.append(slt.wpoly_refined.coef[i])
        poly, yres, reject = polfit_residuals_with_sigma_rejection(
            x=np.array(xp),
            y=np.array(yp),
            deg=2,
            times_sigma_reject=5,
            xlabel='y0_rectified',
            ylabel='coeff[' + str(i) + ']',
            debugplot=args.debugplot
        )
        list_poly.append(poly)
    # step 2: use the variation of each polynomial coefficient with
    # y0_reference_middle to infer the expected wavelength calibration
    # polynomial for each rectifified slitlet
    for slt in measured_slitlets:
        y0_reference_middle = slt.y0_reference_middle
        list_new_coeff = []
        for i in range(args.poldeg_refined + 1):
            new_coeff = list_poly[i](y0_reference_middle)
            list_new_coeff.append(new_coeff)
        slt.wpoly_refined_smoothed = np.polynomial.Polynomial(list_new_coeff)

    # ------------------------------------------------------------------------

    # rectification transformation coefficients ttd_aij and ttd_bij
    # step 1: compute variation of each coefficient as a function of
    # y0_reference_middle of each slitlet
    list_poly_aij = []
    list_poly_bij = []
    ncoef_ttd = ncoef_fmap(args.order_fmap)
    for i in range(ncoef_ttd):
        xp = []
        yp_aij = []
        yp_bij = []
        for slt in measured_slitlets:
            if slt.ttd_aij is not None:
                xp.append(slt.y0_reference_middle)
                yp_aij.append(slt.ttd_aij[i])
                yp_bij.append(slt.ttd_bij[i])
        poly, yres, reject = polfit_residuals_with_sigma_rejection(
            x=np.array(xp),
            y=np.array(yp_aij),
            deg=5,
            times_sigma_reject=5,
            xlabel='y0_rectified',
            ylabel='ttd_aij[' + str(i) + ']',
            debugplot=args.debugplot
        )
        list_poly_aij.append(poly)
        poly, yres, reject = polfit_residuals_with_sigma_rejection(
            x=np.array(xp),
            y=np.array(yp_bij),
            deg=5,
            times_sigma_reject=5,
            xlabel='y0_rectified',
            ylabel='ttd_bij[' + str(i) + ']',
            debugplot=args.debugplot
        )
        list_poly_bij.append(poly)
    # step 2: use the variation of each coefficient with y0_reference_middle
    # to infer the expected rectification transformation for each slitlet
    for slt in measured_slitlets:
        slt.ttd_order_smoothed = args.order_fmap
        y0_reference_middle = slt.y0_reference_middle
        slt.ttd_aij_smoothed = []
        slt.ttd_bij_smoothed = []
        for i in range(ncoef_ttd):
            new_coeff_aij = list_poly_aij[i](y0_reference_middle)
            slt.ttd_aij_smoothed.append(new_coeff_aij)
            new_coeff_bij = list_poly_bij[i](y0_reference_middle)
            slt.ttd_bij_smoothed.append(new_coeff_bij)

    # ------------------------------------------------------------------------

    image2d_rectified = np.zeros((NAXIS2_EMIR, NAXIS1_EMIR))

    for slt in measured_slitlets:

        islitlet = slt.islitlet

        if islitlet % 10 == 0:
            cout = str(islitlet // 10)
        else:
            cout = '.'
        sys.stdout.write(cout)
        if islitlet == list_slitlets[-1]:
            sys.stdout.write('\n')
        sys.stdout.flush()

        nscan_min = int(slt.y0_frontier_lower + 0.5)
        nscan_max = int(slt.y0_frontier_upper + 0.5)

        # slt.debugplot = 12

        # extract 2D image corresponding to the selected slitlet
        if islitlet % 2 == 0:
            slitlet2d = slt.extract_slitlet2d(image2d_even)
        else:
            slitlet2d = slt.extract_slitlet2d(image2d_odd)

        # rectify image
        if slt.ttd_order is not None:
            transformation = 1
        elif slt.ttd_order_smoothed is not None:
            transformation = 2
        else:
            raise ValueError("No ttd transformation defined!")

        slitlet2d_rect = slt.rectify(slitlet2d,
                                     resampling=1,
                                     transformation=transformation)

        ii1 = nscan_min - slt.bb_ns1_orig
        ii2 = nscan_max - slt.bb_ns1_orig + 1

        j1 = slt.bb_nc1_orig - 1
        j2 = slt.bb_nc2_orig
        i1 = slt.bb_ns1_orig - 1 + ii1
        i2 = i1 + ii2 - ii1

        # print('nscan_min, nscan_max:', nscan_min, nscan_max)
        # print('ii1, ii2:', ii1, ii2)
        # print('i1, i2:', i1, i2)
        # print('j1, j2:', j1, j2)

        image2d_rectified[i1:i2, j1:j2] = slitlet2d_rect[ii1:ii2, :]

    if abs(args.debugplot) % 10 != 0:
        ximshow(image2d_rectified, debugplot=12)

    save_ndarray_to_fits(image2d_rectified, args.out_rect)

    # ------------------------------------------------------------------------
    # TODO:
    # (1) rectify each slitlet and generate a rectified and wavelength
    #     calibrated image with all the slitlets
    # (2) save relevant results in JSON file:
    #     - wavelength calibration polynomial for each rectified slitlet
    #       (initial, refined, and refined_smoothed)

    # Note that the following code does not have sense since now the program
    # reads two FITS files with odd- and even-numbered slitlets. In fact,
    # args.fitsfile does not exist.
    """
    if abs(args.debugplot) % 10 != 0:
        ax = ximshow_file(args.fitsfile.name, show=False)
        for slt in measured_slitlets:
            xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
            ylower = \
                slt.list_spectrails[slt.i_lower_spectrail].poly_funct(xdum)
            ax.plot(xdum, ylower, 'b-')
            yupper = \
                slt.list_spectrails[slt.i_upper_spectrail].poly_funct(xdum)
            ax.plot(xdum, yupper, 'b-')
            if slt.list_arc_lines is not None:
                for arcline in slt.list_arc_lines:
                    xdum, ydum = arcline.linspace_pix(
                        start=arcline.ylower_line,
                        stop=arcline.yupper_line
                    )
                    ax.plot(xdum, ydum, 'g-')
        pause_debugplot(args.debugplot, pltshow=True)
    """


if __name__ == "__main__":
    main()
