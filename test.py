from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import numpy as np
import sys

from matplotlib.patches import Rectangle
from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection
from numina.array.display.ximshow import ximshow
from numina.array.display.ximshow import ximshow_file
from numina.array.display.pause_debugplot import pause_debugplot
from scipy import ndimage
from skimage import restoration

from csu_configuration import CsuConfiguration
from dtu_configuration import DtuConfiguration
from emir_definitions import NAXIS1_EMIR
from emir_definitions import NAXIS2_EMIR
from fit_boundaries import bound_params_from_dict
from fit_boundaries import expected_distorted_boundaries
from rescale_array_to_z1z2 import rescale_array_to_z1z2
from rescale_array_to_z1z2 import rescale_array_from_z1z2
from ccd_line import ArcLine
from ccd_line import intersection_spectrail_arcline

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


class Slitlet2D(object):
    """Slitlet2D class definition.

    Parameters
    ==========
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
    ==========
    islitlet : int
        Slitlet number.
    csu_conf : CsuConfiguration object
        Instance of CsuConfiguration.
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
        X coordinate where the rectified y0_reference is computed
        as the Y coordinate of the spectrum trails. The same value
        is used for all the available spectrum trails.
    list_spectrails: list of SpectrumTrail instances
        List of spectrum trails defined.
    list_arclines : list of ArcLine instances
        List with identified arc lines.
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
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    """

    def __init__(self, islitlet, params, parmodel, csu_conf, ymargin=10,
                 debugplot=0):

        # slitlet number
        self.islitlet = islitlet
        self.csu_conf = csu_conf.csu_bar_slit_center

        # horizontal bounding box
        self.bb_nc1_orig = 1
        self.bb_nc2_orig = NAXIS1_EMIR

        # reference abscissa
        self.x0_reference = float(NAXIS1_EMIR) / 2.0 + 0.5  # single float

        # compute spectrum trails:
        # 0 : lower boundary
        # 1 : middle spectrum
        # 2 : upper boundary
        self.list_spectrails = expected_distorted_boundaries(
                islitlet, csu_conf.csu_bar_slit_center[islitlet - 1],
                [0, 0.5, 1], params, parmodel,
                numpts=101, deg=5, debugplot=0
            )
        # update y_rectified computed at x0_reference
        for spectrail in self.list_spectrails:
            spectrail.y_rectified = spectrail.poly_funct(self.x0_reference)

        # determine vertical bounding box
        xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
        ylower = self.list_spectrails[0].poly_funct(xdum)
        yupper = self.list_spectrails[2].poly_funct(xdum)
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

        # debugplot
        self.debugplot = debugplot

    def __repr__(self):
        """Define printable representation of a Slitlet2D instance."""

        # list of associated arc lines
        if self.list_arc_lines is None:
            number_arc_lines = None
        else:
            number_arc_lines = len(self.list_arc_lines)

        # string with all the information
        output = "<Slilet2D instance>\n" + \
            "- islitlet....................: " + \
                 str(self.islitlet) + "\n" + \
            "- csu_conf....................: " + \
                 str(self.csu_conf) + "\n" + \
            "- x0_reference................: " + \
                 str(self.x0_reference) + "\n" + \
            "- bb_nc1_orig.................: " + \
                 str(self.bb_nc1_orig) + "\n" + \
            "- bb_nc2_orig.................: " + \
                 str(self.bb_nc2_orig) + "\n" + \
            "- bb_ns1_orig.................: " + \
                 str(self.bb_ns1_orig) + "\n" + \
            "- bb_ns2_orig.................: " + \
                 str(self.bb_ns2_orig) + "\n" + \
            "- spectrail[0].poly_funct.....:\n" + \
                 str(self.list_spectrails[0].poly_funct) + "\n" + \
            "- spectrail[1].poly_funct.....:\n" + \
                 str(self.list_spectrails[1].poly_funct) + "\n" + \
            "- spectrail[2].poly_funct.....:\n" + \
                 str(self.list_spectrails[2].poly_funct) + "\n" + \
            "- num. of associated arc lines: " + \
               str(number_arc_lines) + "\n" + \
            "- x_inter_orig................:\n" + \
                 str(self.x_inter_orig) + "\n" + \
            "- y_inter_orig................:\n" + \
                 str(self.y_inter_orig) + "\n" + \
            "- x_inter_rect................:\n" + \
                 str(self.x_inter_rect) + "\n" + \
            "- y_inter_rect................:\n" + \
                 str(self.y_inter_rect) + "\n" + \
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
            ylower = self.list_spectrails[0].poly_funct(xdum)
            ax.plot(xdum, ylower, 'b-')
            ymiddle = self.list_spectrails[1].poly_funct(xdum)
            ax.plot(xdum, ymiddle, 'b--')
            yupper = self.list_spectrails[2].poly_funct(xdum)
            ax.plot(xdum, yupper, 'b-')
            pause_debugplot(debugplot=self.debugplot, pltshow=True)

        # return slitlet image
        return slitlet2d

    def locate_unknown_arc_lines(self, slitlet2d,
                                 times_sigma_threshold=4,
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
                ymiddle_slice = \
                    self.list_spectrails[1].poly_funct(xmiddle_slice)
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
                         for k in range(number_arc_lines) ])
        xfit = np.zeros(number_arc_lines)
        # intersection between middle spectrum trail and arc line
        for k in range(number_arc_lines):
            arcline = self.list_arc_lines[k]
            xfit[k], ydum = intersection_spectrail_arcline(
                self.list_spectrails[1], arcline
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
            optiona.

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
            spectrail = self.list_spectrails[1]  # middle spectrum trail
            xroot, yroot = intersection_spectrail_arcline(
                spectrail=spectrail, arcline=arcline
            )
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
                arcline.x_rectified = xroot
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


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='display_slitlet_arrangement')
    parser.add_argument("fitsfile",
                        help="FITS file",
                        type=argparse.FileType('r'))
    parser.add_argument("fitted_bound_param",
                        help="Input JSON with fitted boundary parameters",
                        type=argparse.FileType('r'))
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting & debugging options"
                             " (default=12)",
                        default=12, type=int,
                        choices=DEBUGPLOT_CODES)
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")
    args = parser.parse_args(args)

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    csu_conf = CsuConfiguration()
    csu_conf.define_from_fits(args.fitsfile)
    print(csu_conf)
    raw_input("Pause...")
    dtu_conf = DtuConfiguration()
    dtu_conf.define_from_fits(args.fitsfile)
    print(dtu_conf)
    raw_input("Pause...")

    fitted_bound_param = json.loads(open(args.fitted_bound_param.name).read())
    parmodel = fitted_bound_param['meta-info']['parmodel']
    params = bound_params_from_dict(fitted_bound_param)
    print('-' * 79)
    print('* FITTED BOUND PARAMETERS')
    params.pretty_print()
    raw_input("Pause...")

    # read FITS image
    hdulist = fits.open(args.fitsfile)
    image2d = hdulist[0].data
    hdulist.close()

    islitlet_min = fitted_bound_param['tags']['islitlet_min']
    islitlet_max = fitted_bound_param['tags']['islitlet_max']
    list_islitlets = []

    for islitlet in range(islitlet_min, islitlet_max + 1):

        # define Slitlet2D object
        slt = Slitlet2D(islitlet=islitlet,
                        params=params, parmodel=parmodel,
                        csu_conf=csu_conf,
                        debugplot=args.debugplot)

        # extract 2D image corresponding to the selected slitlet
        slitlet2d = slt.extract_slitlet2d(image2d)

        # locate unknown arc lines
        slt.locate_unknown_arc_lines(slitlet2d=slitlet2d)

        # compute intersections between spectrum trails and arc lines
        slt.debugplot = 12
        slt.xy_spectrail_arc_intersections(slitlet2d=slitlet2d)
        #...
        #...
        #
        if args.debugplot == 0:
            sys.stdout.write('.')
            if islitlet == islitlet_max:
                sys.stdout.write('\n')
            sys.stdout.flush()
        else:
            pause_debugplot(args.debugplot)
        list_islitlets.append(slt)

    if abs(args.debugplot) % 10 != 0:
        ax=ximshow_file(args.fitsfile.name, show=False)
        for slt in list_islitlets:
            xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
            ylower = slt.list_spectrails[0].poly_funct(xdum)
            ax.plot(xdum, ylower, 'b-')
            yupper = slt.list_spectrails[2].poly_funct(xdum)
            ax.plot(xdum, yupper, 'b-')
            if slt.list_arc_lines is not None:
                for arcline in slt.list_arc_lines:
                    xdum, ydum = arcline.linspace_pix(
                        start=arcline.ylower_line,
                        stop=arcline.yupper_line
                    )
                    ax.plot(xdum, ydum, 'g-')
        # pause_debugplot(args.debugplot, pltshow=True)
        pause_debugplot(12, pltshow=True)
        raw_input("Pause...")


if __name__ == "__main__":
    main()
