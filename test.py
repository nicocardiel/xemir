from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import numpy as np
import sys

from matplotlib.patches import Rectangle
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
from ccd_line import SpectrumTrail
from ccd_line import ArcLine

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


class Slitlet2D(object):
    """Slitlet2D class definition.

    """

    def __init__(self, islitlet, params, parmodel, csu_conf, ymargin=10,
                 debugplot=0):

        self.islitlet = islitlet

        self.bb_nc1_orig = 100
        self.bb_nc2_orig = NAXIS1_EMIR - 100

        self.poly_lower_expected, self.poly_upper_expected = \
            expected_distorted_boundaries(
                islitlet, csu_conf.csu_bar_slit_center[islitlet - 1],
                'both', params, parmodel,
                numpts=101, deg=5, debugplot=0
            )
        xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
        ylower = self.poly_lower_expected(xdum)
        yupper = self.poly_upper_expected(xdum)
        self.bb_ns1_orig = int(ylower.min() + 0.5) - ymargin
        if self.bb_ns1_orig < 1:
            self.bb_ns1_orig = 1
        self.bb_ns2_orig = int(yupper.max() + 0.5) + ymargin
        if self.bb_ns2_orig > NAXIS2_EMIR:
            self.bb_ns2_orig = NAXIS2_EMIR

        self.list_arc_lines = None

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
            "- bb_nc1_orig.................: " + \
                 str(self.bb_nc1_orig) + "\n" + \
            "- bb_nc2_orig.................: " + \
                 str(self.bb_nc2_orig) + "\n" + \
            "- bb_ns1_orig.................: " + \
                 str(self.bb_ns1_orig) + "\n" + \
            "- bb_ns2_orig.................: " \
                 + str(self.bb_ns2_orig) + "\n" + \
            "- num. of associated arc lines: " + \
               str(number_arc_lines) + "\n" + \
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

        # return slitlet image
        return slitlet2d

    def locate_unknown_arc_lines(self, slitlet2d,
                                 times_sigma_threshold=4,
                                 minimum_threshold=None,
                                 delta_x_max=30,
                                 delta_y_min=37):
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

        """

        # smooth denoising of slitlet2d
        slitlet2d_rs, coef_rs = rescale_array_to_z1z2(slitlet2d, z1z2=(-1, 1))
        slitlet2d_dn = restoration.denoise_nl_means(slitlet2d_rs,
                                                    patch_size=3,
                                                    patch_distance=2)
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
        if abs(self.debugplot) % 10 != 0:
            # display initial image with zscale cuts
            title = "[slit #" + str(self.islitlet) + "]" + \
                                    " (locate_unknown_arc_lines #1)"
            ximshow(slitlet2d, title=title,
                    debugplot=self.debugplot)
            # display denoised image with zscale cuts
            title = "[slit #" + str(self.islitlet) + "]" + \
                    " (locate_unknown_arc_lines #2)"
            ximshow(slitlet2d_dn, title=title,
                    debugplot=self.debugplot)
            # display image with different cuts
            z1z2 = (q50 + times_sigma_threshold * sigmag,
                    q50 + 2 * times_sigma_threshold * sigmag)
            title = "[slit #" + str(self.islitlet) + "]" + \
                    " (locate_unknown_arc_lines #3)"
            ximshow(slitlet2d_dn, title=title, z1z2=z1z2,
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
        if abs(self.debugplot) % 10 != 0:
            # display all objects identified in the image
            title = "[slit #" + str(self.islitlet) + "]" + \
                    " (locate_unknown_arc_lines #4)"
            z1z2 = (labels2d_objects.min(), labels2d_objects.max())
            ximshow(labels2d_objects, title=title,
                    cbar_label="Object number",
                    z1z2=z1z2, cmap="nipy_spectral",
                    debugplot=self.debugplot)

        # select arc lines by imposing the criteria based on the
        # dimensions of the detected objects
        slices_possible_arc_lines = ndimage.find_objects(labels2d_objects)
        slices_ok = np.repeat([False], no_objects)  # flag
        for i in range(no_objects):
            if abs(self.debugplot) >= 10:
                print('object:', i + 1, slices_possible_arc_lines[i])
            slice_x = slices_possible_arc_lines[i][1]
            slice_y = slices_possible_arc_lines[i][0]
            # note that the width computation doesn't require to
            # add +1 since slice_x.stop (and slice_y.stop) is
            # already the upper limit +1 (in np.array coordinates)
            delta_x = slice_x.stop - slice_x.start
            delta_y = slice_y.stop - slice_y.start
            if delta_x <= delta_x_max and delta_y >= delta_y_min:
                slices_ok[i] = True

        # generate list with ID of arc lines (note that first object is
        # number 0 and not 1)
        list_slices_ok = []
        for i in range(no_objects):
            if slices_ok[i]:
                list_slices_ok.append(i + 1)
        number_arc_lines = len(list_slices_ok)
        if abs(self.debugplot) >= 10:
            print("\nNumber of arc lines finally identified is:",
                  number_arc_lines)
            print("Slice ID of lines passing the selection:\n",
                  list_slices_ok)
        if number_arc_lines == 0:
            raise ValueError("Number of arc lines identified is 0")

        # display arc lines
        if abs(self.debugplot) % 10 != 0:
            # display all objects identified in the image
            title = "[slit #" + str(self.islitlet) + "]" + \
                    " (locate_unknown_arc_lines #5)"
            z1z2 = (labels2d_objects.min(),
                    labels2d_objects.max())
            ax = ximshow(labels2d_objects, show=False, title=title,
                         cbar_label="Object number",
                         z1z2=z1z2, cmap="nipy_spectral",
                         debugplot=self.debugplot)
            # plot rectangle around identified arc lines
            for i in range(no_objects):
                if slices_ok[i]:
                    slice_x = slices_possible_arc_lines[i][1]
                    slice_y = slices_possible_arc_lines[i][0]
                    # note that slice_x and slice_y are given in np.array
                    # coordinates; for that reason +1 transform them into
                    # image coordinates; in addition, -0.5 shift the origin
                    # to the lower left corner of the pixel
                    xini_slice = slice_x.start + 1 - 0.5
                    yini_slice = slice_y.start + 1 - 0.5
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

        # generate mask with all the arc-line points passing the selection
        mask_arc_lines = np.zeros_like(slitlet2d_dn)
        for k in list_slices_ok:
            mask_arc_lines[labels2d_objects == k] = 1

        # adjust individual arc lines passing the selection
        self.list_arc_lines = []  # list of ArcLines
        for k in range(number_arc_lines):  # fit each arc line
            # select points to be fitted for a particular arc line
            xy_tmp = np.where(labels2d_objects == list_slices_ok[k])
            x_tmp = xy_tmp[1] + 1  # np.array coordinates --> image coordinates
            y_tmp = xy_tmp[0] + 1  # np.array coordinates --> image coordinates
            w_tmp = slitlet2d_dn[xy_tmp]
            # declare new ArcLine instance
            arc_line = ArcLine()
            # define new ArcLine using a weighted fit
            # (note that it must be X vs Y)
            arc_line.fit(x=x_tmp, y=y_tmp, deg=1, w=w_tmp, y_vs_x=False)
            # update list with identified ArcLines
            self.list_arc_lines.append(arc_line)

        if abs(self.debugplot) >= 10:
            # print list of arc lines
            print('\nlist_arc_lines:')
            for k in range(number_arc_lines):
                print(k, '->', self.list_arc_lines[k], '\n')

        # display results
        if abs(self.debugplot) % 10 != 0:
            # compute image with only the arc lines passing the selection
            labels2d_arc_lines = labels2d_objects * mask_arc_lines
            # display background image with filtered arc lines
            title = "[slit #" + str(self.islitlet) + "]" + \
                    " (locate_unknown_arc_lines #6)"
            z1z2 = (labels2d_arc_lines.min(),
                    labels2d_arc_lines.max())
            ax = ximshow(labels2d_arc_lines, show=False,
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
    for islitlet in range(islitlet_min, islitlet_max + 1, 2):
        slt = Slitlet2D(islitlet=islitlet,
                        params=params, parmodel=parmodel,
                        csu_conf=csu_conf,
                        debugplot=args.debugplot)
        slitlet2d = slt.extract_slitlet2d(image2d)
        #
        ax = ximshow(slitlet2d, title="Slitlet#" + str(islitlet), show=False)
        xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
        ylower = slt.poly_lower_expected(xdum) - slt.bb_ns1_orig + 1
        ax.plot(xdum, ylower, 'b-')
        yupper = slt.poly_upper_expected(xdum) - slt.bb_ns1_orig + 1
        ax.plot(xdum, yupper, 'b-')
        pause_debugplot(debugplot=args.debugplot, pltshow=True)
        #
        slt.locate_unknown_arc_lines(slitlet2d)
        print(slt)

    if False:
        ax=ximshow_file(args.fitsfile.name, show=False)
        for islitlet in range(islitlet_min, islitlet_max + 1):
            poly_lower_expected, poly_upper_expected = \
                expected_distorted_boundaries(
                    islitlet, csu_conf.csu_bar_slit_center[islitlet - 1],
                    'both', params, parmodel,
                    numpts=101, deg=5, debugplot=0
                )
            xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
            ylower = poly_lower_expected(xdum)
            ax.plot(xdum, ylower, 'b-')
            yupper = poly_upper_expected(xdum)
            ax.plot(xdum, yupper, 'b-')
        pause_debugplot(args.debugplot, pltshow=True)


if __name__ == "__main__":
    main()
