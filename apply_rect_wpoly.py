from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import numpy as np
import sys

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximshow import ximshow
from numina.array.distortion import order_fmap

from rect_wpoly_for_mos import islitlet_progress
from set_wv_enlarged_parameters import set_wv_enlarged_parameters

from emir_definitions import NAXIS1_EMIR
from emir_definitions import NAXIS2_EMIR

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


class Slitlet2D(object):
    """Slitlet2D class definition.

    Parameters
    ----------
    islitlet : int
        Slitlet number.
    megadict : dictionary
        Python dictionary storing the JSON input file where the
        the rectification and wavelength calibration transformations
        for a particular instrument configuration are stored.
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
        as the Y coordinate of the middle spectrum trail. The same value
        is used for all the available spectrum trails.
    list_spectrails: list of numpy.polynomial.Polynomial instances
        List of spectrum trails defined (lower, middle and upper).
    list_frontiers: list of numpy.polynomial.Polynomial instances
        List of spectrum trails defining the slitlet frontiers (lower
        and upper).
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
    ttd_order : int or None
        Polynomial order corresponding to the rectification
        transformation.
    ttd_aij : numpy array
        Polynomial coefficents corresponding to the direct
        rectification transformation coefficients a_ij.
    ttd_bij : numpy array
        Polynomial coefficents corresponding to the direct
        rectification transformation coefficients b_ij.
    tti_aij : numpy array
        Polynomial coefficents corresponding to the inverse
        rectification transformation coefficients a_ij.
    tti_bij : numpy array
        Polynomial coefficents corresponding to the inverse
        rectification transformation coefficients b_ij.
    wpoly : Polynomial instance
        Wavelength calibration polynomial, providing the
        wavelength as a function of pixel number (running from 1 to
        NAXIS1).
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.


    """

    def __init__(self, islitlet, megadict, ymargin, debugplot):
        # slitlet number
        self.islitlet = islitlet
        cslitlet = 'slitlet' + str(islitlet).zfill(2)

        # csu configuration
        tmpcsu = megadict['csu_configuration'][cslitlet]
        self.csu_bar_left = tmpcsu['csu_bar_left']
        self.csu_bar_right = tmpcsu['csu_bar_right']
        self.csu_bar_slit_center = tmpcsu['csu_bar_slit_center']
        self.csu_bar_slit_width = tmpcsu['csu_bar_slit_width']

        # horizontal bounding box
        self.bb_nc1_orig = 1
        self.bb_nc2_orig = NAXIS1_EMIR

        # reference abscissa
        self.x0_reference = float(NAXIS1_EMIR) / 2.0 + 0.5  # single float

        # list of spectrum trails (lower, middle, and upper)
        tmpcontent = megadict['contents'][cslitlet]
        self.list_spectrails = []
        for idum, cdum in zip(range(3), ['lower', 'middle', 'upper']):
            coeff = tmpcontent['spectrail_' + cdum]
            self.list_spectrails.append(np.polynomial.Polynomial(coeff))

        # define reference ordinates using lower, middle and upper spectrails
        # evaluated at x0_reference
        self.y0_reference_lower = self.list_spectrails[0](self.x0_reference)
        self.y0_reference_middle = self.list_spectrails[1](self.x0_reference)
        self.y0_reference_upper = self.list_spectrails[2](self.x0_reference)

        # list of frontiers (lower and upper)
        self.list_frontiers = []
        for idum, cdum in zip(range(2), ['lower', 'upper']):
            coeff = tmpcontent['frontier_' + cdum]
            self.list_frontiers.append(np.polynomial.Polynomial(coeff))

        # define frontier ordinates at x0_reference
        self.y0_frontier_lower = self.list_frontiers[0](self.x0_reference)
        self.y0_frontier_upper = self.list_frontiers[1](self.x0_reference)

        # determine vertical bounding box
        xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
        ylower = self.list_frontiers[0](xdum)
        yupper = self.list_frontiers[1](xdum)
        self.bb_ns1_orig = int(ylower.min() + 0.5) - ymargin
        if self.bb_ns1_orig < 1:
            self.bb_ns1_orig = 1
        self.bb_ns2_orig = int(yupper.max() + 0.5) + ymargin
        if self.bb_ns2_orig > NAXIS2_EMIR:
            self.bb_ns2_orig = NAXIS2_EMIR

        # Rectification coefficients
        self.ttd_aij = tmpcontent['ttd_aij']
        self.ttd_bij = tmpcontent['ttd_bij']
        self.tti_aij = tmpcontent['tti_aij']
        self.tti_bij = tmpcontent['tti_bij']
        # determine order from number of coefficients
        ncoef = len(self.ttd_aij)
        self.ttd_order = order_fmap(ncoef)

        # Wavelength calibration coefficients
        self.wpoly = tmpcontent['wpoly_coeff']

        # debugplot
        self.debugplot = debugplot

    def __repr__(self):
        """Define printable representation of a Slitlet2D instance."""

        # string with all the information
        output = "<Slilet2D instance>\n" + \
            "- islitlet...........: " + \
                 str(self.islitlet) + "\n" + \
            "- csu_bar_left.......: " + \
                 str(self.csu_bar_left) + "\n" + \
            "- csu_bar_right......: " + \
                 str(self.csu_bar_right) + "\n" + \
            "- csu_bar_slit_center: " + \
                 str(self.csu_bar_slit_center) + "\n" + \
            "- csu_bar_slit_width.: " + \
                 str(self.csu_bar_slit_width) + "\n" + \
            "- x0_reference.......: " + \
                 str(self.x0_reference) + "\n" + \
            "- y0_reference_lower.: " + \
                str(self.y0_reference_lower) + "\n" + \
            "- y0_reference_middle: " + \
                 str(self.y0_reference_middle) + "\n" + \
            "- y0_reference_upper.: " + \
                 str(self.y0_reference_upper) + "\n" + \
            "- y0_frontier_lower..: " + \
                 str(self.y0_frontier_lower) + "\n" + \
            "- y0_frontier_upper..: " + \
                 str(self.y0_frontier_upper) + "\n" + \
            "- bb_nc1_orig........: " + \
                 str(self.bb_nc1_orig) + "\n" + \
            "- bb_nc2_orig........: " + \
                 str(self.bb_nc2_orig) + "\n" + \
            "- bb_ns1_orig........: " + \
                 str(self.bb_ns1_orig) + "\n" + \
            "- bb_ns2_orig........: " + \
                 str(self.bb_ns2_orig) + "\n" + \
            "- lower spectrail....:\n\t" + \
                 str(self.list_spectrails[0]) + "\n" + \
            "- middle spectrail...:\n\t" + \
                 str(self.list_spectrails[1]) + "\n" + \
            "- upper spectrail....:\n\t" + \
                 str(self.list_spectrails[2]) + "\n" + \
            "- lower frontier.....:\n\t" + \
                str(self.list_frontiers[0]) + "\n" + \
            "- upper frontier.....:\n\t" + \
                str(self.list_frontiers[1]) + "\n" + \
            "- ttd_order..........: " + str(self.ttd_order) + "\n" + \
            "- ttd_aij............:\n\t" + str(self.ttd_aij) + "\n" + \
            "- ttd_bij............:\n\t" + str(self.ttd_bij) + "\n" + \
            "- tti_aij............:\n\t" + str(self.tti_aij) + "\n" + \
            "- tti_bij............:\n\t" + str(self.tti_bij) + "\n" + \
            "- wpoly..............:\n\t" + str(self.wpoly) + "\n" + \
            "- debugplot...................: " + \
            str(self.debugplot)

        return output

    def extract_slitlet2d(self, image_2k2k):
        """Extract slitlet 2d image from image with original EMIR dimensions.

        Parameters
        ----------
        image_2k2k : 2d numpy array, float
            Original image (dimensions NAXIS1_EMIR * NAXIS2_EMIR)

        Returns
        -------
        slitlet2d : 2d numpy array, float
            Image corresponding to the slitlet region defined by its
            bounding box.

        """

        # protections
        naxis2, naxis1 = image_2k2k.shape
        if naxis1 != NAXIS1_EMIR:
            raise ValueError('Unexpected naxis1')
        if naxis2 != NAXIS2_EMIR:
            raise ValueError('Unexpected naxis2')

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
                self.list_spectrails[0](xdum)
            ax.plot(xdum, ylower, 'b-')
            ymiddle = \
                self.list_spectrails[1](xdum)
            ax.plot(xdum, ymiddle, 'b--')
            yupper = \
                self.list_spectrails[2](xdum)
            ax.plot(xdum, yupper, 'b-')
            ylower_frontier = self.list_frontiers[0](xdum)
            ax.plot(xdum, ylower_frontier, 'b:')
            yupper_frontier = self.list_frontiers[1](xdum)
            ax.plot(xdum, yupper_frontier, 'b:')
            pause_debugplot(debugplot=self.debugplot, pltshow=True)

        # return slitlet image
        return slitlet2d


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='evaluate_rect_wpoly')
    # required arguments
    parser.add_argument("fitsfile",
                        help="Input FITS file",
                        type=argparse.FileType('r'))
    parser.add_argument("--rect_wpoly", required=True,
                        help="Input JSON file with rectification and "
                             "wavelength calibration coefficients",
                        type=argparse.FileType('r'))
    parser.add_argument("--outfile",
                        help="Output FITS file with rectified and "
                             "wavelength calibrated image",
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

    # read calibration structure from JSON file
    rect_wpoly_dict = json.loads(open(args.rect_wpoly.name).read())

    # read FITS image and its corresponding header
    hdulist = fits.open(args.fitsfile)
    header = hdulist[0].header
    image2d = hdulist[0].data
    hdulist.close()

    # protections
    naxis2, naxis1 = image2d.shape
    if naxis1 != header['naxis1'] or naxis2 != header['naxis2']:
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)
        raise ValueError('Something is wrong with NAXIS1 and/or NAXIS2')
    if abs(args.debugplot) >= 10:
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)

    # check that the input FITS file grism and filter match
    filter_name = header['filter']
    if filter_name != rect_wpoly_dict['tags']['filter']:
        raise ValueError("Filter name does not match!")
    grism_name = header['grism']
    if grism_name != rect_wpoly_dict['tags']['grism']:
        raise ValueError("Filter name does not match!")
    if abs(args.debugplot) >= 10:
        print('>>> grism.......:', grism_name)
        print('>>> filter......:', filter_name)

    # read islitlet_min and islitlet_max from input JSON file
    islitlet_min = rect_wpoly_dict['tags']['islitlet_min']
    islitlet_max = rect_wpoly_dict['tags']['islitlet_max']
    if abs(args.debugplot) >= 10:
        print('>>> islitlet_min:', islitlet_min)
        print('>>> islitlet_max:', islitlet_max)

    # ---

    # relevant wavelength calibration parameters for rectified image
    crpix1_enlarged, crval1_enlarged, cdelt1_enlarged, naxis1_enlarged = \
        set_wv_enlarged_parameters(filter_name, grism_name)

    # initialize rectified image
    image2d_rectified_wv = np.zeros((NAXIS2_EMIR, naxis1_enlarged))

    # main loop
    for islitlet in range(islitlet_min, islitlet_max + 1):
        if args.debugplot == 0:
            islitlet_progress(islitlet, islitlet_max)

        # define Slitlet2D object
        slt = Slitlet2D(islitlet=islitlet,
                        megadict=rect_wpoly_dict,
                        ymargin=2,
                        debugplot=args.debugplot)

        if abs(args.debugplot) >= 10:
            print(slt)
            slitlet2d = slt.extract_slitlet2d(image2d)

        pause_debugplot(args.debugplot)



if __name__ == "__main__":
    main()