from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import numpy as np
import sys

from numina.array.display.ximshow import ximshow
from numina.array.display.ximshow import ximshow_file
from numina.array.display.pause_debugplot import pause_debugplot

from csu_configuration import CsuConfiguration
from dtu_configuration import DtuConfiguration
from emir_definitions import NAXIS1_EMIR
from emir_definitions import NAXIS2_EMIR
from fit_boundaries import bound_params_from_dict
from fit_boundaries import expected_distorted_boundaries
from spec_2d_image import Spec2DImage

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES

class Slitlet2D(object):
    """Slitlet2D class definition.

    """

    def __init__(self, islitlet, params, parmodel, csu_conf):
        self.bb_nc1_orig = 1
        self.bb_nc2_orig = NAXIS1_EMIR

        self.poly_lower_expected, self.poly_upper_expected = \
            expected_distorted_boundaries(
                islitlet, csu_conf.csu_bar_slit_center[islitlet - 1],
                'both', params, parmodel,
                numpts=101, deg=5, debugplot=0
            )
        ymargin = 5
        xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
        ylower = self.poly_lower_expected(xdum)
        yupper = self.poly_upper_expected(xdum)
        self.bb_ns1_orig = int(ylower.min() + 0.5) - ymargin
        if self.bb_ns1_orig < 1:
            self.bb_ns1_orig = 1
        self.bb_ns2_orig = int(yupper.max() + 0.5) + ymargin
        if self.bb_ns2_orig > NAXIS2_EMIR:
            self.bb_ns2_orig = NAXIS2_EMIR


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
    for islitlet in range(islitlet_min, islitlet_max + 1):
        slt = Slitlet2D(islitlet, params, parmodel, csu_conf)
        slitlet2d = image2d[(slt.bb_ns1_orig - 1):slt.bb_ns2_orig,
                            (slt.bb_nc1_orig - 1):slt.bb_nc2_orig]
        ax = ximshow(slitlet2d, title="Slitlet#" + str(islitlet), show=False)
        xdum = np.linspace(1, NAXIS1_EMIR, num=NAXIS1_EMIR)
        ylower = slt.poly_lower_expected(xdum) - slt.bb_ns1_orig + 1
        ax.plot(xdum, ylower, 'b-')
        yupper = slt.poly_upper_expected(xdum) - slt.bb_ns1_orig + 1
        ax.plot(xdum, yupper, 'b-')
        pause_debugplot(debugplot=args.debugplot, pltshow=True)

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
