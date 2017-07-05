from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits

from emir_slitlet import EmirSlitlet


def first_look_arc(arc_filename, datadir, grism, slitlet_number, debugplot):
    """Execute first steps to calibrate arc.

    Parameters
    ----------
    arc_filename : string
        File name of FITS file containing the arc image.
    datadir : string
        Directory were data are stored.
    slitlet_number : int
        Number of slitlet.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    """

    infile = datadir + arc_filename
    hdulist = fits.open(infile)
    image_header = hdulist[0].header
    image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape
    hdulist.close()
    if debugplot >= 10:
        print('>>> Reading file:', arc_filename)
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)

    # extract slitlet subimage
    slt = EmirSlitlet(grism_name=grism,
                      slitlet_number=int(slitlet_number),
                      fits_file_name=arc_filename, debugplot=debugplot)
    slitlet2d = slt.extract_slitlet2d(image_2k2k=image2d)

    # locate unknown arc lines and determine middle spectrum trail
    slt.locate_unknown_arc_lines(slitlet2d=slitlet2d)

    # define vertical offsets to compute additional spectrum
    # trails by shifting the middle one
    v_offsets = [17, 8.5, -7, -14]
    slt.additional_spectrail_from_middle_spectrail(v_offsets)

    # two iterations:
    # - iteration 0: use unknown arc lines
    # - iteration 1: use known arc lines (from wavelength
    #                calibration derived in iteration 0
    for iteration in range(2):
        # compute intersections of arc lines with all the spectrum
        # trails
        slt.xy_spectrail_arc_intersections(slitlet2d=slitlet2d)

        # use previous intersections to compute the transformation
        # to rectify the slitlet image
        slt.estimate_tt_to_rectify(order=3, slitlet2d=slitlet2d)

        # rectify the slitlet image
        slitlet2d_rect = slt.rectify(slitlet2d=slitlet2d, order=1)

        # extract median spectrum and identify location of arc lines
        # peaks
        sp_median, fxpeaks, sxpeaks = \
            slt.median_spectrum_from_rectified_image(
                slitlet2d_rect,
                below_middle_spectrail=15,
                above_middle_spectrail=15,
                nwinwidth_initial=7,
                nwinwidth_refined=5,
                times_sigma_threshold=10,
                npix_avoid_border=6
            )

        # wavelength calibration of arc lines found in median spectrum
        slt.wavelength_calibration(
            fxpeaks=fxpeaks,
            lamp="argon_xenon",
            crpix1=1.0,
            error_xpos_arc=0.3,
            times_sigma_r=3.0,
            frac_triplets_for_sum=0.50,
            times_sigma_theil_sen=10.0,
            poly_degree_wfit=3,
            times_sigma_polfilt=10.0,
            times_sigma_inclusion=10.0,
            weighted=False,
            plot_residuals=True
        )

        # overplot wavelength calibration in median spectrum
        if iteration == 1:
            slt.debugplot = 12  # TBR
            slt.overplot_wavelength_calibration(sp_median)
            slt.debugplot = 0  # TBR

        # important: employ the identified arc lines to update
        # slt.list_arc_lines (do not forget this step; otherwise
        # the slt.list_arc_lines will erroneously correspond to
        # the previous arc line list define prior to the last
        # wavelength calibration)
        slt.locate_known_arc_lines(slitlet2d,
                                   below_middle_spectrail=15,
                                   above_middle_spectrail=15,
                                   nwidth=7, nsearch=2)

        print("\n>>> iteration:", iteration)
        print(slt)


if __name__ == "__main__":

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("arc_filename",
                        help="FITS image containing the arc image")
    parser.add_argument("grism",
                        help="Grism name (J, H or K)")
    parser.add_argument("slitlet_number",
                        help="Number of slitlet")
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                        " (default=0)",
                        default=0)
    parser.add_argument("--datadir",
                        help="Directory where data are stored" +
                             " (default = ./)",
                        default="./")
    args = parser.parse_args()

    debugplot = int(args.debugplot)

    first_look_arc(args.arc_filename, args.datadir, args.grism,
                   int(args.slitlet_number), debugplot)

    if debugplot != 2 and debugplot != 12:
        try:
            input("\nPress RETURN to QUIT...")
        except SyntaxError:
            pass
