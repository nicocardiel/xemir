from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import os.path
import sys

from numina.array.display.ximshow import ximshow

from display_slitlet_arrangement import read_csup_from_header
from emir_slitlet import EmirSlitlet
from save_ndarray_to_fits import save_ndarray_to_fits
from subtract_background_median import subtract_background_median

from emir_definitions import VALID_GRISMS
from emir_definitions import VALID_FILTERS
from emir_definitions import NAXIS1_EMIR


def update_megadict(filename, grism, spfilter, list_slitlets,
                    nwidthx_smooth,
                    times_sigma_threshold,
                    minimum_threshold_factor,
                    poly_degree_wfit,
                    megadict,
                    out_1dspec,
                    debugplot=0):
    """Update megadict using arc lines in filename.

    Parameters
    ----------
    filename : string
        Input arc line image.
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.
    list_slitlets : list of integers
        Number of slitlets to be updated.
    nwidthx_smooth : int
        Window in X axis to compute a smoothed version of the image
        (using 1d median filtering) that is subtracted prior to arc
        line peak detection. If this parameter is zero, no background
        image is subtracted.
    times_sigma_threshold : float
        Times (robust) sigma above the median of the spectrum to set
        the minimum threshold when searching for line peaks.
    minimum_threshold_factor : float or None
        If not None, the maximum of the median spectrum is divided by
        this factor to determine an additional threshold to search for
        arc line peaks.
    poly_degree_wfit : int
        Degree for wavelength calibration polynomial.
    megadict : dictionary of dictionaries
        Structure to store the wavelength calibration mapping.
    out_1dspec : string
        Root file name to generate output fits file with 1d spectra.
    debugplot : int
        Determines whether intermediate computations and/or plots are
        displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    """

    # read 2D image
    hdulist = fits.open(filename)
    image_header = hdulist[0].header
    image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape
    hdulist.close()
    if debugplot >= 10:
        print('>>> Reading file:', filename)
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)

    # check that the FITS file has been obtained with EMIR
    instrument = image_header['instrume']
    if instrument != 'EMIR':
        raise ValueError("INSTRUME keyword is not 'EMIR'!")

    # read CSU configuration for FITS header
    csu_bar_left, csu_bar_right, csu_bar_slit_center, csu_bar_slit_width = \
        read_csup_from_header(image_header=image_header, debugplot=0)

    # read grism
    grism_in_header = image_header['grism']
    if grism != grism_in_header:
        raise ValueError("GRISM keyword=" + grism_in_header +
                         " is not the expected value=" + grism)
    # read filter
    spfilter_in_header = image_header['filter']
    if spfilter != spfilter_in_header:
        raise ValueError("FILTER keyword=" + spfilter_in_header +
                         " is not the expected value=" + spfilter)
    # read rotator position angle
    rotang = image_header['rotang']

    # read date-obs
    date_obs = image_header['date-obs']

    for slitlet_number in list_slitlets:
        if debugplot < 10:
            sys.stdout.write('.')
            sys.stdout.flush()
        # declare slitlet instance
        slt = EmirSlitlet(
            grism_name=grism,
            filter_name=spfilter,
            rotang=rotang,
            slitlet_number=slitlet_number,
            fits_file_name=filename,
            date_obs=date_obs,
            csu_bar_left=csu_bar_left[slitlet_number-1],
            csu_bar_right=csu_bar_right[slitlet_number-1],
            csu_bar_slit_center=csu_bar_slit_center[slitlet_number-1],
            csu_bar_slit_width=csu_bar_slit_width[slitlet_number-1],
            debugplot=debugplot)

        # extract slitlet2d
        slitlet2d = slt.extract_slitlet2d(image_2k2k=image2d)

        # subtract background computed from 1d median filtering
        if nwidthx_smooth > 0:
            title = slt.fits_file_name + \
                    " [slit #" + str(slt.slitlet_number) + "]" + \
                    " grism=" + slt.grism_name + \
                    ", filter=" + slt.filter_name
            # split the original image in two halves in order to
            # avoid the transition between channels 1024 and 1025
            # that many times exhibit a break in the signal
            slitlet2d_left = slitlet2d[:, 0:(NAXIS1_EMIR/2)]
            slitlet2d_left = subtract_background_median(
                slitlet2d_left, size_x=nwidthx_smooth, size_y=1,
                title=title, debugplot=debugplot)
            slitlet2d_right = slitlet2d[:, (NAXIS1_EMIR/2):]
            slitlet2d_right = subtract_background_median(
                slitlet2d_right, size_x=nwidthx_smooth, size_y=1,
                title=title, debugplot=debugplot)
            # merge the two halves
            slitlet2d[:, 0:(NAXIS1_EMIR/2)] = slitlet2d_left
            slitlet2d[:, (NAXIS1_EMIR/2):] = slitlet2d_right
            if debugplot % 10 != 0:
                ximshow(slitlet2d,
                        title=title + "\n(after background subtraction)",
                        debugplot=debugplot)

        # define middle, upper and lower spectrum trails from
        # slitlet boundaries
        slt.define_spectrails_from_boundaries(slitlet2d)

        # extract median spectrum around the distorted middle
        # spectrum trail, and identify location of arc lines peaks
        sp_median, fxpeaks, sxpeaks = \
            slt.median_spectrum_around_middle_spectrail(
                slitlet2d,
                below_middle_spectrail=5,
                above_middle_spectrail=5,
                sigma_gaussian_filtering=2,
                nwinwidth_initial=7,
                nwinwidth_refined=5,
                times_sigma_threshold=times_sigma_threshold,
                minimum_threshold_factor=minimum_threshold_factor,
                npix_avoid_border=6
            )

        if out_1dspec is not None:
            save_ndarray_to_fits(sp_median,
                                 out_1dspec + str(slitlet_number).zfill(2) +
                                 '.fits')

        # wavelength calibration of arc lines found in median spectrum
        slt.wavelength_calibration(
            fxpeaks=fxpeaks,
            lamp="argon_xenon_EMIR",
            crpix1=1.0,
            error_xpos_arc=0.3,
            times_sigma_r=3.0,
            frac_triplets_for_sum=0.50,
            times_sigma_theil_sen=100.0,  # 10.0,
            poly_degree_wfit=poly_degree_wfit,
            times_sigma_polfilt=10.0,
            times_sigma_cook=10.0,
            times_sigma_inclusion=10.0,
            weighted=False,
            plot_residuals=False
        )

        if debugplot % 10 != 0:
            slt.overplot_wavelength_calibration(sp_median, fxpeaks)

        # important: employ the identified arc lines to update
        # slt.list_arc_lines
        slt.locate_known_arc_lines(slitlet2d,
                                   below_middle_spectrail=15,
                                   above_middle_spectrail=15,
                                   nwidth=7, nsearch=2)

        # update megadict
        tmp_dict = slt.dict_arc_lines_slitlet()
        main_label = "megadict_grism_" + grism + "_" + \
                     "filter_" + spfilter
        if main_label not in megadict:
            megadict[main_label] = {}
        slitlet_label = "slitlet" + str(slitlet_number).zfill(2)
        if slitlet_label not in megadict[main_label]:
            megadict[main_label][slitlet_label] = {}
        megadict[main_label][slitlet_label][slt.date_obs] = tmp_dict

        # TODO: revise the code below
        """
        # compute intersections of arc lines with all the spectrum
        # trails
        slt.xy_spectrail_arc_intersections(slitlet2d=slitlet2d)

        # use previous intersections to compute the transformation
        # to rectify the slitlet image
        slt.estimate_tt_to_rectify(order=2, slitlet2d=slitlet2d)

        # rectify the slitlet image
        slitlet2d_rect = slt.rectify(slitlet2d=slitlet2d, order=1)

        # pasos adicionales a seguir:
        # - extract mediam spectrum from full rectified slitlet
        # - iterar repetiendo desde calibracion en l.d.o.???
        """


if __name__ == "__main__":

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        help="FITS file or txt file with list of FITS files")
    parser.add_argument("grism",
                        help="Grism name ('J', 'H', 'K', 'LR')")
    parser.add_argument("filter",
                        help="Filter name ('J', 'H', 'Ksp',...)")
    parser.add_argument("tuple_slit_numbers",
                        help="Tuple n1[,n2[,step]] to define slitlet numbers")
    parser.add_argument("--nwidthx_smooth",
                        help="""
                        Window in X axis to compute a smoothed version of the
                        image (using 1d median filtering) that is subtracted
                        prior to arc line peak detection. If this parameter is
                        zero, no background image is subtracted.
                        (default=0)""",
                        default=0)
    parser.add_argument("--times_sigma_threshold",
                        help="""Times (robust) sigma above the median of the
                        spectrum to set the minimum threshold when searching
                        for line peaks.
                        (default=3.0)""",
                        default="3.0")
    parser.add_argument("--minimum_threshold_factor",
                        help="""
                        If not None, the maximum of the median spectrum is
                        divided by this factor to determine an additional
                        threshold to search for arc line peaks.
                        (default=None)""",
                        default=None)
    parser.add_argument("--poly_degree_wfit",
                        help="""
                        Degree for wavelength calibration polynomial.
                        (default=3)""",
                        default="3")
    parser.add_argument("--first_time",
                        help="If 'yes' a new megadict is generated" +
                        " (default=no)",
                        default="no")
    parser.add_argument("--out_1dspec",
                        help="Root file name for 1d spectrum output" +
                        " (default=None)",
                        default=None)
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=0)",
                        default=0)
    args = parser.parse_args()

    # read grism
    grism = args.grism
    if grism not in VALID_GRISMS:
        raise ValueError("Grism=" + grism + " is not in valid grism list")

    # read filter name
    spfilter = args.filter
    if spfilter not in VALID_FILTERS:
        raise ValueError("Filter=" + spfilter + " is not in valid filter list")

    # read slitlet numbers to be computed
    tmp_str = args.tuple_slit_numbers.split(",")
    if len(tmp_str) == 3:
        list_slitlets = range(int(tmp_str[0]),
                              int(tmp_str[1])+1,
                              int(tmp_str[2]))
    elif len(tmp_str) == 2:
        list_slitlets = range(int(tmp_str[0]),
                              int(tmp_str[1])+1,
                              1)
    elif len(tmp_str) == 1:
        list_slitlets = [int(tmp_str[0])]
    else:
        raise ValueError("Invalid tuple for slitlet numbers")

    # read nwidthx_smooth
    nwidthx_smooth = int(args.nwidthx_smooth)

    # read times_sigma_threshold
    times_sigma_threshold = float(args.times_sigma_threshold)

    # read minimum_threshold_factor
    if args.minimum_threshold_factor is None:
        minimum_threshold_factor = None
    else:
        minimum_threshold_factor = float(args.minimum_threshold_factor)

    # read poly_degree_wfit
    poly_degree_wfit = int(args.poly_degree_wfit)

    # read first_time
    first_time = (args.first_time == "yes")

    # read out_1dspec value
    out_1dspec = args.out_1dspec

    # read debugplot value
    debugplot = int(args.debugplot)

    # check for input file
    filename = args.filename
    if not os.path.isfile(filename):
        raise ValueError("File " + filename + " not found!")

    # define megadict prior to new computation
    if first_time:
        megadict = {}
        print('\n>>> Generating new megadict from scratch')
    else:
        megadict_file = "megadict_grism_" + grism + \
                        "_filter_" + spfilter + ".json"
        if not os.path.isfile(megadict_file):
            raise ValueError("File " + megadict_file + " not found!")
        else:
            megadict = json.loads(open(megadict_file).read())
            print('\n>>> Initializing megadict from previous file:')
            print(megadict_file)

    # if input file is a txt file, assume it is a list of FITS files
    if filename[-4:] == ".txt":
        with open(filename) as f:
            file_content = f.read().splitlines()
        list_fits_files = []
        for line in file_content:
            if len(line) > 0:
                if line[0] != '#':
                    tmpfile = line.split()[0]
                    if not os.path.isfile(tmpfile):
                        raise ValueError("File " + tmpfile + " not found!")
                    list_fits_files.append(tmpfile)
    else:
        list_fits_files = [filename]

    # update megadict
    for ifile, myfile in enumerate(list_fits_files):
        print('>>> Reading file ' + str(ifile + 1) + "/" +
              str(len(list_fits_files)) + ":\n" + myfile)
        update_megadict(filename=myfile,
                        list_slitlets=list_slitlets,
                        grism=grism, spfilter=spfilter,
                        nwidthx_smooth=nwidthx_smooth,  # 401 (J,H)
                        times_sigma_threshold=times_sigma_threshold,  # 3 (J,H)
                        minimum_threshold_factor=minimum_threshold_factor, # 200,  # 100 (J), 400 (H)
                        poly_degree_wfit=poly_degree_wfit,
                        megadict=megadict,
                        out_1dspec=out_1dspec,
                        debugplot=debugplot)

    # save new version of megadict
    # print(json.dumps(megadict, indent=4, sort_keys=True))
    outfile = "megadict_grism_" + grism + "_filter_" + spfilter + ".json"
    with open(outfile, 'w') as fstream:
        json.dump(megadict, fstream, indent=4, sort_keys=True)
        print('>>> Saving file ' + outfile)
