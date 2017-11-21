from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import numpy as np

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximshow import ximshow


TOLERANCE = -6  # power of ten

if __name__ == "__main__":

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list",
                        help="txt file with list of initial images")
    parser.add_argument("output_fits_filename",
                        help="filename of output enlarged FITS image")
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                        " (default=0)",
                        default=0)
    args = parser.parse_args()

    debugplot = int(args.debugplot)

    table = np.genfromtxt(args.input_list, dtype=[('filename', '|S100')])
    list_of_files = table['filename']
    number_of_files = len(list_of_files)

    list_y_boundaries = []

    # declare auxiliary arrays to store image basic parameters
    naxis1 = np.zeros(number_of_files, dtype=np.int)
    naxis2 = np.zeros(number_of_files, dtype=np.int)
    crpix1 = np.zeros(number_of_files)
    crval1 = np.zeros(number_of_files)
    cdelt1 = np.zeros(number_of_files)
    wv_min = np.zeros(number_of_files)
    wv_max = np.zeros(number_of_files)

    # read basic parameters for all the images
    for i in range(number_of_files):
        infile = list_of_files[i]
        hdulist = fits.open(infile)
        image_header = hdulist[0].header
        hdulist.close()
        naxis1[i] = image_header['naxis1']
        naxis2[i] = image_header['naxis2']
        crpix1[i] = image_header['crpix1']
        crval1[i] = image_header['crval1']
        cdelt1[i] = image_header['cdelt1']
        wv_min[i] = crval1[i] + (1 - crpix1[i]) * cdelt1[i]
        wv_max[i] = crval1[i] + (naxis1[i] - crpix1[i]) * cdelt1[i]
        if debugplot >= 10:
            print("\n>>> Image " + str(i + 1) + " of " + str(number_of_files))
            print('>>> file..:', infile)
            print('>>> NAXIS1:', naxis1[i])
            print('>>> NAXIS2:', naxis2[i])
            print('>>> CRPIX1:', crpix1[i])
            print('>>> CRVAL1:', crval1[i])
            print('>>> CDELT1:', cdelt1[i])
            print('>>> wv_min:', wv_min[i])
            print('>>> wv_max:', wv_max[i])

    # check that CDELT1 is the same for all the images
    cdelt1_enlarged = cdelt1[0]
    cdelt1_comp = np.repeat(cdelt1_enlarged, number_of_files)
    if not np.allclose(cdelt1, cdelt1_comp):
        raise ValueError("CDELT1 values are different")

    # check that CRPIX1 differences are integer values
    for i in range(number_of_files):
        delta = crpix1[i] - crpix1[0]
        if abs(delta - round(delta)) > 10 ** TOLERANCE:
            raise ValueError("CRPIX1 differences are not integer values")

    # compute minimum and maximum wavelengths for the final enlarged
    # image
    wv_min_enlarged = wv_min.min()
    wv_max_enlarged = wv_max.max()
    crpix1_enlarged = 1.0
    crval1_enlarged = wv_min_enlarged
    if debugplot >= 0:
        print("\n>>> Final wavelength coverage:",
              wv_min_enlarged, wv_max_enlarged)

    # determine dimensions of the final enlarged image
    fnaxis1 = (wv_max_enlarged - wv_min_enlarged) / cdelt1_enlarged + 1.0
    if abs(fnaxis1 - round(fnaxis1)) > 10 ** TOLERANCE:
        raise ValueError("Final NAXIS1 value is not an integer!")
    naxis1_enlarged = int(round(fnaxis1))
    naxis2_enlarged = naxis2.sum()
    if debugplot >= 0:
        print(">>> Final NAXIS1, NAXIS2.....:",
              naxis1_enlarged, naxis2_enlarged)

    # define array to store final enlarged image
    image2d_enlarged = np.zeros((naxis2_enlarged, naxis1_enlarged))

    # insert individual images in the final enlarged image
    for i in range(number_of_files):
        infile = list_of_files[i]
        hdulist = fits.open(infile)
        image2d = hdulist[0].data
        hdulist.close()

        # location of individual image within the final enlarged image
        # > indices i1, i2: image rows
        # > indices j1, j2: image columns
        if i == 0:
            i1 = 0
        else:
            i1 = sum(naxis2[:i])
        i2 = i1 + naxis2[i]

        fj1 = (crval1[i] - crval1_enlarged) / cdelt1_enlarged
        if abs(fj1 - round(fj1)) > 10 ** TOLERANCE:
            raise ValueError("Unexpected j1 value")
        j1 = int(round(fj1))
        j2 = j1 + naxis1[i]

        image2d_enlarged[i1:i2, j1:j2] = image2d

        # store boundary to separate subimages in final plot
        list_y_boundaries.append(i2 + 0.5)

    # save final enlarged image
    hdu = fits.PrimaryHDU(image2d_enlarged)
    hdu.header.set('CRPIX1', crpix1_enlarged, 'Reference pixel')
    hdu.header.set('CRVAL1', crval1_enlarged, 'Reference pixel')
    hdu.header.set('CDELT1', cdelt1_enlarged, 'Reference pixel')
    hdu.writeto(args.output_fits_filename, clobber=True)
    if debugplot >= 10:
        print("\n>>> Generating output file:", args.output_fits_filename)

    # if requested, display image
    if debugplot % 10 != 0:
        ax = ximshow(image2d_enlarged,
                     cbar_label="Number of counts\n\n" +
                                args.output_fits_filename,
                     image_bbox=(1, naxis1_enlarged, 1, naxis2_enlarged),
                     show=False)
        ax.grid(False)
        # display additional wavelength scale
        xmin, xmax = ax.get_xlim()
        ax2 = ax.twiny()
        ax2.grid(False)
        ax2.set_xlim([crval1_enlarged + (xmin-1)*cdelt1_enlarged,
                      crval1_enlarged + (xmax-1)*cdelt1_enlarged])
        ax2.set_xlabel('wavelength (Angstrom)')
        # display additional scale indicating number of image
        ax3 = ax.twinx()
        ax3.set_ylim([0.5, number_of_files + 0.5])
        ax3.set_ylabel("image number (different pos_pix values)")
        ax3.grid(False)
        # display separation between subimages
        if False:
            for y_boundary in list_y_boundaries:
                ax.axhline(y_boundary, color="grey")
        # show final plot
        pause_debugplot(debugplot, pltshow=True)
