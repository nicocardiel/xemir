from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import numpy as np

from rsix.subsets_of_fileinfo_from_txt import subsets_of_fileinfo_from_txt


def compute_abba_result(list_of_fileinfo, outfile):
    """Compute A-B-(B-A).

    Parameters
    ----------
    list_of_4_files : list of strings
        List of the for files corresponding to the ABBA observation.
    outfile : string
        Base name for the output FITS file name

    """

    # check number of images
    if len(list_of_fileinfo) != 4:
        raise ValueError("Unexpected number of ABBA files: " +
                         str(len(list_of_fileinfo)))

    # avoid PyCharm warnings
    # (local variable might be referenced before assignment)
    naxis1 = 0
    naxis2 = 0
    image_header = None

    # check image dimensions
    for i in range(4):
        hdulist = fits.open(list_of_fileinfo[i].filename)
        if i == 0:
            image_header = hdulist[0].header
            naxis1 = image_header['naxis1']
            naxis2 = image_header['naxis2']
            hdulist.close()
        else:
            image_header_ = hdulist[0].header
            naxis1_ = image_header_['naxis1']
            naxis2_ = image_header_['naxis2']
            hdulist.close()
            if naxis1 != naxis1_ or naxis2 != naxis2_:
                print('>>> naxis1, naxis2..:', naxis1, naxis2)
                print('>>> naxis1_, naxis2_:', naxis1_, naxis2_)
                raise ValueError("Image dimensions do not agree!")

    # read the four images
    factor_ini = np.array([1.0, -1.0, 0.0, 0.0])
    factor_end = np.array([0.0, 0.0, -1.0, 1.0])
    factor = np.array([1.0, -1.0, -1.0, 1.0])
    result_ini = np.zeros((naxis2, naxis1))
    result_end = np.zeros((naxis2, naxis1))
    result = np.zeros((naxis2, naxis1))
    for i in range(4):
        hdulist = fits.open(list_of_fileinfo[i].filename)
        image2d = hdulist[0].data.astype(np.float)
        hdulist.close()
        result_ini += factor_ini[i] * image2d
        result_end += factor_end[i] * image2d
        result += factor[i] * image2d

    # save results, including partial subtractions
    hdu = fits.PrimaryHDU(result_ini.astype(np.float), image_header)
    hdu.writeto(outfile + '_sub1.fits', clobber=True)
    hdu = fits.PrimaryHDU(result_end.astype(np.float), image_header)
    hdu.writeto(outfile + '_sub2.fits', clobber=True)
    hdu = fits.PrimaryHDU(result.astype(np.float), image_header)
    hdu.writeto(outfile + '.fits', clobber=True)


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='compute_ABBA')
    parser.add_argument("txt_file",
                        help="txt file with list of ABBA subsets")
    args = parser.parse_args(args)

    # execute function
    dict_of_subsets = subsets_of_fileinfo_from_txt(args.txt_file)
    for idict in range(len(dict_of_subsets)):
        tmpdict = dict_of_subsets[idict]
        tmplist = tmpdict['list_of_fileinfo']
        print('\n>>> Label: ', tmpdict['label'])
        for item in tmplist:
            print(item)

        compute_abba_result(tmplist, tmpdict['label'])


if __name__ == "__main__":

    main()
