from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits

from numina.array.display.list_fits_files_from_txt \
    import list_fits_files_from_txt

from emirdrp.core import EMIR_PIXSCALE
from emirdrp.processing.datamodel import get_dtur_from_header


def get_dtu_displacement_2hdr(hdr_reference, hdr_comparison,
                              file_comparison=None, debugplot=0):
    """Return the image shift of one image relative to a reference one.

    Parameters
    ----------
    hdr_reference : FITS header
        Header of the FITS file to be used as reference.
    hdr_comparison : string
        Header of the FITS file to be compared with the reference file.
    file_comparison : string or None
        Name of the FITS file to be compared with the reference file.
    debugplot : int
        Determines whether intermediate computations and/or plots are
        displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    Returns
    -------
    vec_reference : list of two floats
        X and Y corresponding to the DTU position for reference file.
    vec_comparison : list of two floats
        X and Y corresponding to the DTU position for the comparison
        file.
    delta_vec : list of two floats
        Difference in X and Y between de DTU location of the comparison
        and the reference files.

    """

    dtub_reference, dtur_reference = get_dtur_from_header(hdr_reference)
    xfac_reference = dtur_reference[0] / EMIR_PIXSCALE
    yfac_reference = -dtur_reference[1] / EMIR_PIXSCALE
    vec_reference = [yfac_reference, xfac_reference]

    dtub_comparison, dtur_comparison = get_dtur_from_header(hdr_comparison)
    xfac_comparison = dtur_comparison[0] / EMIR_PIXSCALE
    yfac_comparison = -dtur_comparison[1] / EMIR_PIXSCALE
    vec_comparison = [yfac_comparison, xfac_comparison]

    delta_vec = [yfac_comparison - yfac_reference,
                 xfac_comparison - xfac_reference]

    if debugplot >= 10:
        vec_ref_txt = '{0:8.3f}  {1:8.3f}'.format(
            vec_reference[0], vec_reference[1])
        vec_txt = '{0:8.3f}  {1:8.3f}'.format(
            vec_comparison[0], vec_comparison[1])
        delta_vec_txt = '{0:8.3f}  {1:8.3f}'.format(delta_vec[0], delta_vec[1])
        if file_comparison is None:
            infotxt = ""
        else:
            infotxt = " --> " + file_comparison
        print(vec_ref_txt + "  |" + vec_txt + "  |" + delta_vec_txt + infotxt)

    return vec_reference, vec_comparison, delta_vec


def get_dtu_displacement_2images(file_reference, file_comparison, debugplot=0):
    """Return the image shift of one image relative to a reference one.

    Parameters
    ----------
    file_reference : string
        Name of the FITS file to be used as reference.
    file_comparison : string
        Name of the FITS file to be compared with the reference file.
    debugplot : int
        Determines whether intermediate computations and/or plots are
        displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    Returns
    -------
    vec_ref : list of two floats
        X and Y corresponding to the DTU position for reference file.
    vec : list of two floats
        X and Y corresponding to the DTU position for the comparison
        file.
    delta_vec : list of two floats
        Difference in X and Y between de DTU location of the comparison
        and the reference files.

    """

    hdulist = fits.open(file_reference)
    hdr_refeference = hdulist[0].header
    hdulist.close()

    hdulist = fits.open(file_comparison)
    hdr_comparison = hdulist[0].header
    hdulist.close()

    vec_reference, vec_comparison, delta_vec = get_dtu_displacement_2hdr(
        hdr_refeference, hdr_comparison, file_comparison, debugplot
    )

    return vec_reference, vec_comparison, delta_vec


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='offsets_dtu')
    parser.add_argument("reference",
                        help="reference FITS file")
    parser.add_argument("filename",
                        help="FITS file or txt file with list of FITS files")
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=10)",
                        default=10)
    args = parser.parse_args(args)

    if args.debugplot is None:
        debugplot = 0
    else:
        debugplot = int(args.debugplot)

    reffile = args.reference
    list_fits_files = list_fits_files_from_txt(args.filename)

    if debugplot >= 10:
        print('============================================================')
        print('   DTU reference    |   DTU comparison   |     Delta DTU    ')
        print('....................|....................|..................')
        print('     X         Y    |     X         Y    |     X         Y  ')
        print('------------------------------------------------------------')

    for myfile in list_fits_files:
        get_dtu_displacement_2images(file_reference=reffile,
                                     file_comparison=myfile,
                                     debugplot=debugplot)
    if debugplot >= 10:
        print('============================================================')


if __name__ == "__main__":

    main()
