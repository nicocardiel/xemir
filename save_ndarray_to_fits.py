from __future__ import division
from __future__ import print_function

import numpy as np
from astropy.io import fits


def save_ndarray_to_fits(array=None, file_name=None,
                         main_header=None,
                         crpix1=None, crval1=None, cdelt1=None,
                         clobber=True):
    """Save numpy array into a FITS file with the provided filename.

    Parameters
    ----------
    array : numpy array, floats
        Array to be exported as the FITS file.
    file_name : string
        File name for the FITS file.
    main_header : astropy FITS header
        Header to be introduced in the output file.
    crpix1 : float
        If not None, this value is used for the keyword CRPIX1.
    crval1 : float
        If not None, this value is used for the keyword CRVAL1.
    cdelt1 : float
        If not None, this value is used for the keyword CDELT1.
    clobber : bool
        If True, the file is overwritten (in the case it already
        exists).

    """

    # protections
    if type(array) is not np.ndarray:
        raise ValueError("array=" + str(array) +
                         " must be a numpy.ndarray")
    if file_name is None:
        raise ValueError("file_name is not defined in save_ndarray_to_fits")

    # write to FITS file (casting to np.float)
    if main_header is None:
        hdu = fits.PrimaryHDU(array.astype(np.float))
    else:
        hdu = fits.PrimaryHDU(array.astype(np.float), main_header)

    # set additional FITS keywords if requested
    if crpix1 is not None:
        hdu.header.set('CRPIX1', crpix1,
                       'Reference pixel')
    if crval1 is not None:
        hdu.header.set('CRVAL1', crval1,
                       'Reference wavelength corresponding to CRPIX1 ')
    if cdelt1 is not None:
        hdu.header.set('CDELT1', cdelt1,
                       'Linear dispersion (angstrom/pixel) ')

    # write output file
    hdu.writeto(file_name, overwrite=clobber)
