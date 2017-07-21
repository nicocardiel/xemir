from __future__ import division
from __future__ import print_function

from astropy.io import fits


class DtuConfiguration:
    """DTU Configuration class definition.

    Attributes
    ----------
    xdtu : float
        XDTU fits keyword value.
    ydtu : float
        YDTU fits keyword value.
    zdtu : float
        ZDTU fits keyword value.
    xdtu_0 : float
        XDTU_0 fits keyword value.
    ydtu_0 : float
        YDTU_0 fits keyword value.
    zdtu_0 : float
        ZDTU_0 fits keyword value.
    defined : bool
        Indicates whether the DTU parameters have been properly defined.

    """

    def __init__(self):
        self.xdtu = None
        self.ydtu = None
        self.zdtu = None
        self.xdtu_0 = None
        self.ydtu_0 = None
        self.zdtu_0 = None
        self.defined = False

    def __str__(self):
        output = "<DtuConfiguration instance>\n"
        if self.defined:
            strdum = "- XDTU..: {0:8.3f}\n".format(self.xdtu)
            output += strdum
            strdum = "- YDTU..: {0:8.3f}\n".format(self.ydtu)
            output += strdum
            strdum = "- ZDTU..: {0:8.3f}\n".format(self.zdtu)
            output += strdum
            strdum = "- XDTU_0: {0:8.3f}\n".format(self.xdtu_0)
            output += strdum
            strdum = "- YDTU_0: {0:8.3f}\n".format(self.ydtu_0)
            output += strdum
            strdum = "- ZDTU_0: {0:8.3f}\n".format(self.zdtu_0)
            output += strdum
        else:
            output += "- XDTU..:  None\n"
            output += "- YDTU..:  None\n"
            output += "- ZDTU..:  None\n"
            output += "- XDTU_0:  None\n"
            output += "- YDTU_0:  None\n"
            output += "- ZDTU_0:  None\n"
        return output

    def define_from_fits(self, fitsobj, extnum=0):
        """Define class members from header information in FITS file.

        Parameters
        ----------
        fitsobj: file object
            FITS file whose header contains the DTU information
            needed to initialise the members of this class.
        extnum : int
            Extension number (first extension is 0)

        """

        # read input FITS file
        hdulist = fits.open(fitsobj)
        image_header = hdulist[extnum].header
        hdulist.close()

        # define DTU variables
        self.xdtu = image_header['xdtu']
        self.ydtu = image_header['ydtu']
        self.zdtu = image_header['zdtu']
        self.xdtu_0 = image_header['xdtu_0']
        self.ydtu_0 = image_header['ydtu_0']
        self.zdtu_0 = image_header['zdtu_0']

        # the attributes have been properly set
        self.defined = True
