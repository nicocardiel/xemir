from __future__ import division
from __future__ import print_function

from astropy.io import fits

from emirdrp.core import EMIR_NBARS


class SlitletArrangement:
    """SlitletArangement definition.

    Attributes
    ----------
    csu_bar_left : list of floats
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
    xdtu : float
        XDTU fits keyword value.
    ydtu : float
        YDTU fits keyword value.
    xdtu_0 : float
        XDTU_0 fits keyword value.
    ydtu_0 : float
        YDTU_0 fits keyword value.
    defined : bool
        Indicates whether the CSU parameters have been properly defined.

    """

    def __init__(self):
        self.csu_bar_left = None
        self.csu_bar_right = None
        self.csu_bar_slit_center = None
        self.csu_bar_slit_width = None
        self.xdtu = None
        self.ydtu = None
        self.xdtu_0 = None
        self.ydtu_0 = None
        self.defined = False

    def __str__(self):
        output = "<SlitletArrangement instance>\n"
        if self.defined:
            strdum = "- XDTU..: {0:8.3f}\n".format(self.xdtu)
            output += strdum
            strdum = "- YDTU..: {0:8.3f}\n".format(self.ydtu)
            output += strdum
            strdum = "- XDTU_0: {0:8.3f}\n".format(self.xdtu_0)
            output += strdum
            strdum = "- YDTU_0: {0:8.3f}\n".format(self.ydtu_0)
            output += strdum
        else:
            output += "- XDTU..:  None\n"
            output += "- YDTU..:  None\n"
            output += "- XDTU_0:  None\n"
            output += "- YDTU_0:  None\n"
        for i in range(EMIR_NBARS):
            ibar = i + 1
            strdum = "- [BAR{0:2d}] left, right, center, width: ".format(ibar)
            output += strdum
            if self.defined:
                strdum = "{0:7.3f} {1:7.3f} {2:7.3f} {3:7.3f}\n".format(
                    self.csu_bar_left[i], self.csu_bar_right[i],
                    self.csu_bar_slit_center[i], self.csu_bar_slit_width[i]
                )
                output += strdum
            else:
                output += 4 * "   None " + "\n"
        return output

    def define_from_fits(self, fitsfile, extnum=0):
        """Define class members from header information in FITS file.

        Parameters
        ----------
        fitsfile: file object
            FITS file whose header contains the CSU bar information
            needed to initialise the members of this class.
        extnum : int
            Extension number (first extension is 0)

        """

        # read input FITS file
        hdulist = fits.open(fitsfile)
        image_header = hdulist[extnum].header
        hdulist.close()

        # define DTU variables
        self.xdtu = image_header['xdtu']
        self.ydtu = image_header['ydtu']
        self.xdtu_0 = image_header['xdtu_0']
        self.ydtu_0 = image_header['ydtu_0']

        # declare arrays to store configuration of CSU bars
        self.csu_bar_left = []
        self.csu_bar_right = []
        self.csu_bar_slit_center = []
        self.csu_bar_slit_width = []

        for i in range(EMIR_NBARS):
            ibar = i + 1
            keyword = 'CSUP' + str(ibar)
            if keyword in image_header:
                self.csu_bar_left.append(image_header[keyword])
            else:
                raise ValueError("Expected keyword " + keyword + " not found!")
            keyword = 'CSUP' + str(ibar + EMIR_NBARS)
            if keyword in image_header:
                # set the same origin as the one employed for csu_bar_left
                self.csu_bar_right.append(341.5 - image_header[keyword])
            else:
                raise ValueError("Expected keyword " + keyword + " not found!")
            self.csu_bar_slit_center.append(
                (self.csu_bar_left[i] + self.csu_bar_right[i]) / 2
            )
            self.csu_bar_slit_width.append(
                self.csu_bar_right[i] - self.csu_bar_left[i]
            )

        # data have been properly set
        self.defined = True
