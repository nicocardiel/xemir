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
    defined : bool
        Indicates whether the CSU parameters have been properly defined.

    """

    def __init__(self):
        self.csu_bar_left = None
        self.csu_bar_right = None
        self.csu_bar_slit_center = None
        self.csu_bar_slit_width = None
        self.defined = False

    def __str__(self):
        output = "<SlitletArrangement instance>\n"
        for ibar in range(EMIR_NBARS):
            cbar = str(ibar).zfill(2)
            output += "- [" + cbar + "] left.......:"
            if self.defined:
                output += str(self.csu_bar_left[ibar]) + "\n"
            else:
                output += "None\n"
            output += "- [" + cbar + "] right......:"
            if self.defined:
                output += str(self.csu_bar_right[ibar]) + "\n"
            else:
                output += "None\n"
            output += "- [" + cbar + "] slit center:"
            if self.defined:
                output += str(self.csu_bar_slit_center[ibar]) + "\n"
            else:
                output += "None\n"
            output += "- [" + cbar + "]slit width.:"
            if self.defined:
                output += str(self.csu_bar_slit_width[ibar]) + "\n"
            else:
                output += "None\n"
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

        Returns
        -------

        """

        # read input FITS file
        hdulist = fits.open(fitsfile)
        image_header = hdulist[extnum].header
        hdulist.close()

        # declare arrays to store CSU bar configuration
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
