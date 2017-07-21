from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import numpy as np

from emirdrp.core import EMIR_NBARS

from numina.array.display.ximshow import ximshow
from numina.array.display.pause_debugplot import pause_debugplot

"""
Compute image distortion using the equations derived by Carlos Gonzalez
in imaging mode (from astrometric measurements of stars).

convert virtual pixel to real plxel
macro exvp 2{
define  cf (0.1944*pi/(180*3600))
set ra=sqrt(($1-1024.5)**2+($2-1024.5)**2)
set r=$cf*ra
set rr1=(1+14606.7*r**2+1739716115.1*r**4)
set thet=atan(($1-1024.5)/($2-1024.5))
set thet=($2<1024.5)?thet-pi:thet
set nx1=(rr1*ra*sin(thet))+1024.5
set ny1=(rr1*ra*cos(thet))+1024.5
define nnx (nx1[0])
define nny (ny1[0])
write standard  in real pixals $nnx $nny
}

convert real pixel to virtual pixel.
macro pvex 2{
define  cf (0.1944*pi/(180*3600))
set ra=sqrt(($1-1024.5)**2+($2-1024.5)**2)
set r=$cf*ra
set rr1=(1.000051-14892*r**2 -696254464*r**4)
set thet=atan(($1-1024.5)/($2-1024.5))
set thet=($2<1024)?thet-pi:thet
set nx1=(rr1*ra*sin(thet))+1024.5
set ny1=(rr1*ra*cos(thet))+1024.5
define nnx (nx1[0])
define nny (ny1[0])
write standard  in virtual pixals $nnx $nny
}

"""


def exvp_scalar(x, y, x0, y0, c1, c2):
    """Convert virtual pixel to real pixel.

    Parameters
    ----------
    x : float
        X coordinate (pixel).
    y : float
        Y coordinate (pixel).
    x0 : float
        X coordinate of reference pixel.
    y0 : float
        Y coordinate of reference pixel.
    c1 : float
        Coefficient corresponding to the term r**2 in distortion
        equation.
    c2 : float
        Coefficient corresponding to the term r**4 in distortion
        equation.

    Returns
    -------
    xdist, ydist : tuple of floats
        Distorted coordinates.

    """
    # plate scale: 0.1944 arcsec/pixel
    # conversion factor (in radian/pixel)
    factor = 0.1944 * np.pi/(180.0*3600)
    # distance from image center (pixels)
    r_pix = np.sqrt((x - x0)**2 + (y - y0)**2)
    # distance from imagen center (radians)
    r_rad = factor * r_pix
    # radial distortion: this number is 1.0 for r=0 and increases
    # slightly (reaching values around 1.033) for r~sqrt(2)*1024
    # (the distance to the corner of the detector measured from the
    # center)
    rdist = (1 + c1 * r_rad**2 + c2 * r_rad**4)
    # angle measured from the Y axis towards the X axis
    theta = np.arctan((x - x0)/(y - y0))
    if y < y0:
        theta = theta - np.pi
    # distorted coordinates
    xdist = (rdist * r_pix * np.sin(theta)) + x0
    ydist = (rdist * r_pix * np.cos(theta)) + y0
    return xdist, ydist


def exvp(x, y, x0, y0, c1, c2):
    """Convert virtual pixel(s) to real pixel(s).

    This function makes use of exvp_scalar(), which performs the
    conversion for a single point (x, y), over an array of X and Y
    values.

    Parameters
    ----------
    x : array-like
        X coordinate (pixel).
    y : array-like
        Y coordinate (pixel).
    x0 : float
        X coordinate of reference pixel.
    y0 : float
        Y coordinate of reference pixel.
    c1 : float
        Coefficient corresponding to the term r**2 in distortion
        equation.
    c2 : float
        Coefficient corresponding to the term r**4 in distortion
        equation.

    Returns
    -------
    xdist, ydist : tuple of floats (or two arrays of floats)
        Distorted coordinates.

    """
    if all([np.isscalar(x), np.isscalar(y)]):
        xdist, ydist = exvp_scalar(x, y, x0=x0, y0=y0, c1=c1, c2=c2)
        return xdist, ydist
    elif any([np.isscalar(x), np.isscalar(y)]):
        raise ValueError("invalid mixture of scalars and arrays")
    else:
        xdist = []
        ydist = []
        for x_, y_ in zip(x, y):
            xdist_, ydist_ = exvp_scalar(x_, y_, x0=x0, y0=y0, c1=c1, c2=c2)
            xdist.append(xdist_)
            ydist.append(ydist_)
        return np.array(xdist), np.array(ydist)


def main(args=None):
    parser = argparse.ArgumentParser(prog='distortion_emir_spec')
    parser.add_argument("fitsfile",
                        help="FITS file name",
                        type=argparse.FileType('r'))
    args = parser.parse_args(args)

    hdulist = fits.open(args.fitsfile.name)
    image2d = hdulist[0].data
    hdulist.close()

    ax = ximshow(image2d=image2d, show=False, geometry=(0, 0, 700, 600))

    slit_height = 33.5
    slit_gap = 3.95
    slit_dist = slit_height + slit_gap
    y_baseline = 2  # 1(J-J), 7(H-H), 3(K-Ksp), -85(LR-HK), -87(LR-YJ)
    x0 = 1024.5
    y0 = 1024.5
    c1 = 14606.7
    c2 = 1739716115.1

    xp = np.linspace(1, 2048, num=100)
    xv = np.linspace(24.5, 2024.5, num=21)
    for islitlet in range(1, EMIR_NBARS + 1):
        # y-coordinates at x=1024.5
        ybottom = y_baseline + (islitlet - 1) * slit_dist
        ytop = ybottom + slit_height
        # undistorted lower and upper slitlet boundaries
        yp_bottom = np.ones(100) * ybottom
        yp_top = np.ones(100) * ytop
        # distorted lower boundary
        xdist, ydist = exvp(xp, yp_bottom, x0=x0, y0=y0, c1=c1, c2=c2)
        ax.plot(xdist, ydist, 'b-')
        # distorted upper boundary
        xdist, ydist = exvp(xp, yp_top, x0=x0, y0=y0, c1=c1, c2=c2)
        ax.plot(xdist, ydist, 'g-')
        # display distorsion of vertical lines
        for xv_ in xv:
            xdist, ydist = exvp([xv_, xv_], [ybottom, ytop],
                                x0=x0, y0=y0, c1=c1, c2=c2)
            ax.plot(xdist, ydist, 'c-')
        # display central point and label at x0
        xdist_bottom, ydist_bottom = exvp(x0, ybottom,
                                          x0=x0, y0=y0, c1=c1, c2=c2)
        xdist_top, ydist_top = exvp(x0, ytop,
                                    x0=x0, y0=y0, c1=c1, c2=c2)
        ax.text(1024.5, (ydist_bottom + ydist_top)/2, str(islitlet),
                fontsize=10, va='center', ha='center',
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="grey", ),
                color='blue', fontweight='bold', backgroundcolor='white')

    pause_debugplot(12, pltshow=True)


if __name__ == "__main__":

    main()
