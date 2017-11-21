from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import numpy as np
import os.path

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.polfit_residuals import polfit_residuals

from fit_pol_surface_renorm import fit_pol_surface_renorm
from fit_pol_surface_renorm import eval_pol_surface_renorm
from numina.array.stats import summary as statsummary

from emirdrp.core import EMIR_NBARS

from emir_definitions import VALID_GRISMS
from emir_definitions import VALID_FILTERS


def integrity_check(megadict, grism, spfilter):
    """Integrity check of megadict content.

    Parameters
    ----------
    megadict : dictionary of dictionaries
        Structure employed to store megadict information.
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.

    """

    # check main label in JSON file
    main_label = "megadict_grism_" + grism + "_filter_" + spfilter
    if megadict.keys() != [main_label]:
        raise ValueError("Invalid initial key in " + megadict_file)

    # slitlet keys
    valid_slitlets = ["slitlet" + str(i).zfill(2) for i in
                      range(1, EMIR_NBARS + 1)]
    read_slitlets = megadict[main_label].keys()
    read_slitlets.sort()
    for tmp_slitlet in read_slitlets:
        if tmp_slitlet not in valid_slitlets:
            raise ValueError("Unexpected slitlet key: " + tmp_slitlet)
        # for each slitlet, check valid DATE-OBS (ISO 8601)
        read_dateobs = megadict[main_label][tmp_slitlet].keys()
        read_dateobs.sort()
        for tmp_dateobs in read_dateobs:
            try:
                datetime.strptime(tmp_dateobs, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                print("Unexpected date_obs key: " + tmp_dateobs)
                raise
            # for each DATE-OBS, check expected fields
            tmp_dict = megadict[main_label][tmp_slitlet][tmp_dateobs]
            if "number_arc_lines" in tmp_dict:
                number_arc_lines = tmp_dict['number_arc_lines']
            else:
                print("ERROR:")
                print("grism...:", grism)
                print("slitlet.:", tmp_slitlet)
                print("date_obs:", tmp_dateobs)
                raise ValueError("number_arc_lines not found!")
            valid_keys = ["arcline" + str(i).zfill(4)
                          for i in range(number_arc_lines)] + \
                         ["csu_bar_slit_center",
                          "csu_bar_slit_width",
                          "number_arc_lines",
                          "rotang",
                          "wcal_cdelt1",
                          "wcal_crmax1",
                          "wcal_crmin1",
                          "wcal_crpix1",
                          "wcal_crval1",
                          "wcal_poly_coeff",
                          "z_info1",
                          "z_info2"]
            read_keys = tmp_dict.keys()
            for tmp_key in read_keys:
                if tmp_key not in valid_keys:
                    print("ERROR:")
                    print("grism...:", grism)
                    print("slitlet.:", tmp_slitlet)
                    print("date_obs:", tmp_dateobs)
                    raise ValueError("Unexpected key: " + tmp_key)
            # for each arcline, check expected fields
            for i in range(number_arc_lines):
                tmp_arcline = tmp_dict["arcline" + str(i).zfill(4)]
                valid_keys = ["reference", "slope", "xpos_0", "ypos_0",
                              "xpos_1", "ypos_1", "xpos_2", "ypos_2"]
                read_keys = tmp_arcline.keys()
                for tmp_key in read_keys:
                    if tmp_key not in valid_keys:
                        print("ERROR:")
                        print("grism...:", grism)
                        print("slitlet.:", tmp_slitlet)
                        print("date_obs:", tmp_dateobs)
                        print("arcline.:", tmp_arcline)
                        raise ValueError("Unexpected key: " + tmp_key)
            # check wcal_poly_coeff
            wcal_poly_coeff = tmp_dict["wcal_poly_coeff"]
            if type(wcal_poly_coeff) is not list:
                print("ERROR:")
                print("grism...:", grism)
                print("slitlet.:", tmp_slitlet)
                print("date_obs:", tmp_dateobs)
                raise ValueError("wcal_poly_coeff is not a list!")
    print("* Integrity check OK!")


def plot_crvalues_csu(megadict, grism, spfilter, debugplot=0):
    """Plot variation of CRMIN1, CRMAX1, CDELT1 with csu_bar_slit_center

    Parameters
    ----------
    megadict : dictionary of dictionaries
        Structure employed to store megadict information.
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.
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

    # declare local variables as (empty) python lists
    csu_bar_slit_center = []
    crmin1 = []
    crmax1 = []
    cdelt1 = []
    colsym = []

    # define main_label
    main_label = "megadict_grism_" + grism + "_filter_" + spfilter

    # determine number of available slitlets and define colors
    list_slitlets = megadict[main_label].keys()
    list_slitlets.sort()
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(list_slitlets)))

    # explore the whole megadict structure
    # (note that idum is not the number of the slitlet since some
    # slitlets can be missing)
    for idum, slitlet_label in enumerate(list_slitlets):
        list_date_obs = megadict[main_label][slitlet_label].keys()
        list_date_obs.sort()
        for date_obs in list_date_obs:
            tmp_dict = megadict[main_label][slitlet_label][date_obs]
            csu_bar_slit_center.append(
                tmp_dict['csu_bar_slit_center'])
            crmin1.append(tmp_dict['wcal_crmin1'])
            crmax1.append(tmp_dict['wcal_crmax1'])
            cdelt1.append(tmp_dict['wcal_cdelt1'])
            colsym.append(colors[idum])

    # transform python lists into numpy arrays
    csu_bar_slit_center = np.array(csu_bar_slit_center)
    crmin1 = np.array(crmin1)
    crmax1 = np.array(crmax1)
    cdelt1 = np.array(cdelt1)

    # plot each parameter as a function of csu_bar_slit_center
    parameter_list = ["crmin1", "crmax1", "cdelt1"]
    parameter_dict = dict(crmin1=crmin1, crmax1=crmax1, cdelt1=cdelt1)
    for name in parameter_list:
        parameter = parameter_dict[name]
        print(">>> " + name.upper() + ":")
        statsummary(parameter, debug=True)
        poly_dum, yres_dum = polfit_residuals(
            x=csu_bar_slit_center,
            y=parameter,
            color=colsym,
            xlabel='csu_bar_slit_center (mm)',
            ylabel=name.upper(),
            title='Grism ' + grism,
            deg=3, debugplot=debugplot)


def determine_unique_wavelengths(megadict, grism, spfilter):
    """Determine unique wavelengths in megadict.

    Parameters
    ----------
    megadict : dictionary of dictionaries
        Structure employed to store megadict information.
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.

    Returns
    -------
    list_unique_wavelengths : python list of strings
        List with string representation of unique wavelengths.

    """

    # define main_label
    main_label = "megadict_grism_" + grism + "_filter_" + spfilter

    # to avoid errors when comparing floats, the actual wavelengths are
    # handled as strings
    array_unique_wavelengths = np.array([], dtype='|S9')

    # explore the whole megadict structure
    list_slitlets = megadict[main_label].keys()
    list_slitlets.sort()
    for slitlet_label in list_slitlets:
        list_date_obs = megadict[main_label][slitlet_label].keys()
        list_date_obs.sort()
        for date_obs in list_date_obs:
            tmp_dict = megadict[main_label][slitlet_label][date_obs]
            number_arc_lines = tmp_dict['number_arc_lines']
            for i in range(number_arc_lines):
                arcline_label = "arcline" + str(i).zfill(4)
                tmp_reference = tmp_dict[arcline_label]['reference']
                array_unique_wavelengths = np.append(
                    array_unique_wavelengths, "{0:9.3f}".format(tmp_reference))
    array_unique_wavelengths = np.unique(array_unique_wavelengths)

    # return result as a python list
    return array_unique_wavelengths.tolist()


def model_unique_wavelength(megadict, grism, spfilter, unique_wavelength,
                            parameter, debugplot=0):
    """Compute model for a given wavelength using information in megadict.

    Parameters
    ----------
    megadict : dictionary of dictionaries
        Structure employed to store megadict information.
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.
    unique_wavelength : string
        String representation of unique wavelength.
    parameter : string
        Parameter to be modeled: 'xpos_0', 'ypos_0', 'xpos_1',
        'ypos_1', 'xpos_2' or 'ypos_2'.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    Returns
    -------
    status : bool
        Return True if the model has been computed, False otherwise.
    min_csu_bar_slit_center : float
        Minimum value of the fitted X values (csu_bar_slit_center).
    max_csu_bar_slit_center : float
        Maximum value of the fitted X values (csu_bar_slit_center).
    min_slit_number : int
        Minimum value of the fitted Y value (slitlet number).
    max_slit_number : int
        Maximum value of the fitted Y value (slitlet number).
    degx : int
        Maximum degree for X values (csu_bar_slit_center).
    degy : int
        Maximum degree for Y values (slitlet number).
    maskdeg : None or 2d numpy array, bool
        If mask[ix, iy] == True, the term x**ix + y**iy was employed
        in the fit. If maskdeg == None, all the terms were fitted.
    coeff : 1d numpy array, floats
        Fitted coefficients.
    norm_factors : tuple of floats
        Coefficients bx, cx, by, cy, bz, and cz employed to normalize
        the (x,y,z) values to the [-1,1] interval.

    """

    # protections
    if parameter not in ["xpos_0", "ypos_0",
                         "xpos_1", "ypos_1",
                         "xpos_2", "ypos_2"]:
        raise ValueError("Invalid parameter: " + parameter)

    # define main_label
    main_label = "megadict_grism_" + grism + "_filter_" + spfilter

    # declare local variables as (empty) python lists
    csu_bar_slit_center = []
    slit_number = []
    param = []

    # explore the whole megadict structure to fill the previous lists
    list_slitlets = megadict[main_label].keys()
    list_slitlets.sort()
    for slitlet_label in list_slitlets:
        islitlet = int(slitlet_label[7:9])  # determine slitlet number
        list_date_obs = megadict[main_label][slitlet_label].keys()
        list_date_obs.sort()
        for date_obs in list_date_obs:
            tmp_dict = megadict[main_label][slitlet_label][date_obs]
            number_arc_lines = tmp_dict['number_arc_lines']
            for i in range(number_arc_lines):
                arcline_label = "arcline" + str(i).zfill(4)
                tmp_reference = tmp_dict[arcline_label]['reference']
                tmp_reference_char = "{0:9.3f}".format(tmp_reference)
                if tmp_reference_char == unique_wavelength:
                    csu_bar_slit_center.append(tmp_dict['csu_bar_slit_center'])
                    slit_number.append(islitlet)
                    param.append(tmp_dict[arcline_label][parameter])

    # transform lists into numpy arrays
    csu_bar_slit_center = np.array(csu_bar_slit_center)
    slit_number = np.array(slit_number)
    param = np.array(param)

    # determine data ranges
    min_csu_bar_slit_center = csu_bar_slit_center.min()
    max_csu_bar_slit_center = csu_bar_slit_center.max()
    min_slit_number = slit_number.min()
    max_slit_number = slit_number.max()

    # number of available points for fit
    nfit = len(csu_bar_slit_center)
    number_unique_slitlets = len(np.unique(slit_number))
    if nfit < 6 or number_unique_slitlets < 3:
        status = False
        degx = 0
        degy = 0
        maskdeg = None
        coeff = None
        norm_factors = None
        return (status,
                min_csu_bar_slit_center, max_csu_bar_slit_center,
                min_slit_number, max_slit_number,
                degx, degy, maskdeg,
                coeff, norm_factors)

    # fit surfaces
    x = csu_bar_slit_center
    y = slit_number
    z = param
    degx = 2
    degy = 2
    maskdeg = np.ones((degx+1, degy+1))
    maskdeg[1, 2] = False
    maskdeg[2, 1] = False
    maskdeg[2, 2] = False
    coeff, norm_factors = fit_pol_surface_renorm(x=x, y=y, z=z,
                                                 degx=degx, degy=degy,
                                                 maskdeg=maskdeg)

    # plot surface
    if debugplot % 10 != 0:
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                             np.linspace(y.min(), y.max(), 100))
        zz = eval_pol_surface_renorm(xx, yy, coeff, norm_factors,
                                     degx=degx, degy=degy, maskdeg=maskdeg)
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('csu_bar_slit_center (mm)')
        ax.set_ylabel('slitlet number')
        im_show = plt.imshow(zz, origin='low', aspect='auto',
                             interpolation='nearest',
                             extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(im_show, shrink=1.0,
                     label=parameter,
                     orientation="horizontal")
        plt.scatter(x, y, c=z,
                    marker='o', edgecolor='k', s=100)
        plt.title("Grism: " + grism + ", Wavelength: " + unique_wavelength)
        pause_debugplot(debugplot, pltshow=True)

        # plot results
        colors = cm.rainbow(np.linspace(0, 1, EMIR_NBARS+1))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for slitlet_label in list_slitlets:
            islitlet = int(slitlet_label[7:9])  # determine slitlet number
            xp = []
            yp = []
            ymodel = []
            for val in zip(csu_bar_slit_center, slit_number, param):
                if val[1] == islitlet:
                    xp.append(val[0])
                    yp.append(val[2])
                    ymodel.append(eval_pol_surface_renorm(
                        x=np.array([val[0]]),
                        y=np.array([islitlet]),
                        coeff=coeff,
                        norm_factors=norm_factors,
                        degx=degx, degy=degy, maskdeg=maskdeg)[0])
            ax.scatter(xp, yp, color=colors[islitlet], marker='o',
                       edgecolor='k', s=100)
            ax.scatter(xp, ymodel, color=colors[islitlet], marker='s',
                       edgecolor='k', s=50)
        ax.set_xlabel('csu_bar_slit_center (mm)')
        ax.set_ylabel(parameter)
        plt.title("Grism: " + grism + ", Wavelength: " + unique_wavelength)
        pause_debugplot(debugplot, pltshow=True)

    # return results
    status = True
    return (status,
            min_csu_bar_slit_center, max_csu_bar_slit_center,
            min_slit_number, max_slit_number,
            degx, degy, maskdeg,
            coeff, norm_factors)


def all_elements_in_lis_equal(x):
    """Check that all elements in a list are equal.

    The method has been extracted from:
    http://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical

    Parameters
    ----------
    x : list
        List to be checked.

    Returns
    -------
    status : bool
        Return True if all element are equal, False otherwise.

    """

    status = not any((x[i] != x[i+1] for i in range(0, len(x)-1)))
    return status


def generate_modeldict(megadict, grism, spfilter, debugplot=0):
    """Compute dictionary modeldict.

    The output dictionary modeldict stores the models fitted for each
    unique wavelength (xpos_0, ypos_0, xpos_1, ypos_1, xpos_2, ypos_2
    as a function of csu_bar_slit_center and the slitlet number).

    Parameters
    ----------
    megadict : dictionary of dictionaries
        Structure employed to store megadict information.
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.
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
    modeldict : dictionary
        Structure where the model fitted for each unique wavelength is
        stored.

    """

    # declare output dictionary
    modeldict = {}

    # determine unique wavelengths
    list_unique_wavelengths = determine_unique_wavelengths(
        megadict, grism, spfilter
    )

    # define list of parameters
    parameter_list = ['xpos_0', 'ypos_0',
                      'xpos_1', 'ypos_1',
                      'xpos_2', 'ypos_2']

    # fit model for each unique wavelength
    for unique_wavelength in list_unique_wavelengths:
        print("Fitting model for reference=" + unique_wavelength)
        # fit model for each parameter
        status_list = []
        min_csu_bar_slit_center_list = []
        max_csu_bar_slit_center_list = []
        min_slit_number_list = []
        max_slit_number_list = []
        degx_list = []
        degy_list = []
        maskdeg_list = []
        coeff_list = []
        norm_factors_list = []
        for parameter in parameter_list:
            status, min_csu_bar_slit_center, max_csu_bar_slit_center, \
            min_slit_number, max_slit_number, degx, degy, maskdeg, \
            coeff, norm_factors = model_unique_wavelength(
                megadict=megadict,
                grism=grism,
                spfilter=spfilter,
                unique_wavelength=unique_wavelength,
                parameter=parameter,
                debugplot=debugplot
            )
            status_list.append(status)
            min_csu_bar_slit_center_list.append(min_csu_bar_slit_center)
            max_csu_bar_slit_center_list.append(max_csu_bar_slit_center)
            min_slit_number_list.append(min_slit_number)
            max_slit_number_list.append(max_slit_number)
            degx_list.append(degx)
            degy_list.append(degy)
            maskdeg_list.append(maskdeg)
            coeff_list.append(coeff)
            norm_factors_list.append(norm_factors)

        # check there are enough values to fit the model
        if np.all(status_list):
            # check consistency of solutions
            if not all_elements_in_lis_equal(min_csu_bar_slit_center_list):
                print(">>> min_csu_bar_slit_center_list:",
                      min_csu_bar_slit_center_list)
                raise ValueError("Inconsistent values")
            if not all_elements_in_lis_equal(max_csu_bar_slit_center_list):
                print(">>> max_csu_bar_slit_center_list:",
                      max_csu_bar_slit_center_list)
                raise ValueError("Inconsistent values")
            if not all_elements_in_lis_equal(min_slit_number_list):
                print(">>> min_slit_number_list:",
                      min_slit_number_list)
                raise ValueError("Inconsistent values")
            if not all_elements_in_lis_equal(max_slit_number_list):
                    print(">>> max_slit_number_list:",
                          max_slit_number_list)
                    raise ValueError("Inconsistent values")
            # store information in dictionary modeldict
            modeldict[unique_wavelength] = {
                'min_csu_bar_slit_center': min_csu_bar_slit_center_list[0],
                'max_csu_bar_slit_center': max_csu_bar_slit_center_list[0],
                'min_slit_number': min_slit_number_list[0],
                'max_slit_number': max_slit_number_list[0]}
            for ival, parameter in enumerate(parameter_list):
                key = 'degx_' + parameter
                modeldict[unique_wavelength][key] = degx_list[ival]
                key = 'degy_' + parameter
                modeldict[unique_wavelength][key] = degy_list[ival]
                key = 'maskdeg_' + parameter
                modeldict[unique_wavelength][key] = maskdeg_list[ival]
                key = 'coeff_' + parameter
                modeldict[unique_wavelength][key] = coeff_list[ival]
                key = 'norm_factors_' + parameter
                modeldict[unique_wavelength][key] = norm_factors_list[ival]
        else:
            print("WARNING: not enough values to fit " + unique_wavelength)

    # return result
    return modeldict


def get_info_for_wv_in_slitlet(megadict, grism, spfilter,
                               unique_wavelength, islitlet):
    """Return information for a particular wavelength in a given slitlet.

    Determine all the possible (xpos_i, ypos_i) values as a function
    of csu_bar_slit_center for a particular wavelength in the selected
    slitlet.

    Parameters
    ----------
    megadict : dictionary of dictionaries
        Structure employed to store megadict information.
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.
    unique_wavelength : string
        String representation of unique wavelength.
    islitlet : int
        Slitlet number.

    Returns
    -------
    csu_bar_slit_center : 1d numpy array, float
        Array of csu_bar_slit_center values.
    xpos_0 : 1d numpy array, floats
        Array of xpos_0 values.
    ypos_0 : 1d numpy array, floats
        Array of ypos_0 values.
    xpos_1 : 1d numpy array, floats
        Array of xpos_1 values.
    ypos_1 : 1d numpy array, floats
        Array of ypos_1 values.
    xpos_2 : 1d numpy array, floats
        Array of xpos_2 values.
    ypos_2 : 1d numpy array, floats
        Array of ypos_2 values.

    """

    # define main_label
    main_label = "megadict_grism_" + grism + "_filter_" + spfilter

    # declare initial variables as (empty) python lists
    csu_bar_slit_center = []
    xpos_0 = []
    ypos_0 = []
    xpos_1 = []
    ypos_1 = []
    xpos_2 = []
    ypos_2 = []

    # explore the megadict structure
    slitlet_label = "slitlet" + str(islitlet).zfill(2)
    list_date_obs = megadict[main_label][slitlet_label].keys()
    list_date_obs.sort()
    for date_obs in list_date_obs:
        tmp_dict = megadict[main_label][slitlet_label][date_obs]
        number_arc_lines = tmp_dict['number_arc_lines']
        for i in range(number_arc_lines):
            arcline_label = "arcline" + str(i).zfill(4)
            tmp_reference = tmp_dict[arcline_label]['reference']
            tmp_reference_char = "{0:9.3f}".format(tmp_reference)
            if tmp_reference_char == unique_wavelength:
                csu_bar_slit_center.append(
                    tmp_dict["csu_bar_slit_center"]
                )
                xpos_0.append(tmp_dict[arcline_label]['xpos_0'])
                ypos_0.append(tmp_dict[arcline_label]['ypos_0'])
                xpos_1.append(tmp_dict[arcline_label]['xpos_1'])
                ypos_1.append(tmp_dict[arcline_label]['ypos_1'])
                xpos_2.append(tmp_dict[arcline_label]['xpos_2'])
                ypos_2.append(tmp_dict[arcline_label]['ypos_2'])

    # transform lists into numpy arrays
    csu_bar_slit_center = np.array(csu_bar_slit_center)
    xpos_0 = np.array(xpos_0)
    ypos_0 = np.array(ypos_0)
    xpos_1 = np.array(xpos_1)
    ypos_1 = np.array(ypos_1)
    xpos_2 = np.array(xpos_2)
    ypos_2 = np.array(ypos_2)

    return csu_bar_slit_center, xpos_0, ypos_0, xpos_1, ypos_1, xpos_2, ypos_2


def plot_xpos_ypos_slope_vs_csu(megadict, grism, spfilter,
                                modeldict, islitlet, debugplot=0):
    """Plot xpos_i, ypos_i (i=0, 1 and 3) vs. csu_bar_slit_center.

    Parameters
    ----------
    megadict : dictionary of dictionaries
        Structure employed to store megadict information.
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.
    modeldict : dictionary
        Structure where the model fitted for each unique wavelength is
        stored.
    islitlet : int
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

    # determine unique wavelengths
    list_unique_wavelengths = determine_unique_wavelengths(
        megadict, grism, spfilter
    )

    for parameter in ['xpos_0', 'ypos_0',
                      'xpos_1', 'ypos_1',
                      'xpos_2', 'ypos_2']:
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        for unique_wavelength in list_unique_wavelengths:
            csu_bar_slit_center, xpos_0, ypos_0, xpos_1, ypos_1, \
            xpos_2, ypos_2 = get_info_for_wv_in_slitlet(
                megadict, grism, spfilter,
                unique_wavelength,
                islitlet
            )
            npoints_fit = len(csu_bar_slit_center)
            if npoints_fit > 0:
                if parameter == 'xpos_0':
                    param = xpos_0
                elif parameter == 'ypos_0':
                    param = ypos_0
                elif parameter == 'xpos_1':
                    param = xpos_1
                elif parameter == 'ypos_1':
                    param = ypos_1
                elif parameter == 'xpos_2':
                    param = xpos_2
                elif parameter == 'ypos_2':
                    param = ypos_2
                else:
                    raise ValueError("Unexpected parameter = " + parameter)
                xmin = min(xmin, csu_bar_slit_center.min())
                xmax = max(xmax, csu_bar_slit_center.max())
                ymin = min(ymin, param.min())
                ymax = max(ymax, param.max())
                ax.plot(csu_bar_slit_center, param, "o-")
                tmpdict = modeldict[unique_wavelength]
                if tmpdict['coeff_xpos'] is not None:
                    param_model = eval_pol_surface_renorm(
                        x=csu_bar_slit_center,
                        y=np.array([islitlet] * npoints_fit),
                        coeff=tmpdict['coeff_' + parameter],
                        norm_factors=tmpdict['norm_factors_' + parameter],
                        degx=tmpdict['degx_' + parameter],
                        degy=tmpdict['degy_' + parameter],
                        maskdeg=tmpdict['maskdeg_' + parameter])
                    ymin = min(ymin, param_model.min())
                    ymax = max(ymax, param_model.max())
                    ax.plot(csu_bar_slit_center, param_model, "k*")
        dx = xmax - xmin
        xmin -= dx/20
        xmax += dx/20
        ax.set_xlim([xmin, xmax])
        dy = ymax - ymin
        ymin -= dy/20
        ymax += dy/20
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel("csu_bar_slit_center (mm)")
        if parameter in ['xpos_0', 'xpos_1', 'xpos_2']:
            ax.set_ylabel(parameter + " (pixel, from 1 to NAXIS1)")
        else:
            ax.set_ylabel(parameter + " (pixel, from 1 to NAXIS2)")
        plt.title("Grism: " + grism +
                  ", Slitlet: " + str(islitlet))
        pause_debugplot(debugplot, pltshow=True)


def generate_wvdict(megadict, grism, spfilter, modeldict, debugplot=0):
    """Generate wvdict dictionary.

    Parameters
    ----------
    megadict : dictionary of dictionaries
        Structure employed to store megadict information.
    grism : string
        Character string ("J", "H", "K" or LR) indicating the grism.
    spfilter : string
        Character string ("J", "H", "Ksp",...) indicating the filter.
    modeldict : dictionary
        Structure where the model fitted for each unique wavelength is
        stored.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    Returns
    -------
    wvdict : dictionary of dictionary
        Structure employed to store wvdict information.

    """

    # define main_label
    main_label = "wvdict_grism_" + grism + "_filter_" + spfilter
    main_label_megadict = "megadict_grism_" + grism + "_filter_" + spfilter

    # determine unique wavelengths
    list_unique_wavelengths = determine_unique_wavelengths(
        megadict, grism, spfilter
    )

    # remove unique wavelengths without model
    list_unique_wavelengths_filtered = []
    for unique_wavelength in list_unique_wavelengths:
        if unique_wavelength in modeldict.keys():
            list_unique_wavelengths_filtered.append(unique_wavelength)

    # generate wvdict dictionary
    wvdict = {main_label : {}}
    for islitlet in range(1, EMIR_NBARS+1):
        if islitlet == 1:
            print("---ooo---")
        print("Slitlet number: ", islitlet)
        slitlet_label = "slitlet" + str(islitlet).zfill(2)
        wvdict[main_label][slitlet_label] = {}
        if slitlet_label in megadict[main_label_megadict].keys():
            # fit xpos_i, ypos_i vs. csu_bar_slit_center for the
            # particular slitlet and unique wavelength using both the
            # data and the models
            for unique_wavelength in list_unique_wavelengths_filtered:
                csu_bar_slit_center, xpos_0, ypos_0,\
                    xpos_1, ypos_1, xpos_2, ypos_2 = \
                    get_info_for_wv_in_slitlet(megadict, grism, spfilter,
                                               unique_wavelength,
                                               islitlet)
                npoints_fit = len(csu_bar_slit_center)
                modeldict_wv = modeldict[unique_wavelength]
                min_slit_model = modeldict_wv['min_slit_number']
                max_slit_model = modeldict_wv['max_slit_number']
                min_csu_model = modeldict_wv['min_csu_bar_slit_center']
                max_csu_model = modeldict_wv['max_csu_bar_slit_center']
                # the line has been detected in the current slitlet
                # in more than one position (allowing for a polynomial
                # fit)
                if npoints_fit > 1:
                    # define new entry in wvdict
                    wvdict[main_label][slitlet_label][unique_wavelength] = {}
                    # alias for previous structure
                    wvdictz = wvdict[main_label][slitlet_label][unique_wavelength]
                    # store data range for csu_bar_slit_center
                    wvdictz['data_csu_bar_slit_center_0'] = \
                        csu_bar_slit_center.min()
                    wvdictz['data_csu_bar_slit_center_1'] = \
                        csu_bar_slit_center.max()
                    # store ranges for fitted data in models
                    wvdictz['model_csu_bar_slit_center_0'] = min_csu_model
                    wvdictz['model_csu_bar_slit_center_1'] = max_csu_model
                    wvdictz['model_islitlet_0'] = min_slit_model
                    wvdictz['model_islitlet_1'] = max_slit_model
                    # set polynomial degree
                    if npoints_fit == 2:
                        poldeg = 1
                    else:
                        poldeg = 2
                    # generate fits to relevant parameters
                    parameter_list = ['xpos_0', 'ypos_0',
                                      'xpos_1', 'ypos_1',
                                      'xpos_2', 'ypos_2']
                    parameter_dict = dict(xpos_0=xpos_0, ypos_0=ypos_0,
                                          xpos_1=xpos_1, ypos_1=ypos_1,
                                          xpos_2=xpos_2, ypos_2=ypos_2)
                    for param_name in parameter_list:
                        parameter = parameter_dict[param_name]
                        # polynomial fit to original data
                        poly, yres = polfit_residuals(
                            x=csu_bar_slit_center,
                            y=parameter,
                            deg=poldeg,
                            xlabel='csu_bar_slit_center',
                            ylabel=param_name,
                            title="Grism: " + grism +
                                  ", Slitlet: " + str(islitlet) +
                                  ", wv: " + unique_wavelength,
                            debugplot=debugplot)
                        poly_label = "data_poly_" + param_name + "_vs_csu"
                        wvdictz[poly_label] = dict(enumerate(poly.coef))
                        # model predictions
                        param_model = eval_pol_surface_renorm(
                            x=csu_bar_slit_center,
                            y=np.array([islitlet] * len(csu_bar_slit_center)),
                            coeff=modeldict_wv['coeff_' + param_name],
                            norm_factors=modeldict_wv['norm_factors_' +
                                                      param_name],
                            degx=modeldict_wv['degx_' + param_name],
                            degy=modeldict_wv['degy_' + param_name],
                            maskdeg=modeldict_wv['maskdeg_' + param_name])
                        # polynomial fit to modeled data
                        poly, yres = polfit_residuals(
                            x=csu_bar_slit_center,
                            y=param_model,
                            deg=poldeg,
                            xlabel='csu_bar_slit_center',
                            ylabel=param_name + ' (model)',
                            title="Grism: " + grism +
                                  ", Slitlet: " + str(islitlet) +
                                  ", wv: " + unique_wavelength,
                            debugplot=debugplot)
                        poly_label = "model_poly_" + param_name + "_vs_csu"
                        wvdictz[poly_label] = dict(enumerate(poly.coef))
                else:
                    # check that the model was fitted with data
                    # corresponding to slitlets above and below the
                    # current one, and that there is a non null range
                    # in csu_bar_slit_center
                    if (min_slit_model <= islitlet <= max_slit_model) and \
                            (min_csu_model < max_csu_model):
                        # include wavelength in wvdict structure
                        wvdict[main_label][slitlet_label][unique_wavelength] = {}
                        wvdictz = \
                            wvdict[main_label][slitlet_label][unique_wavelength]
                        # store ranges for fitted data in models
                        wvdictz['model_csu_bar_slit_center_0'] = min_csu_model
                        wvdictz['model_csu_bar_slit_center_1'] = max_csu_model
                        wvdictz['model_islitlet_0'] = min_slit_model
                        wvdictz['model_islitlet_1'] = max_slit_model
                        # define range in csu_bar_slit_center
                        npoints_fit = 20
                        poldeg = 2
                        csu_bar_slit_center = np.linspace(
                            start=min_csu_model,
                            stop=max_csu_model,
                            num=npoints_fit
                        )
                        # generate fits to relevant parameters
                        parameter_list = ['xpos_0', 'ypos_0',
                                          'xpos_1', 'ypos_1',
                                          'xpos_2', 'ypos_2']
                        for param_name in parameter_list:
                            # model predictions
                            param_model = eval_pol_surface_renorm(
                                x=csu_bar_slit_center,
                                y=np.array(
                                    [islitlet] * len(csu_bar_slit_center)),
                                coeff=modeldict_wv['coeff_' + param_name],
                                norm_factors=modeldict_wv['norm_factors_' +
                                                          param_name],
                                degx=modeldict_wv['degx_' + param_name],
                                degy=modeldict_wv['degy_' + param_name],
                                maskdeg=modeldict_wv['maskdeg_' + param_name])
                            # polynomial fit to modeled data
                            poly, yres = polfit_residuals(
                                x=csu_bar_slit_center,
                                y=param_model,
                                deg=poldeg,
                                xlabel='csu_bar_slit_center',
                                ylabel=param_name + ' (model)',
                                title="Grism: " + grism +
                                      ", Slitlet: " + str(islitlet) +
                                      ", wv: " + unique_wavelength,
                                debugplot=debugplot)
                            poly_label = "model_poly_" + param_name + "_vs_csu"
                            wvdictz[poly_label] = dict(enumerate(poly.coef))
        else:
            print("WARNING: this slit is not in megadict!")
            for unique_wavelength in list_unique_wavelengths_filtered:
                modeldict_wv = modeldict[unique_wavelength]
                min_slit_model = modeldict_wv['min_slit_number']
                max_slit_model = modeldict_wv['max_slit_number']
                min_csu_model = modeldict_wv['min_csu_bar_slit_center']
                max_csu_model = modeldict_wv['max_csu_bar_slit_center']
                # check that the model was fitted with data
                # corresponding to slitlets above and below the current
                # one, and there is a non null range in
                # csu_bar_slit_center
                if (min_slit_model <= islitlet <= max_slit_model) and \
                        (min_csu_model < max_csu_model):
                    # include wavelength in wvdict structure
                    wvdict[main_label][slitlet_label][unique_wavelength] = {}
                    wvdictz = \
                        wvdict[main_label][slitlet_label][unique_wavelength]
                    # store ranges for fitted data in models
                    wvdictz['model_csu_bar_slit_center_0'] = min_csu_model
                    wvdictz['model_csu_bar_slit_center_1'] = max_csu_model
                    wvdictz['model_islitlet_0'] = min_slit_model
                    wvdictz['model_islitlet_1'] = max_slit_model
                    # define range in csu_bar_slit_center
                    npoints_fit = 20
                    poldeg = 2
                    csu_bar_slit_center = np.linspace(
                        start=min_csu_model,
                        stop=max_csu_model,
                        num=npoints_fit
                    )
                    # generate fits to relevant parameters
                    parameter_list = ['xpos_0', 'ypos_0',
                                      'xpos_1', 'ypos_1',
                                      'xpos_2', 'ypos_2']
                    for parameter in parameter_list:
                        # model predictions
                        param_model = eval_pol_surface_renorm(
                            x=csu_bar_slit_center,
                            y=np.array([islitlet] * len(csu_bar_slit_center)),
                            coeff=modeldict_wv['coeff_' + param_name],
                            norm_factors=modeldict_wv['norm_factors_' +
                                                      param_name],
                            degx=modeldict_wv['degx_' + param_name],
                            degy=modeldict_wv['degy_' + param_name],
                            maskdeg=modeldict_wv['maskdeg_' + param_name])
                        # polynomial fit to modeled data
                        poly, yres = polfit_residuals(
                            x=csu_bar_slit_center,
                            y=param_model,
                            deg=poldeg,
                            xlabel='csu_bar_slit_center',
                            ylabel=param_name + ' (model)',
                            title="Grism: " + grism +
                                  ", Slitlet: " + str(islitlet) +
                                  ", wv: " + unique_wavelength,
                            debugplot=debugplot)
                        poly_label = "model_poly_" + param_name + "_vs_csu"
                        wvdictz[poly_label] = dict(enumerate(poly.coef))

    return wvdict


if __name__ == "__main__":

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("grism",
                        help="Grism name ('J', 'H', 'K', 'LR')")
    parser.add_argument("filter",
                        help="Filter name ('J', 'H', 'Ksp',...)")
    parser.add_argument("--stat_plots",
                        help="""
                        Display CRMIN1, CRMAX1, CDELT1 vs csu_bar_slit_center.
                        (default=no)""",
                        default="no")
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=0)",
                        default=0)
    args = parser.parse_args()

    # read grism
    grism = args.grism
    if grism not in VALID_GRISMS:
        raise ValueError("Unexpected grism: " + grism)

    # read filter name
    spfilter = args.filter
    if spfilter not in VALID_FILTERS:
        raise ValueError("Filter=" + spfilter + " is not in valid filter list")

    # read stat_plots
    stat_plots = (args.stat_plots == "yes")

    # read debugplot value
    debugplot = int(args.debugplot)

    # read megadict_grism_[grism]_filter_[filter].json
    main_label = "megadict_grism_" + grism + "_filter_" + spfilter
    megadict_file = main_label + ".json"
    if os.path.isfile(megadict_file):
        megadict = json.loads(open(megadict_file).read())
        print('\n>>> Reading megadict from file:')
        print(megadict_file)
    else:
        raise ValueError("File " + megadict_file + " not found!")
    # print(json.dumps(megadict, indent=4, sort_keys=True))

    # integrity check
    integrity_check(megadict, grism, spfilter)

    # variation of CRMIN1, CRMAX1 and CDELT1 with csu_bar_slit_center
    if stat_plots:
        plot_crvalues_csu(megadict, grism, spfilter, debugplot=12)

    # compute model for each unique wavelength with enough
    # measurements as a function of csu_bar_slit_center and slitlet
    # number
    modeldict = generate_modeldict(megadict, grism, spfilter,
                                   debugplot=debugplot)

    # display xpos_i, ypos_i (i=0, 1, 2) vs. csu_bar_slit_center for
    # all the unique wavelengths in a given slitlet
    if debugplot % 10 != 0:
        for islitlet in range(1, EMIR_NBARS + 1):
            print("Slitlet number: ", islitlet)
            slitlet_label = "slitlet" + str(islitlet).zfill(2)
            if slitlet_label in megadict[main_label].keys():
                plot_xpos_ypos_slope_vs_csu(megadict, grism, spfilter,
                                            modeldict, islitlet,
                                            debugplot=debugplot)
            else:
                print("WARNING: this slit is not in megadict!")

    # generate wvdict structure and store it in
    # wvdict_grism_[grism]_filter_[filter].json file
    wvdict = generate_wvdict(megadict, grism, spfilter, modeldict)
    outfile = "wvdict_grism_" + grism + "_filter_" + spfilter + ".json"
    with open(outfile, 'w') as fstream:
        json.dump(wvdict, fstream, indent=4, sort_keys=True)
        print('>>> Saving file ' + outfile)
