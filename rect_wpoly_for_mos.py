from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import numpy as np
import sys
from uuid import uuid4

from numina.array.display.fileinfo import list_fileinfo_from_txt
from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection
from numina.array.distortion import ncoef_fmap

from dtu_configuration import DtuConfiguration

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def islitlet_progress(islitlet, islitlet_max):
    """Auxiliary function to print out progress in loop of slitlets.

    Parameters
    ----------
    islitlet : int
        Current slitlet number.
    islitlet_max : int
        Maximum slitlet number.

    """
    if islitlet % 10 == 0:
        cout = str(islitlet // 10)
    else:
        cout = '.'
    sys.stdout.write(cout)
    if islitlet == islitlet_max:
        sys.stdout.write('\n')
    sys.stdout.flush()


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='rect_wpoly_for_mos')
    # required arguments
    parser.add_argument("input_list",
                        help="TXT file with list JSON files derived from "
                             "longslit data")
    parser.add_argument("--out_MOSlibrary", required=True,
                        help="Output JSON file with results",
                        type=argparse.FileType('w'))
    # optional arguments
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting & debugging options"
                             " (default=0)",
                        default=0, type=int,
                        choices=DEBUGPLOT_CODES)
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")
    args = parser.parse_args(args)

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    # ---

    # Read input TXT file with list of JSON files
    list_json_files = list_fileinfo_from_txt(args.input_list)
    nfiles = len(list_json_files)
    if abs(args.debugplot) >= 10:
        print('>>> Number of input JSON files:', nfiles)
        for item in list_json_files:
            print(item)
    if nfiles < 2:
        raise ValueError("Insufficient number of input JSON files")

    # protections: check consistency of grism, filter and DTU configuration
    json_first_longslit = json.loads(open(list_json_files[0].filename).read())
    dtu_conf = DtuConfiguration()
    dtu_conf.define_from_dictionary(json_first_longslit['dtu_configuration'])
    filter_name = json_first_longslit['tags']['filter']
    grism_name = json_first_longslit['tags']['grism']
    islitlet_min = json_first_longslit['tags']['islitlet_min']
    islitlet_max = json_first_longslit['tags']['islitlet_max']
    for ifile in range(1, nfiles):
        json_tmp = json.loads(open(list_json_files[ifile].filename).read())
        dtu_conf_tmp = DtuConfiguration()
        dtu_conf_tmp.define_from_dictionary(json_tmp['dtu_configuration'])
        filter_tmp = json_tmp['tags']['filter']
        grism_tmp = json_tmp['tags']['grism']
        islitlet_min_tmp = json_tmp['tags']['islitlet_min']
        islitlet_max_tmp = json_tmp['tags']['islitlet_max']
        if dtu_conf != dtu_conf_tmp:
            print(dtu_conf)
            print(dtu_conf_tmp)
            raise ValueError("Unexpected different DTU configurations found")
        if filter_name != filter_tmp:
            print(filter_name)
            print(filter_tmp)
            raise ValueError("Unexpected different filter found")
        if grism_name != grism_tmp:
            print(grism_name)
            print(grism_tmp)
            raise ValueError("Unexpected different grism found")
        if islitlet_min != islitlet_min_tmp:
            print(islitlet_min)
            print(islitlet_min_tmp)
            raise ValueError("Unexpected different islitlet_min_found")
        if islitlet_max != islitlet_max_tmp:
            print(islitlet_max)
            print(islitlet_max_tmp)
            raise ValueError("Unexpected different islitlet_max_found")

    # ---

    # Read and store all the longslit data
    list_json_longslits = []
    for ifile in range(nfiles):
        json_tmp = json.loads(open(list_json_files[ifile].filename).read())
        list_json_longslits.append(json_tmp)

    # ---

    # Check that all the expected slitlets are defined

    for ifile in range(nfiles):
        tmpdict = list_json_longslits[ifile]['contents']
        for islitlet in range(islitlet_min, islitlet_max + 1):
            cslitlet = 'slitlet' + str(islitlet).zfill(2)
            if cslitlet not in tmpdict:
                raise ValueError(cslitlet + " not found!")

    # ---

    # Initialize structure to save results into an ouptut JSON file
    outdict = {}
    outdict['instrument'] = 'EMIR'
    outdict['meta-info'] = {}
    outdict['meta-info']['creation_date'] = datetime.now().isoformat()
    outdict['meta-info']['description'] = \
        'wavelength calibration polynomials and rectification ' \
        'coefficients for MOS'
    outdict['meta-info']['recipe_name'] = 'undefined'
    outdict['meta-info']['origin'] = {}
    outdict['meta-info']['origin']['wpoly_longslits'] = {}
    for ifile in range(nfiles):
        cdum = 'longslit_' + str(ifile + 1).zfill(3) + '_uuid'
        outdict['meta-info']['origin']['wpoly_longslits'][cdum] = \
            list_json_longslits[ifile]['uuid']
    outdict['tags'] = {}
    outdict['tags']['grism'] = grism_name
    outdict['tags']['filter'] = filter_name
    outdict['tags']['islitlet_min'] = islitlet_min
    outdict['tags']['islitlet_max'] = islitlet_max
    outdict['dtu_configuration'] = dtu_conf.outdict()
    outdict['uuid'] = str(uuid4())
    outdict['contents'] = {}

    # check that order for rectification transformations is the same for all
    # the slitlets and longslit configurations
    order_check_list = []
    for ifile in range(nfiles):
        tmpdict = list_json_longslits[ifile]['contents']
        for islitlet in range(islitlet_min, islitlet_max + 1):
            cslitlet = 'slitlet' + str(islitlet).zfill(2)
            ttd_order = tmpdict[cslitlet]['ttd_order']
            if ttd_order is not None:
                order_check_list.append(ttd_order)
            ttd_order_modeled = tmpdict[cslitlet]['ttd_order_modeled']
            order_check_list.append(ttd_order_modeled)
    # remove duplicates in list
    order_no_duplicates = list(set(order_check_list))
    if len(order_no_duplicates) != 1:
        print('order_no_duplicates:', order_no_duplicates)
        raise ValueError('tdd_order is not constant!')
    order = int(order_no_duplicates[0])
    ncoef_rect = ncoef_fmap(order)
    if abs(args.debugplot) >= 10:
        print('>>> ttd_order....:', order)
        print('>>> ncoef_rect...:', ncoef_rect)

    # check that polynomial degree in frontiers and spectrails are the same
    poldeg_check_list = []
    for ifile in range(nfiles):
        tmpdict = list_json_longslits[ifile]['contents']
        for islitlet in range(islitlet_min, islitlet_max + 1):
            cslitlet = 'slitlet' + str(islitlet).zfill(2)
            tmppoly = tmpdict[cslitlet]['frontier']['poly_coef_lower']
            poldeg_check_list.append(len(tmppoly) - 1)
            tmppoly = tmpdict[cslitlet]['frontier']['poly_coef_upper']
            poldeg_check_list.append(len(tmppoly) - 1)
            tmppoly = tmpdict[cslitlet]['spectrail']['poly_coef_lower']
            poldeg_check_list.append(len(tmppoly) - 1)
            tmppoly = tmpdict[cslitlet]['spectrail']['poly_coef_middle']
            poldeg_check_list.append(len(tmppoly) - 1)
            tmppoly = tmpdict[cslitlet]['spectrail']['poly_coef_upper']
            poldeg_check_list.append(len(tmppoly) - 1)
    # remove duplicates in list
    poldeg_no_duplicates = list(set(poldeg_check_list))
    if len(poldeg_no_duplicates) != 1:
        print('poldeg_no_duplicates:', order_no_duplicates)
        raise ValueError('poldeg is not constant in frontiers and '
                         'spectrails!')
    poldeg = int(poldeg_no_duplicates[0])
    if abs(args.debugplot) >= 10:
        print('>>> poldeg.......:', poldeg)

    # ---

    # Interpolate rectification polynomial coefficients

    # note: when aij and bij have not been computed, we use the modeled
    # version aij_modeled and bij_modeled
    for islitlet in range(islitlet_min, islitlet_max + 1):
        if abs(args.debugplot) == 0:
            islitlet_progress(islitlet, islitlet_max)
        cslitlet = 'slitlet' + str(islitlet).zfill(2)
        outdict['contents'][cslitlet] = {}
        for keycoef in ['ttd_aij', 'ttd_bij', 'tti_aij', 'tti_bij']:
            list_cc_rect = []
            for icoef in range(ncoef_rect):
                list_csu_bar_slit_center = []
                list_cij = []
                for ifile in range(nfiles):
                    tmpdict = list_json_longslits[ifile]['contents'][cslitlet]
                    csu_bar_slit_center = tmpdict['csu_bar_slit_center']
                    list_csu_bar_slit_center.append(csu_bar_slit_center)
                    cij = tmpdict[keycoef]
                    if cij is not None:
                        list_cij.append(cij[icoef])
                    else:
                        cij_modeled = tmpdict[keycoef + '_modeled']
                        if cij_modeled is None:
                            raise ValueError("Unexpected cij_modeled=None!")
                        else:
                            list_cij.append(cij_modeled[icoef])
                poly, yres, reject = polfit_residuals_with_sigma_rejection(
                    x=np.array(list_csu_bar_slit_center),
                    y=np.array(list_cij),
                    deg=2,
                    times_sigma_reject=5,
                    xlabel='csu_bar_slit_center',
                    ylabel=keycoef + '[' + str(icoef) + ']',
                    title=cslitlet,
                    debugplot=args.debugplot
                )
                list_cc_rect.append(poly.coef.tolist())
            outdict['contents'][cslitlet][keycoef] = {}
            for k, tmpcoef in enumerate(list_cc_rect):
                outdict['contents'][cslitlet][keycoef][k] = tmpcoef

    # ---

    # Interpolate wavelength calibration polynomial coefficients

    # note: when wpoly_refined_coeff have not been computed, we use the
    # wpoly_modeled_coeff
    for islitlet in range(islitlet_min, islitlet_max + 1):
        if abs(args.debugplot) == 0:
            islitlet_progress(islitlet, islitlet_max)
        cslitlet = 'slitlet' + str(islitlet).zfill(2)
        list_cc_rect = []
        for icoef in range(poldeg + 1):
            list_csu_bar_slit_center = []
            list_cij = []
            for ifile in range(nfiles):
                tmpdict = list_json_longslits[ifile]['contents'][cslitlet]
                csu_bar_slit_center = tmpdict['csu_bar_slit_center']
                list_csu_bar_slit_center.append(csu_bar_slit_center)
                cij = tmpdict['wpoly_refined_coeff']
                if cij is not None:
                    list_cij.append(cij[icoef])
                else:
                    cij_modeled = tmpdict['wpoly_modeled_coeff']
                    if cij_modeled is None:
                        raise ValueError("Unexpected cij_modeled=None!")
                    else:
                        list_cij.append(cij_modeled[icoef])
            poly, yres, reject = polfit_residuals_with_sigma_rejection(
                x=np.array(list_csu_bar_slit_center),
                y=np.array(list_cij),
                deg=2,
                times_sigma_reject=5,
                xlabel='csu_bar_slit_center',
                ylabel='wpoly_coef' + '[' + str(icoef) + ']',
                title=cslitlet,
                debugplot=args.debugplot
            )
            list_cc_rect.append(poly.coef.tolist())
        outdict['contents'][cslitlet]['wpoly_coeff'] = {}
        for k, tmpcoef in enumerate(list_cc_rect):
            outdict['contents'][cslitlet]['wpoly_coeff'][k] = tmpcoef

    # ---

    # Save resulting JSON structure
    with open(args.out_MOSlibrary.name, 'w') as fstream:
        json.dump(outdict, fstream, indent=2, sort_keys=True)
        print('>>> Saving file ' + args.out_MOSlibrary.name)


if __name__ == "__main__":
    main()
