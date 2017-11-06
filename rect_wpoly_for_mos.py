from __future__ import division
from __future__ import print_function

import argparse
import json
import sys

from numina.array.display.fileinfo import list_fileinfo_from_txt
from numina.array.display.pause_debugplot import DEBUGPLOT_CODES

from dtu_configuration import DtuConfiguration


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='rect_wpoly_for_mos')
    # required arguments
    parser.add_argument("input_list",
                        help="TXT file with list JSON files derived from "
                             "longslit data")

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

    # ------------------------------------------------------------------------

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
    dtu_conf_first_longslit = DtuConfiguration()
    dtu_conf_first_longslit.define_from_dictionary(
        json_first_longslit['dtu_configuration'])
    filter_first_longslit = json_first_longslit['tags']['filter']
    grism_first_longslit = json_first_longslit['tags']['grism']
    for ifile in range(1, nfiles):
        json_tmp = json.loads(open(list_json_files[ifile].filename).read())
        dtu_conf_tmp = DtuConfiguration()
        dtu_conf_tmp.define_from_dictionary(json_tmp['dtu_configuration'])
        filter_tmp = json_tmp['tags']['filter']
        grism_tmp = json_tmp['tags']['grism']
        if dtu_conf_first_longslit != dtu_conf_tmp:
            print(dtu_conf_first_longslit)
            print(dtu_conf_tmp)
            raise ValueError("Unexpected different DTU configurations found")
        if filter_first_longslit != filter_tmp:
            print(filter_first_longslit)
            print(filter_tmp)
            raise ValueError("Unexpected different filter found")
        if grism_first_longslit != grism_tmp:
            print(grism_first_longslit)
            print(grism_tmp)
            raise ValueError("Unexpected different grism found")


if __name__ == "__main__":
    main()
