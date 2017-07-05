from __future__ import division
from __future__ import print_function

from astropy.io import fits
import json
import numpy as np
import re
from skimage import transform

from emir_slitlet import ArcLinesMap
from emir_slitlet import EmirSlitlet
from emir_definitions import NAXIS1_EMIR
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.rutilities import LinearModelYvsX
from save_ndarray_to_fits import save_ndarray_to_fits

def generate_arc_mapping(source_data_dir, grism, slitlet_number,
                         poly_degree_wfit,
                         order_tt,
                         megadict,
                         debugplot=0):
    """Generate arc line mapping for a partircular slitlet.

    Parameters
    ----------
    source_data_dir : string
        Directory where the input arc line images are stored.
    grism : string
        Single character ("J", "H" or "K") indicating the grism.
    slitlet_number : int
        Slitlet number.
    poly_degree_wfit : int
        Degree for wavelength calibration polynomial.
    order_tt : int
        Order to compute transformation.
    megadict : dictionary of dictionaries
        Structure to store the wavelength calibration mapping.
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

    table_file = "arcs_slit" + str(slitlet_number).zfill(2) + \
                 "_" + grism + "band.txt"
    if debugplot >= 10:
        print(">>> Reading file " + table_file)
    table = np.genfromtxt(table_file,
                          dtype=[('filename', '|S100'), ('pos_pix', '<f8')])
    pos_pix_array = table['pos_pix']
    number_of_slit_positions = len(table['filename'])

    # list to store the 2d slitlet images
    list_slitlet2d = []
    list_slitlet2d_rect = []
    # list to store the EmirSlitlet instances
    list_slt = []
    # numpy array with y0_reference values (middle spectrum trails)
    y0_reference_array = np.zeros(number_of_slit_positions)
    # numpy array with csu_bar_slit_center values
    csu_bar_slit_center_array = np.zeros(number_of_slit_positions)

    # unknown arc lines and determination of middle spectrum trail
    for islitpos in range(number_of_slit_positions):
        # define the FITS file name
        fits_file = table['filename'][islitpos]

        # read the FITS image
        infile = source_data_dir + fits_file
        hdulist = fits.open(infile)
        image_header = hdulist[0].header
        image2d = hdulist[0].data
        naxis2, naxis1 = image2d.shape
        hdulist.close()
        if debugplot >= 10:
            print('>>> Reading file:', infile)
            print('>>> NAXIS1:', naxis1)
            print('>>> NAXIS2:', naxis2)

        grism_in_header = image_header['grism']
        if grism != grism_in_header:
            raise ValueError("GRISM keyword=" + grism_in_header +
                             " is not the expected value=" + grism)
        date_obs = image_header['date-obs']

        # declare slitlet instance
        slt = EmirSlitlet(grism_name=grism,
                          slitlet_number=slitlet_number,
                          fits_file_name=fits_file,
                          date_obs=date_obs,
                          debugplot=debugplot)

        # estimate csu_bar_slit_center from pos_pix values (location
        # of the slits, in pixels, measured in images obtained at the
        # IAC laboratory), using for the linear transformation the
        # coefficients derived from images obtained during the first
        # commissioning of EMIR
        slt.csu_bar_slit_center = 1.23100431 + \
                                  0.16390062 * pos_pix_array[islitpos]

        # store slitlet in list for future access
        list_slt.append(slt)

        # extract slitlet2d
        slitlet2d = slt.extract_slitlet2d(image_2k2k=image2d)
        list_slitlet2d.append(slitlet2d)

        # locate unknown arc lines and determine middle spectrum trail
        slt.locate_unknown_arc_lines(
            slitlet2d=slitlet2d,
            times_sigma_threshold=8,
            delta_x_max=30,
            delta_y_min=37,
            deg_middle_spectrail=2,
            dist_middle_min=15,
            dist_middle_max=28
        )

        # store value of y0_reference for middle spectrum trail
        y0_reference_array[islitpos] = slt.y0_reference[0]
        # store value of csu_bar_slit_center
        csu_bar_slit_center_array[islitpos] = slt.csu_bar_slit_center

    # polynomial fit of y0_reference vs csu_bar_slit_center
    polfit_y0_reference, res_y0_reference_array = polfit_residuals(
        x=csu_bar_slit_center_array, y=y0_reference_array, deg=2,
        xlabel="csu_bar_slit_center (mm)",
        ylabel="y0_reference (pixel)" +
               "\nof middle spectrum trails",
        title=table_file,
        use_r=False,
        debugplot=debugplot
    )

    # correct y0_reference of middle spectrum trail of each slitlet to
    # follow the prediction of the last polynomial fit
    for islitpos in range(number_of_slit_positions):
        slt = list_slt[islitpos]
        slt.y0_reference[0] -= res_y0_reference_array[islitpos]

    # compute wavelength calibration
    list_solutions_wv = []
    for islitpos in range(number_of_slit_positions):
        slt = list_slt[islitpos]
        slitlet2d = list_slitlet2d[islitpos]

        # define vertical offsets to compute additional spectrum
        # trails by shifting the middle one
        ## v_offsets = [17, 8.5, -7, -14]
        v_offsets = [18, -18]
        slt.additional_spectrail_from_middle_spectrail(v_offsets)

        # two iterations:
        # - iteration 0: use unknown arc lines
        # - iteration 1: use known arc lines (from wavelength
        #                calibration derived in iteration 0
        for iteration in range(2):
            # compute intersections of arc lines with all the spectrum
            # trails
            slt.xy_spectrail_arc_intersections(slitlet2d=slitlet2d)

            # use previous intersections to compute the transformation
            # to rectify the slitlet image
            slt.estimate_tt_to_rectify(order=2, slitlet2d=slitlet2d)

            # rectify the slitlet image
            slitlet2d_rect = slt.rectify(slitlet2d=slitlet2d, order=1)
            if iteration == 1:
                list_slitlet2d_rect.append(slitlet2d_rect)
            if True:
                output_file = re.sub(".fits$",
                                     "_warped" + str(iteration) + ".fits",
                                     table['filename'][islitpos])
                print(">>> output file:", output_file)
                save_ndarray_to_fits(slitlet2d_rect, output_file)

            # extract median spectrum and identify location of arc lines
            # peaks
            sp_median, fxpeaks, sxpeaks = \
                slt.median_spectrum_from_rectified_image(
                    slitlet2d_rect,
                    below_middle_spectrail=16,
                    above_middle_spectrail=16,
                    nwinwidth_initial=7,
                    nwinwidth_refined=5,
                    times_sigma_threshold=10,
                    npix_avoid_border=6
                )

            # wavelength calibration of arc lines found in median spectrum
            slt.wavelength_calibration(
                fxpeaks=fxpeaks,
                lamp="argon_xenon",
                crpix1=1.0,
                error_xpos_arc=0.3,
                times_sigma_r=3.0,
                frac_triplets_for_sum=0.50,
                times_sigma_theil_sen=10.0,
                poly_degree_wfit=poly_degree_wfit,
                times_sigma_polfilt=10.0,
                times_sigma_cook=10.0,
                times_sigma_inclusion=10.0,
                weighted=False,
                plot_residuals=True  # TBR
            )

            # overplot wavelength calibration in median spectrum
            if iteration == 1:
                slt.debugplot = 11  # TBR
                slt.overplot_wavelength_calibration(sp_median)
                slt.debugplot = 0  # TBR

            # important: employ the identified arc lines to update
            # slt.list_arc_lines (do not forget this step; otherwise
            # the slt.list_arc_lines will erroneously correspond to
            # the previous arc line list defined prior to the last
            # wavelength calibration)
            slt.debugplot = 11  # TBR
            slt.locate_known_arc_lines(slitlet2d,
                                       below_middle_spectrail=16,
                                       above_middle_spectrail=16,
                                       nwidth=7, nsearch=2)
            slt.debugplot = 0  # TBR

            # ---
            # Check, using R, the necessary global polynomial degree by
            # fitting orthogonal polynomials
            if False:
                myfit = LinearModelYvsX(x=slt.solution_wv.xpos,
                                        y=slt.solution_wv.wv,
                                        degree=poly_degree_wfit+2, raw=False)
                print(">>> Fit with R, using orthogonal polynomials:")
                print(myfit.summary)
                myfit_raw = LinearModelYvsX(x=slt.solution_wv.xpos,
                                            y=slt.solution_wv.wv,
                                            degree=poly_degree_wfit, raw=True)
                print(">>> Fit with R:\n", myfit_raw.coeff_estimate)
                print(">>> Fit with python:\n", slt.solution_wv.coeff)
                pause_debugplot(12)

            print("\n>>> iteration:", iteration)
            print(slt)
            if iteration == 1:
                # seguir aqui
                # seguir aqui
                # coger el diccionario devuelto por la siguiente
                # funcion e insertarlo en un diccionario mas global
                # (tipo megadict, que es un parametro de la funcion en
                # la que nos encontramos ahora mismo)
                tmp_dict = slt.dict_arc_lines_slitlet()
                if not megadict.has_key(grism):
                    megadict[grism] = {}
                slitlet_label = "slitlet"  + str(slitlet_number).zfill(2)
                if not megadict[grism].has_key(slitlet_label):
                    megadict[grism][slitlet_label] = {}
                megadict[grism][slitlet_label][slt.date_obs] = tmp_dict
                # la siguiente linea es independiente de lo anterior
                list_solutions_wv.append(slt.solution_wv)

    # variation of CRVAL1 and CDELT1 with csu_bar_slit_center
    if True:
        print(">>> csu_bar_slit_center, CRVAL1, CRMIN1, CRMAX1, CDELT1:")
        for islitpos in range(number_of_slit_positions):
            slt = list_slt[islitpos]
            print(csu_bar_slit_center_array[islitpos],
                  list_solutions_wv[islitpos].crval1_linear,
                  list_solutions_wv[islitpos].crmin1_linear,
                  list_solutions_wv[islitpos].crmax1_linear,
                  list_solutions_wv[islitpos].cdelt1_linear,
                  (list_solutions_wv[islitpos].crmax1_linear -
                   list_solutions_wv[islitpos].crval1_linear) /
                  slt.cdelt1_enlarged)
        poly_dum, yres_dum = polfit_residuals(
            x=csu_bar_slit_center_array,
            y=np.array([list_solutions_wv[islitpos].crval1_linear for
                        islitpos in range(number_of_slit_positions)]),
            xlabel='csu_bar_slit_center (mm)',
            ylabel='CRVAL1', title=table_file,
            deg=3, debugplot=12)
        poly_dum, yres_dum = polfit_residuals(
            x=csu_bar_slit_center_array,
            y=np.array([list_solutions_wv[islitpos].crmax1_linear for
                        islitpos in range(number_of_slit_positions)]),
            xlabel='csu_bar_slit_center (mm)',
            ylabel='CRMAX1', title=table_file,
            deg=3, debugplot=12)
        poly_dum, yres_dum = polfit_residuals(
            x=csu_bar_slit_center_array,
            y=np.array([list_solutions_wv[islitpos].cdelt1_linear for
                        islitpos in range(number_of_slit_positions)]),
            xlabel='csu_bar_slit_center (mm)',
            ylabel='CDELT1', title=table_file,
            deg=3, debugplot=12)

    # generate arc lines mapping
    json_filename = re.sub(".txt$", "_mapping.json", table_file)
    arc_lines_map = ArcLinesMap(
        list_solutions_wv,
        csu_bar_slit_center_array,
        json_output=json_filename,
        debugplot=12
    )
    print(arc_lines_map)

    # ---
    # Check that the json file containing the arc lines mapping can be
    # used to generate an ArcLinesMap instance with the same mapping
    if False:
        json_filename = re.sub(".txt$", "_mapping.json", table_file)
        arc_lines_map_bis = ArcLinesMap(json_input=json_filename,
                                        debugplot=12)

        for csu_bar_slit_center in csu_bar_slit_center_array:
            predicted_poly_wv = arc_lines_map.predict_poly_wv(
                csu_bar_slit_center=csu_bar_slit_center,
                deg=poly_degree_wfit, times_sigma_reject=5,
                extrapolation=False
            )
            print("Predicted polynomial:\n", predicted_poly_wv)
            predicted_poly_wv = arc_lines_map_bis.predict_poly_wv(
                csu_bar_slit_center=csu_bar_slit_center,
                deg=poly_degree_wfit, times_sigma_reject=5,
                extrapolation=False
            )
            print("Predicted polynomial_bis:\n", predicted_poly_wv)

    # RECTIFY IMAGE WITH A LINEAR SAMPLING IN THE WAVELENGTH DIRECTION
    # compute intersections of arc lines with all the spectrum trails
    for islitpos in range(number_of_slit_positions):
        slt = list_slt[islitpos]
        slitlet2d = list_slitlet2d[islitpos]

        # compute intersections of arc lines with all the spectrum
        # trails
        slt.xy_spectrail_arc_intersections(slitlet2d=slitlet2d,
                                           apply_wv_calibration=True)

        # use previous intersections to compute the transformation to
        # rectify the slitlet image
        slt.estimate_tt_to_rectify(order=order_tt, slitlet2d=slitlet2d)

        # rectify the slitlet image
        slitlet2d_rect = slt.rectify(slitlet2d=slitlet2d, order=1)
        if True:
            output_file = re.sub(".fits$", "_warped2.fits",
                                 table['filename'][islitpos])
            print(">>> output file:", output_file)
            save_ndarray_to_fits(slitlet2d_rect, output_file,
                                 crpix1=1.0,
                                 crval1=slt.crval1_slitlet,
                                 cdelt1=slt.cdelt1_slitlet)

        # optional: apply directly wavelength calibration polynomial
        # to the rectified image prior to the computation of a global
        # transformation that incorporates such calibration
        # (this combination of transformations leads to a rectified
        # and wavelength calibrated slitlet2d which exhibits a larger
        # spectral resolution degradation)
        if True:
            if order_tt != 3:
                raise ValueError("Unexpected order_tt=" + str(order_tt))
            # coefficients of tti
            a00 = (slt.solution_wv.coeff[0]-slt.crval1_slitlet)/\
                  slt.cdelt1_slitlet + slt.crpix1_slitlet
            a10 = slt.solution_wv.coeff[1]/slt.cdelt1_slitlet
            a11 = 0.0
            a20 = slt.solution_wv.coeff[2]/slt.cdelt1_slitlet
            a21 = 0.0
            a22 = 0.0
            a30 = slt.solution_wv.coeff[3]/slt.cdelt1_slitlet
            a31 = 0.0
            a32 = 0.0
            a33 = 0.0
            # compute reverse transformation ttd
            xdum = np.arange(NAXIS1_EMIR)
            fdum = np.polynomial.Polynomial([a00, a10, a20, a30])
            ydum = fdum(xdum)
            poly = np.polynomial.Polynomial.fit(ydum, xdum, deg=3)
            poly = np.polynomial.Polynomial.cast(poly)
            a00 = poly.coef[0]
            a10 = poly.coef[1]
            a20 = poly.coef[2]
            a30 = poly.coef[3]
            slt.ttd = transform.PolynomialTransform(
                np.vstack(
                    [np.array([a00, a10, a11, a20, a21,
                               a22, a30, a31, a32, a33]),
                     np.array([0.0, 0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0])]
                )
            )
            slitlet2d_rect = list_slitlet2d_rect[islitpos]
            slitlet2d_rect_bis = slt.rectify(slitlet2d=slitlet2d_rect, order=1)
            if True:
                output_file = re.sub(".fits$", "_warped3.fits",
                                     table['filename'][islitpos])
                print(">>> output file:", output_file)
                save_ndarray_to_fits(slitlet2d_rect_bis, output_file,
                                     crpix1=1.0,
                                     crval1=slt.crval1_slitlet,
                                     cdelt1=slt.cdelt1_slitlet)

if __name__ == "__main__":

    first_time = False
    grism = 'J'
    if first_time:
        megadict = {}
    else:
        megadict = json.loads(open("megadict_" + grism + ".json").read())

    for slitlet_number in [2]:
        generate_arc_mapping(
            source_data_dir="/Users/cardiel/w/GTC/emir/" + \
                  "20160331_arccalibration_data/GRISM" + grism + "/",
            grism=grism, slitlet_number=slitlet_number, poly_degree_wfit=3,
            order_tt=3, megadict=megadict, debugplot=0)

    # print(json.dumps(megadict, indent=4, sort_keys=True))
    with open("megadict_" + grism + ".json", 'w') as fstream:
        json.dump(megadict, fstream, indent=4, sort_keys=True)


# TODO:
    #
    # - comprobar que podemos leer megadict y calcular los mismos
    #   polinomios de interpolacion que obteniamos antes
    # - salvar informacion en formato json para cada arco
    #   (correspondiente a slitlets individuales) que podamos
    #   calibrar en l.d.o.; con un script independiente podremos luego
    #   leer esa informacion para estimar el middle spectrum trail
    #   (y spectrum trails adicionales) y la calibracion en longitud de
    #   onda de cualquier slitlet.
    # - Sobra ArcLinesMap en EmirSlitlet (lo hemos sustituido por
    #   los scripts generate_megadict y read_megadict).
    # - extraer un slitlet final un poco mejor recortado en la
    #   direccion espacial
    # - determinar que hay que salvar (fichero json) de EmirSlitlet
    #   para poder calibrar en l.d.o. un slitlet que no tenga arco
    #   (puedo usar el propio slitlet de un arco para ver que tal
    #   queda)
    # - representar CRVAL1 y CDELT1 aproximados, calculados en cada
    #   calibracion en l.d.o., y representar estos valores frente a
    #   csu_bar_slit_center (posicion de la rendija deducida de la
    #   posicion de las barras, en mm). Extrapolar para estimar
    #   valor extremos de estos parametros cuando la rendija este
    #   ubicada en los bordes. Determinar el tama~no maximo de una
    #   imagen que pueda albergar cualquiera de los slitlets
    # - eliminar reflejos ???
    # - ver si podemos quitarle el continuo a los espectros para que
    #   quede un poco mejor (quizas usando wavelets?)
    # - elegir nombres adecuados para los ficheros json que almacenan la
    #   calibracion en l.d.o. aproximada
    # - utilizar la mascara de defectos cosmeticos

    # Done:
    # - ver que ocurre cuando alguna linea de arco se ve cortada por
    #   el bounding box del slitlet -> Parece que el metodo es robusto
    # - ver por que parece fallar la rutina de calculo de la distorsion
    #   cuando degree=3 (aunque antes si funcionaba: problema de
    #   redondeo?). En su dia introduje un parametro local llamado
    #   n_step_remove para utilizar solo una fraccion de los puntos en
    #   el calculo de la transformacion de distorsion. Al final parece
    #   claro que es un problema relacionado con tratar de ajustar un
    #   grado demasiado alto en todos los pasos del procedimiento.
    #   Reservamos el grado 3 para la correcion final en la que
    #   incluimos tanto la correcion de distorsion geometrica como la
    #   calibracion en longitud de onda.
