#!/bin/sh

# read CSUP<number> keywords from FITS header and display CSU bar locations
alias EMIR_display_slitlet_arrangement=\
'python ~/s/xemir/display_slitlet_arrangement.py'

# display distortion map using mathematical expression
alias EMIR_distorsion_emir_spec=\
'python ~/s/xemir/distorsion_emir_spec.py'

# fit boundaries
alias EMIR_fit_boundaries=\
'python ~/s/xemir/fit_boundaries.py'

# overplot boundaries on image
alias EMIR_overplot_bounddict=\
'python ~/s/xemir/overplot_bounddict.py'

# compute slitlet boundaries from continuum lamp exposures
alias EMIR_slitlet_boundaries_from_continuum=\
'python ~/s/xemir/slitlet_boundaries_from_continuum.py'

# compute wavelength calibration polynomials and rectification transformation
# from longslits observed with odd-numbered and even-numbered slitlets
alias EMIR_wpoly_from_longslit=\
'python ~/s/xemir/wpoly_from_longslit.py'

# variation of each coefficient of rectication and wavelength calibration
# transformations as a function of csu_bar_slit_center
alias EMIR_rect_wpoly_for_mos=\
'python ~/s/xemir/rect_wpoly_for_mos.py'

# evaluate rectification and wavelength calibration transformations
alias EMIR_evaluate_rect_wpoly=\
'python ~/s/xemir/evaluate_rect_wpoly.py'

