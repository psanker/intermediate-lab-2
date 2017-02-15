# -*- coding: utf-8 -*-

#############################################################
# 1. Imp-orts
#############################################################

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from astropy import units as u
from astropy import constants as const
from sympy import *
from matplotlib import mlab
from matplotlib import patches
import decimal as dec

from scipy.optimize import curve_fit

# Allows LaTeX output in Jupyter and in matplotlib
init_printing(use_latex=True, use_unicode=True)

#############################################################
# 2. Constants
#############################################################

PI     = np.pi
TWO_PI = 2*PI

kB   = const.k_B
R    = const.R
NA   = const.N_A
G    = const.G
g    = const.g0
mu    = 1.317e25 * G
R_E  = 6371 * 1000
h    = const.h
hbar = const.hbar
c    = const.c
m_e  = const.m_e
m_n  = const.m_n
m_p  = const.m_p
R_H  = 2.18e-18 * u.J

#############################################################
# 3. General Functions
#############################################################

def lsq(x, y):
    assert len(x) == len(y), 'Array dimensions do not match'
    n = float(len(x)) # Don't lose precision with int * float multiplication

    # compute covariance matrix and correlation coefficient of data
    cov  = np.cov(x, y)
    varx = cov[0][0]
    vary = cov[1][1]
    sxy  = cov[0][1]
    r    = sxy / (np.sqrt(vary) * np.sqrt(varx))

    # lambda expression for a line
    # dummy parameter array of [1, 1]
    f    = lambda x, *p: p[0]*x + p[1]
    pars = [1, 1]

    pvals, pcov = curve_fit(f, x, y, p0=pars)

    m, b = pvals
    sm   = np.sqrt(pcov[0, 0])
    sb   = np.sqrt(pcov[1, 1])
    sy   = np.sqrt(vary)

    # y = mx + b; r is correlation
    return m, b, sy, sm, sb, r

#############################################################
# 4. Data
#############################################################

wavelength_4358_V_1 = np.array([0, .68, .78, .85, .95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85])
wavelength_4358_d_1 = np.array([-60, 3, 5, 6, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9])

wavelength_4358_V_2 = np.array([0, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
wavelength_4358_d_2 = np.array([-61, -5, 0, 3, 5, 7, 8, 8, 8, 9, 9, 9, 9, 9, 9])

# Appended for complete data set
wavelength_4358_V   = np.append(wavelength_4358_V_1, wavelength_4358_V_2)
wavelength_4358_d   = np.append(wavelength_4358_d_1, wavelength_4358_d_2)

def plot_4358():
    m, b, sy, sm, sb, r = lsq(wavelength_4358_V,wavelength_4358_d)

    x = np.linspace(0, 1.9, 1000)

    plt.figure()
    plt.plot(wavelength_4358_V, wavelength_4358_d, 'r.')
    plt.plot(x, m*x + b, 'b--')
    plt.show()