# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

from scipy import stats
from astropy import units as u
from astropy import constants as const
from matplotlib import mlab
from matplotlib import patches

from scipy.optimize import curve_fit

# Allows LaTeX output in Jupyter and in matplotlib
sp.init_printing(use_latex=True, use_unicode=True)

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

def power_fit(x, y, n):
    assert len(x) == len(y), 'Array dimensions do not match'

    # compute covariance matrix and correlation coefficient of data
    cov  = np.cov(x, y)
    varx = cov[0][0]
    vary = cov[1][1]
    sxy  = cov[0][1]
    r    = sxy / (np.sqrt(vary) * np.sqrt(varx))

    f    = lambda x, *p: p[0]*((x - p[1])**n)
    pars = [1, 1]

    pvals, pcov = curve_fit(f, x, y, p0=pars)

    A, x0 = pvals

    # y = A*x^n + y0; r is correlation
    return A, x0, r

#############################################################
# 4. Data
#############################################################

wavelength_N = 50.0

green_D  = (np.array([.440, .440, .447]) - np.array([.428, .428, .430])) * 1e-3 #meters
red_D    = (np.array([.445, .445, .448]) - np.array([.428, .426, .428])) * 1e-3 #meters
orange_D = (np.array([.442, .446, .441]) - np.array([.427, .428, .423])) * 1e-3 #meters
dist_err = .002 * 1e-3 #meters

L       = .079 #meters
dL      = .001 #meters

green_N = np.array([62.0, 57.0])
red_N   = np.array([50.0, 49.0])
dN      = 1.0

green_lm = 510.0 * 1e-9 #meters
red_lm   = 650.0 *1e-9 #meters

#############################################################
# 5. Lab-Specific Functions
#############################################################

def compute_lambda(arr):
    N   = wavelength_N

    col = (2. * arr) / N

    mu = np.mean(col)
    sl = np.std(col)

    # s1 = np.sqrt(N**(-2.) + (dist_err/np.mean(arr))**2 ) * mu
    # I still don't think this accounts for enough error; this does not include
    # the instability of the air table as the micrometer was being pushed.
    # It would be much more safe to use the deviation as a measurement of uncertainty.

    return mu, sl

def get_wavelength():

    mu_g, s_g = compute_lambda(green_D)
    mu_r, s_r = compute_lambda(red_D)
    mu_o, s_o = compute_lambda(orange_D)

    return ('Green: %1.3e ± %1.3e\nRed: %1.3e ± %1.3e\nOrange: %1.3e ± %1.3e' % (mu_g, s_g, mu_r, s_r, mu_o, s_o))

def find_refraction(arrN, col):
    l, dl = compute_lambda(col)

    N  = np.mean(arrN)
    dN = np.ceil(np.std(arrN)) # Better to overestimate than under in this case

    n  = ((N * l) / (2. * L)) + 1.
    mu = np.mean(n)

    s1 = (l / (2. * L)) * dN
    s2 = ((N * l) / (2. * L**2.)) * dL
    s3 = (N / (2. * L)) * dl

    return mu, np.sqrt(s1**2 + s2**2 + s3**2.)

def get_refraction():
    gn, sgn = find_refraction(green_N, green_D)
    rn, srn = find_refraction(red_N, red_D)

    return('Green: %1.6f ± %1.6f\nRed: %1.6f ± %1.6f\n' % (gn, sgn, rn, srn))
