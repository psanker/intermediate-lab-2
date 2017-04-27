# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import sys, getopt

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

green_D  = (np.array([.440, .440, .447]) - np.array([.428, .428, .430])) * 10e-3 #meters
red_D    = (np.array([.445, .445, .448]) - np.array([.428, .426, .428])) * 10e-3 #meters
orange_D = (np.array([.442, .446, .441]) - np.array([.427, .428, .423])) * 10e-3 #meters
dist_err = .002 * 10e-3 #meters

L = .79 #meters
dL = .01 #meters
green_N = np.array([62.0, 57.0])
red_N = np.array([50.0, 49.0])
dN = 1.0
green_lm = 510.0 * 10e-9 #meters
red_lm = 650.0 *10e-9 #meters



#############################################################
# 5. Lab-Specific Functions
#############################################################
def get_wavelength():
    greens = 2*green_D / wavelength_N
    g_mu = np.mean(greens)
    sg = np.sqrt( (1./50.0)**2 + (dist_err/np.mean(green_D))**2 )*g_mu
    reds = 2*red_D / wavelength_N
    r_mu = np.mean(reds)
    sr = np.sqrt( (1./50.0)**2 + (dist_err/np.mean(red_D))**2 )*r_mu
    oranges = 2*orange_D / wavelength_N
    o_mu = np.mean(oranges)
    so = np.sqrt( (1./50.0)**2 + (dist_err/np.mean(orange_D))**2 )*o_mu
    return ('Green: %1.3e ± %1.3e\nRed: %1.3e ± %1.3e\nOrange: %1.3e ± %1.3e' % (g_mu, sg, r_mu, sr, o_mu, so))

def find_refraction(N, lm):
    n = (N*lm / (2*L)) + 1
    return n

def get_refraction():
    gn = np.mean(find_refraction(green_N, green_lm))
    sgn = np.sqrt( (dL/L)**2 + (dN/np.mean(green_N))**2 )*gn
    rn = np.mean(find_refraction(red_N, red_lm))
    srn = np.sqrt( (dL/L)**2 + (dN/np.mean(red_N))**2 )*rn
    return('Green: %1.3f ± %1.3f\nRed: %1.3f ± %1.3f\n' % (gn, sgn, rn, srn))
