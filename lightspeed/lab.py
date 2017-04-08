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
import decimal as dec

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
q_e  = const.e

ZERO = 0.00001

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

    try:
        # Should help catch FPEs
        if varx == 0.0:
            varx = ZERO
        elif vary == 0.0:
            vary = ZERO

        if (np.sqrt(vary) is not np.sqrt(varx)):
            r = float(sxy) / (np.sqrt(vary) * np.sqrt(varx))
        else:
            r = np.sign(sxy) * 1
    except Exception as err:
        print(str(err))
        r = np.sign(sxy) * 1

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

source_dist = .095 #meters
mirror_dist = .05 #meters
xtra = source_dist - mirror_dist
temp = 20.0 #celsius

air_x = np.array([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]) * 2 * 1e-2 + xtra
air_t = np.array([0.0, 36.0, 96.0, 176.0, 236.0, 316.0, 400.0, 436.0, 536.0, 616.0]) * 1e-12 #seconds
xi = np.ones([4])*56.5 * 1e-2 + xtra
water_xf = np.array([89.0, 87.3, 86.4, 86.8]) * 1e-2 + xtra
poly_xf = np.array([78.0, 77.9, 79.0, 77.9]) * 1e-2 + xtra
glass_xf = np.array([84.0, 83.6, 86.7, 85.3]) * 1e-2 + xtra
unknown_xf = np.array([106.3, 109.9, 107.7, 107.8]) * 1e-2 + xtra
disterr = np.sqrt((.002/(air_x - xtra))**2 + (.002/.095)**2 + (.01/.5)**2) * air_x
timeerr = 5.7 * 1e-12

#############################################################
# 5. Lab-specific functions
#############################################################

def plot_airspeed():
    x = np.linspace(0, 616*1e-12, 1000)

    m, b, sy, sm, sb, r = lsq(air_t, air_x)
    plt.figure()
    plt.plot(air_t, air_x, 'r.', label='Air Data')
    plt.plot(x, m*x+b, 'b-', label='Linear Fit')
    plt.annotate('$y=mx + b$\n$m=$%1.3e$\pm$%1.3e\n$b=$%1.3e$\pm$%1.3e\n$r=$%1.3f' % (m, sm, b, sb, r), xy=(2.9e-10, 2.15), xytext=(3.1e-10, 2.08), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))
    plt.xlabel('Time Delay (s)')
    plt.ylabel('Path Length (m)')
    plt.legend(loc='upper left')
