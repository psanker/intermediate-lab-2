# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

from os import path

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

PI = np.pi
TWO_PI = 2 * PI

kB = const.k_B
R = const.R
NA = const.N_A
G = const.G
g = const.g0
mu = 1.317e25 * G
R_E = 6371 * 1000
h = const.h
hbar = const.hbar
c = const.c
m_e = const.m_e
m_n = const.m_n
m_p = const.m_p
R_H = 2.18e-18 * u.J
q_e = const.e
mu0 = const.mu0

ZERO = 0.00001

#############################################################
# 3. General Functions
#############################################################


def lsq(x, y):
    assert len(x) == len(y), 'Array dimensions do not match'
    n = float(len(x))  # Don't lose precision with int * float multiplication

    # compute covariance matrix and correlation coefficient of data
    cov = np.cov(x, y)
    varx = cov[0][0]
    vary = cov[1][1]
    sxy = cov[0][1]

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
    f = lambda x, *p: p[0] * x + p[1]
    pars = [1, 1]

    pvals, pcov = curve_fit(f, x, y, p0=pars)

    m, b = pvals
    sm = np.sqrt(pcov[0, 0])
    sb = np.sqrt(pcov[1, 1])
    sy = np.sqrt(vary)

    # y = mx + b; r is correlation
    return m, b, sy, sm, sb, r

#############################################################
# 4. Data
#############################################################

res = np.array([16.1, 20.1, 25.1, 30.1, 31.6, 35.6, 40.6, 45.6, 50.6, 55.6, 60.6, 65.6, 70.6, 74.6, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0])
current = np.array([.347, .438, .546, .658, .694, .781, .891, 1.0, 1.109, 1.219, 1.328, 1.436, 1.550, 1.632, 1.612, 1.716, 1.831, 1.933, 2.043, 2.152])
Bfield = 2.115 * current

dW = 1.8
I_modpp = 2*np.sqrt(2)*0.493

#############################################################
# 5. Lab-specific functions
#############################################################


def plot_line():
    '''
    Plots a linear fit for the Magnetic Field and Resonance Frequency Data
    '''
    m, b, sy, sm, sb, r = lsq(Bfield, res)

    x = np.linspace(Bfield[0], Bfield[-1], 1000)

    plt.figure()

    plt.plot(x, m*x + b, 'b--', label='Linear Fit')
    plt.errorbar(Bfield, res, xerr=2.115*.001, fmt='r.', ecolor='k', label='Magnetic Field Data')

    plt.annotate('$y=mx + b$\n$m=$%.3f$\pm$%.3f\n$b=$%.3f$\pm$%.3f\n$r=$%.4f' % (m, sm, b, sb, r), xy=(2.6, 54), xytext=(3.1, 27.4), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))
    plt.xlabel('Magnetic Field ($mT$)')
    plt.ylabel('Resonant Frequency ($MHz$)')
    plt.legend(loc='upper left')


def get_gee():
    '''
    Calculates the value of g, the ratio of the electron’s magnetic moment in units of the Bohr magneton to its angular momentum in units of hbar, using the slope of the linear fit
    '''
    m, b, sy, sm, sb, r = lsq(Bfield, res)

    muB = (q_e.value*hbar.value) / (2*m_e.value)

    slope = m * 10**9 #for unit conversion

    gee = (h.value * slope) / muB

    dgee = (sm / m) * gee

    return gee, dgee


def get_deltaB():
    '''
    Calculates delta B, the full width at half height of the resonance in terms of the magnetic field.
    '''
    deltaB = ((I_modpp*dW) / 10) * 2.115
    sdeltaB = np.sqrt((.2/1.8)**2 + (.001/.493)**2)*deltaB
    return deltaB, sdeltaB
