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


#Laser Source

bg_laser = .007 #background voltage

pos_laser = np.array([3.33, 3.37, 3.39, 3.43, 3.49, 3.7, 3.71, 3.78, 3.8, 3.82, 3.84, 3.85, 3.88, 4.42, 5.1, 5.13, 5.15, 5.16, 5.17, 5.19, 5.21, 5.27, 5.46, 5.53, 5.56, 5.59, 5.61, 5.65])
intensity_laser = np.array([.008, .05, .107, .248, .34, .342, .349, .608, .813, 1.018, 1.2, 1.318, 1.471, 1.518, 1.473, 1.372, 1.252, 1.138, 1.016, .836, .654, .462, .465, .386, .259, .139, .053, .008]) - bg_laser

#Bulb Source

#############################################################
# 5. Lab-specific functions
#############################################################

def get_singlemaxleft():
    '''
    Estimates the maximum intensity and uncertainty for the left side single slit pattern

    Returns an array of [Intensity max, Intensity std, Position Max, Position std]
    '''
    li = np.array([.34, .342, .349])
    mu_li = np.mean(li)
    dli = np.std(li)
    lp = np.array([3.49, 3.7, 3.71])
    mu_lp = np.mean(lp)
    dlp = np.std(lp)

    return np.array([mu_li, dli, mu_lp, dlp])

def get_singlemaxright():
    '''
    Estimates the maximum intensity and uncertainty for the right side single slit pattern

    Returns an array of [Intensity max, Intensity std, Position Max, Position std]
    '''
    ri = np.array([.462, .465])
    mu_ri = np.mean(ri)
    dri = np.std(ri)
    rp = np.array([5.27, 5.46])
    mu_rp = np.mean(rp)
    drp = np.std(rp)

    return np.array([mu_ri, dri, mu_rp, drp])

def get_doublemax():
    '''
    Estimates the maximum intensity and uncertainty for the double slit pattern

    Returns an array of [Intensity max, Intensity std, Position, Postion uncertainty]
    '''
    i = np.array([1.509, 1.518])
    mu_i = np.mean(i)
    di = np.std(i)
    pos = 4.42
    dp = .01

    return np.array([mu_i, di, pos, dp])

def plot_laser():
    '''
    Plots the Two Slit Intensity Pattern
    '''
    leftside = get_singlemaxleft()
    rightside = get_singlemaxright()
    double = get_doublemax()
    Intensitymax = np.array([leftside[0], 1.518, rightside[0]])
    Intensityerr = np.array([leftside[1], double[1], rightside[1]])
    Positionmax = np.array([leftside[2], 4.42, rightside[2]])
    Positionerr = np.array([leftside[3], .01, rightside[3]])


    plt.figure()
    plt.plot(pos_laser, intensity_laser, '.b-', label='Two Slit Interference Pattern')
    plt.errorbar(Positionmax, Intensitymax, xerr=Positionerr, yerr=Intensityerr, fmt='ro', ecolor='k', label='Intensity Maximums')

    plt.legend(loc='lower center')
    plt.xlabel('Detector Position ($mm$)')
    plt.ylabel('Intensity ($V$)')
