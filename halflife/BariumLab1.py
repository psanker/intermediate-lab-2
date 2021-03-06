# -*- coding: utf-8 -*-
#############################################################
#
# lab1 analysis
#
# 1. Imports
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
# 4. Lab-Specific Functions
#############################################################

def get_s_distance(val, mu, s):
    return (val - mu) / s

#############################################################
# 5. Plotting Functions
#############################################################
def plot_volts_cps(a, b):
    plt.figure()
    plt.plot(a, b)
    plt.xlabel('Voltage ($V$)')
    plt.ylabel('Counts per Second')
    plt.annotate('Threshold', xy=(370, 75), xytext=(390, 65), arrowprops=dict(facecolor='black', headwidth=6, width=.2, shrink=0.05))

def plot_cpm_counts(arr):
    plt.figure()

    mu = np.mean(arr)
    s  = np.std(arr)

    x = np.linspace(mu - (4*s), mu + (4*s), 1000)
    plt.plot(x, mlab.normpdf(x, mu, s), 'b-', label='No floor')

    plt.plot(x, mlab.normpdf(x, np.floor(mu), np.floor(s)), 'r-', label='Floored')

    plt.xlabel('Counts per min')
    plt.ylabel('Probability density')
    plt.legend(loc='upper right')

def plot_times_counts(t, cs):
    x = np.linspace(0, 4, 1000)

    m, b, sy, sm, sb, r = lsq(t, np.log(cs))

    plt.figure()
    plt.plot(t, np.log(cs), 'r.')
    plt.plot(x, m*x + b, 'b-', label=('$r=%f$' % (r)))

    plt.xlabel('Time (minutes)')
    plt.ylabel('$\log($Counts per 0.5min$)$')
    plt.legend(loc='upper right')

    return m, sm

def plot_thickness_intensity(T, I, xmin, xmax):
    x = np.linspace(xmin, xmax, 1000)

    m, b, sy, sm, sb, r = lsq(T, np.log(I))

    plt.figure()
    plt.plot(T, np.log(I), 'r.')
    plt.plot(x, m*x + b, c=np.random.rand(3,1), label=('$r=%f$' % (r)))

    plt.xlabel('Thickness (mg/cm$^2$)')
    plt.ylabel('$\log($Intensity$)$')
    plt.legend(loc='upper right')

def plot_half_life(lam, dlam):
    mu  = np.log(2) / np.abs(lam)
    s   = (np.log(2) * dlam) / (np.abs(lam))**2

    x = np.linspace(mu-(4*s), mu+(4*s), 1000)

    plt.figure()
    plt.plot(x, mlab.normpdf(x, mu, s), 'b-', label='Computed')

    plt.axvline(2.55, label='Theoretical', ls='--', color='k')
    plt.xlabel('Time (min)')
    plt.legend(loc='upper right')

    print('Half-life: %f±%f' % (mu, s))
    print('Half-life difference from theory: %f' % (get_s_distance(2.55, mu, s)))

def plot_interference(dis, counts, cmb_mu):
    cmbmu = 0.1 * cmb_mu # correction from cpm to cp(0.1m)
    dl    = 0.5

    plt.figure()
    plt.errorbar(dis, counts, xerr=dl, fmt='.', color='r')
    plt.axhline(cmbmu, ls='--', color='k', label='Background range')

    plt.axes().add_patch(
        patches.Rectangle(
            (-10, 0),
            60,
            cmbmu,
            hatch='\\',
            fill=False
        )
    )

    plt.xlabel('Distance ($cm$)')
    plt.ylabel('Counts per 0.1 sec')
    plt.legend(loc='upper right')

#############################################################
# 6. Program Main
#
#    Put all the program execution code below here
#############################################################

# Calibration data; for 'CMB' and Geiger Tube characteristics
hv_test_volts = np.array([315, 335, 355, 375, 395, 415, 435, 455, 475])
hv_test_cps   = np.array([11.8, 71.3, 73.2, 75.1, 76.4, 74.3, 75.0, 75.6, 79.7])
counts_cmb    = np.array([49, 50, 55, 48, 45])

# Barium decay results
ba_times  = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
ba_counts = np.array([163, 138, 137, 114, 103, 88, 86, 80, 67])

# distance vs. source interference
# each dl was ~0.5cm
interference_distance = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]) # cm
interference_counts   = np.array([24, 12, 9, 7, 3, 3, 3, 1, 1, 3])       # counts per 0.1min

times_beta_abs  = np.array([.42, .45, .43, .48, .45, .56, .59, .64, .73, .95, 1.16, 1.85, 3.04]) #in minutes
beta_intensity  = 1./times_beta_abs #inverse minutes

times_gamma_abs = np.array([.62, .66, .72, .65, .7, .61, .61, .61, .54, .64, .57, .62, .58, .64, .65, .72, .77, .89])
gamma_intensity = 1./times_gamma_abs

abs_thicknessGamma = np.array([0, 4.5, 6.5, 14.1, 28.1, 59.1, 102, 129, 161, 206, 258, 328, 419, 516, 849, 1890, 3632, 7435]) #mg/cm^2
abs_thicknessBeta  = abs_thicknessGamma[0:13]

# Print expected CMB cpm
cpm_cmb    = counts_cmb / 2
mu_cpm_cmb = np.mean(cpm_cmb)
s_cpm_cmb  = np.std(cpm_cmb)
# print('CMB Counts/min: %f ± %f' % (mu_cpm_cmb, s_cpm_cmb))
# print('CMB Counts/min: %1.0f ± %1.0f' % (np.floor(mu_cpm_cmb), np.floor(s_cpm_cmb)))

# Ok now to draw some shit
plot_volts_cps(hv_test_volts, hv_test_cps)
plot_cpm_counts(cpm_cmb)

plot_interference(interference_distance, interference_counts, mu_cpm_cmb + s_cpm_cmb)

lam, dlam = plot_times_counts(ba_times, ba_counts - (mu_cpm_cmb / 2.)) # correction for CMB
print('λ: %f±%f Hz' % (-1.*lam, dlam))
plot_half_life(lam, dlam)

plot_thickness_intensity(abs_thicknessBeta, beta_intensity, 0, 450)
plot_thickness_intensity(abs_thicknessGamma[0:11], gamma_intensity[0:11], 0, 360)

#I broke up the gamma into two because the trend doesn't show until the last few data points
plot_thickness_intensity(abs_thicknessGamma[12:], gamma_intensity[12:], 390, 7470)

# All plots at once!
plt.show()

# @end
