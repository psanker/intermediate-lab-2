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

# Loading in all runs then concatenating all data to one array set
# HOWEVER, use these to find local maxes
# Note the usage of os.path.abspath(); this resolves any relative path
# errors and keeps track of potential bugs easier. Also, note the presence
# of 'franckhertz/'. Python tends to hoist all its loaded modules into the
# same executing directory as the parent script, so all file references
# must be made with respect to DataMaster's location.

volta_1, voltb_1 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-1.txt'), skiprows=2, unpack=True)
volta_2, voltb_2 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-2.txt'), skiprows=2, unpack=True)
volta_3, voltb_3 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-3.txt'), skiprows=2, unpack=True)
volta_4, voltb_4 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-4.txt'), skiprows=2, unpack=True)
volta_5, voltb_5 = np.loadtxt(path.abspath(
    './franckhertz/30vramp-5.txt'), skiprows=2, unpack=True)

# Do NOT use these to find local maxes
volta = np.concatenate((volta_1, volta_2, volta_3, volta_4, volta_5), axis=0)
voltb = np.concatenate((voltb_1, voltb_2, voltb_3, voltb_4, voltb_5), axis=0)

# Saturated Peak Data

volta_s, voltb_s = np.loadtxt(path.abspath('./franckhertz/saturated-ramp.txt'), skiprows=2, unpack=True)

#############################################################
# 5. Lab-specific functions
#############################################################


def find_peaks(x, y):
    '''
    From two arrays [x] and [y], find the points of local maxima

    Returns an array of coords [ [x_1, y_1], [x_2, y_2], ..., [x_n, y_n] ] which are local maxima

    If no maxima found, return empty set []
    '''

    assert len(x) == len(y), 'Array dimensions must match'

    maxima = []

    for i in range(len(x)):
        # Firstly, check if x[i] is an extreme value of the set
        if i == 0:
            continue # We know that the first index of our data is not a maximum
        elif i == (len(x) - 1):
            break # Do not consider endpoint a maximum, even though it is a boundary

        else:
            if y[i-1] < y[i] and y[i+1] < y[i]:
                maxima.append([x[i], y[i]])

    if len(maxima) == 0:
        return maxima
    else:
        return np.array(maxima)

def get_peaks1():
    '''
    Returns the set of peaks for the first run
    '''

    return find_peaks(volta_1, voltb_1)

def get_peaks2():
    '''
    Returns the set of peaks for the second run
    '''

    return find_peaks(volta_2, voltb_2)

def get_peaks3():
    '''
    Returns the set of peaks for the third run
    '''

    return find_peaks(volta_3, voltb_3)

def get_peaks4():
    '''
    Returns the set of peaks for the fourth run
    '''

    return find_peaks(volta_4, voltb_4)

def get_peaks5():
    '''
    Returns the set of peaks for the fifth run
    '''

    return find_peaks(volta_5, voltb_5)

def average_peaks():
    '''
    Finds the average value of each peak with corrected propagated error

    Returns array of [[x1, y1, sx1, sy1], [x2, y2, sx2, sy2], ...]
    '''
    peaks = np.array([get_peaks1(), get_peaks2(), get_peaks3(), get_peaks4(), get_peaks5()])

    peaks_6 = []
    peaks_7 = []

    # Firstly, sort by which peak arrays have length 6 and 7
    for i in range(len(peaks)):
        if len(peaks[i]) == 6:
            peaks_6.append(peaks[i])
        elif len(peaks[i] == 7):
            peaks_7.append(peaks[i])
        else:
            print('Peak length for index %d is neither 6 nor 7' % (i))

    # Cast as numpy arrays and extract sets of {x, y} data per each index
    peaks_6 = np.array(peaks_6)
    peaks_7 = np.array(peaks_7)

    x   = []
    y   = []

    for i in range(5):
        if i < 2:
            x.append((peaks_7[i]).T[0])
            y.append((peaks_7[i]).T[1])
        else:
            # Shift index back 2, prepend dummy number to make total array have sound dimensions
            x.append(np.insert((peaks_6[i - 2]).T[0], 0, [-99.]))
            y.append(np.insert((peaks_6[i - 2]).T[1], 0, [-99.]))

    x = np.array(x)
    y = np.array(y)

    # Now, find mean and deviation of columns of x and y
    col_x = x.T
    col_y = y.T

    mu_x = []
    s_x  = []
    mu_y = []
    s_y  = []

    for i in range(len(col_x)):

        # Filter out the dummy numbers
        if i == 0:
            xbar = np.mean(col_x[i][:2])
            ybar = np.mean(col_y[i][:2])

            sx   = np.std(col_x[i][:2])
            sy   = np.std(col_y[i][:2])

            mu_x.append(xbar)
            mu_y.append(ybar)

            s_x.append(sx)
            s_y.append(sy)
        else:
            xbar = np.mean(col_x[i])
            ybar = np.mean(col_y[i])

            sx   = np.std(col_x[i])
            sy   = np.std(col_y[i])

            mu_x.append(xbar)
            mu_y.append(ybar)

            s_x.append(sx)
            s_y.append(sy)

    # Cast to NumPy arrays and export
    mu_x = np.array(mu_x)
    mu_y = np.array(mu_y)
    s_x  = np.array(s_x)
    s_y  = np.array(s_y)

    out = []

    for i in range(len(mu_x)):
        out.append([mu_x[i], mu_y[i], s_x[i], s_y[i]])

    return np.array(out)

def get_avgpeaks():
    return average_peaks()

def get_avgsep():
    '''
    Find the average voltage separation between the peak points
    '''

    chanA = average_peaks()[1:].T[0] * 10.
    diffs = []

    for i in range(len(chanA)):
        if i == 0:
            continue
        else:
            diffs.append(chanA[i] - chanA[i - 1])

    return np.array(diffs)

def get_wavelength():
    '''
    Determine the computed energy from the average voltage difference for an electron
    through the voltage.
    '''

    diffs = get_avgsep()

    V     = np.mean(diffs)
    sV    = np.std(diffs)

    lam   = (h.value * c.value) / (q_e.value * V)
    slam  = ((h.value * c.value) / (q_e.value * V**2.)) * sV

    return ('%1.3f ± %1.3f nm' % (lam * 1e9, slam * 1e9))

def plot_peaks():
    '''
    Test plotting function; checks if concatenated data resembles what we measured
    '''

    plt.figure()
    plt.plot(volta*10., voltb, 'r.')

    avg = average_peaks().T # The transpose makes this next line a one-liner
    plt.errorbar(avg[0]*10., avg[1], xerr=avg[2]*10., yerr=avg[3], fmt='go', ecolor='k', label='Average peaks')

    plt.legend(loc='upper left')
    plt.xlabel('Channel A Voltage ($V$)')
    plt.ylabel('Channel B Voltage ($V$)')

def plot_avgsep():
    '''
    Plot calculated separation voltage compared to expected value
    '''

    diffs = get_avgsep()
    mu    = np.mean(diffs)
    s     = np.std(diffs)

    x     = np.linspace(mu - 4*s, mu + 4*s, 1000)

    plt.figure()
    plt.plot(x, mlab.normpdf(x, mu, s), 'b-', label='$%1.3f \pm %1.3f V$' % (mu, s))
    plt.axvline(4.9, ls='--', color='k', label='Expected: $4.9 V$')

    plt.legend(loc='upper left')

def plot_saturated():
    '''
    Plot the saturated voltage data
    '''

    plt.figure()
    plt.plot(volta_s, voltb_s, 'g-.', label='Saturated Signal')
    plt.legend(loc='upper left')
    plt.xlabel('Channel A Voltage ($V$)')
    plt.ylabel('Channel B Voltage ($V$)')
