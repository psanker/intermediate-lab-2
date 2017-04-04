# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

from os import path
import io

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
    sm  = np.sqrt(pcov[0, 0])
    sb  = np.sqrt(pcov[1, 1])
    sy  = np.sqrt(vary)

    # y = mx + b; r is correlation
    return m, b, sy, sm, sb, r

# Calculates the approximate second derivative at an index point with a specified dx
def second_deriv(arr, i, dx):

    if dx <= 0.:
        raise Exception('dx must be greater than 0')

    if len(arr) == (i + 1) or i == 0:
        raise Exception('Cannot perform second derivative at endpoints')

    else:
        return (arr[i + 1] - 2. * arr[i] + arr[i - 1]) / dx


#############################################################
# 4. Data
#############################################################

# All CSV data comes in the form of [theta, counts / s...]
# each column after the first is a single run

f = io.open(path.abspath('./bragg/calibrated.csv'), 'r', encoding='utf-8')
deg, calibrated = np.loadtxt(f, skiprows=1, delimiter=',', usecols=(0, 5), unpack=True)

# NaCl, varying current
f = io.open(path.abspath('./bragg/SaltyCurrent.csv'), encoding='utf-8')
inacl_deg, inacl_c1, inacl_c2, inacl_c3, inacl_c4 = np.loadtxt(f, skiprows=1, delimiter=',', unpack=True)

# NaCl, varying voltage
f = io.open(path.abspath('./bragg/SaltyVoltage.csv'), encoding='utf-8')
vnacl_deg, vnacl_c1, vnacl_c2, vnacl_c3, vnacl_c4, vnacl_c5 = np.loadtxt(f, skiprows=1, delimiter=',', unpack=True)

# Al crystal sample
f = io.open(path.abspath('./bragg/AlCrystal.csv'), encoding='utf-8')
alcry_deg, alcry_counts = np.loadtxt(f, skiprows=1, delimiter=',', unpack=True)

# Al slab sample
f = io.open(path.abspath('./bragg/AlNotCrystal.csv'), encoding='utf-8')
alslab_deg, alslab_counts = np.loadtxt(f, skiprows=1, delimiter=',', unpack=True)


volts = np.array([15, 20, 25, 30, 35]) #keV
currents = np.array([0.4, 0.6, 0.8, 1.0]) #mA

# Lattice constant for NaCl crystal
nacl_a = 0.564e-9


#############################################################
# 5. Lab-specific functions
#############################################################

def find_peaks(x, y):
    '''
    Finds peaks of dataset by combing through potential maxima (using a second derivative test)
    and then picks out the 2 largest maxima, the values representing Ka and Kb
    '''

    if len(x) != len(y):
        raise Exception('Cannot have arrays of differing length')

    if len(x) < 3:
        raise Exception('Not enough data points')

    # Concatenate the two arrays into a matrix and transpose so that each kth index corresponds to a pair of coordinates
    concat = np.array([np.array(x), np.array(y)]).T

    maxima = []
    dx     = concat[1,0] - concat[0,0]

    for k in range(len(concat)):

        # Reject the endpoints
        if k == 0:
            continue
        elif k == len(concat) - 1:
            break

        # Perform second deriv test to filter out minima and saddle points
        sd = (concat[k+1, 1] - 2.*concat[k, 1] + concat[k - 1, 1]) / dx

        if sd < 0:
            if concat[k - 1, 1] < concat[k, 1] and concat[k + 1, 1] < concat[k, 1]:
                maxima.append(concat[k])

    # Now, return the two largest maxima, corresponding to the Ka and Kb points
    filtered_maxima = []

    for i in range(2):
        foo = 0 # dummy index to keep track of index with max value

        for j in range(len(maxima)):
            # Reject endpoints again
            if j == 0:
                continue
            elif j == len(maxima) - 1:
                break

            if maxima[j][1] > maxima[foo][1]:
                foo = j

        filtered_maxima.append(maxima[foo])
        maxima = np.delete(maxima, foo, 0)

    return np.array(filtered_maxima)

def find_cutoff(x, y, limit=0, tolerance=0.1):
    '''
    Find a linear fit which approximately finds the voltage cutoff point
    '''

    work_x = x[limit:]
    work_y = y[limit:]

    assert len(work_x) == len(work_y), 'Array dimensions must match to find cutoff'

    # Before beginning, set all 0 terms to machine zero
    for i in range(len(work_x)):
        if work_x[i] == 0.0:
            work_x[i] = ZERO

        if work_y[i] == 0.0:
            work_y[i] = ZERO

    m, b, sy, sm, sb, r = (0., 0., 0., 0., 0., 0.)
    search = True

    while (search):

        # Generate LSQ linear fit
        m, b, sy, sm, sb, r = lsq(work_x, work_y)

        # Similar to the photoelectric lab, find deviations and normalize
        diffs    = ((m * work_x + b) - work_y)**2 / (work_y**2)
        meandiff = np.sum(np.sqrt(diffs)) / float(len(diffs))

        if meandiff > tolerance:
            work_x = work_x[:-1]
            work_y = work_y[:-1]
        else:
            search = False

    return m, b, sy, sm, sb, r

def cutoff_angle(x, y, lim=[], tol=[]):
    '''
    For n sets of y datasets, compute the collective cutoff point

    x: array
    y: matrix (n arrays)
    limit: array of start limit choices per each y set
    tolerance: array of tolerance choices per each y set
    '''

    assert len(y) == len(lim) and len(y) == len(tol), 'All limit and tolerance values must match y data set length'

    arr_x  = []
    arr_sx = []
    arr_sy = []

    for i in range(len(y)):
        m, b, sy, sm, sb, r = find_cutoff(x, y[i], limit=lim[i], tolerance=tol[i])

        mux   = (-b / m)
        sx2 = ((1. / m) * sy)**2. + ((-1. / m) * sb)**2. + ((b / m**2) * sm)**2.

        arr_x.append(mux)
        arr_sx.append(sx2)
        arr_sy.append(sy**2.)

    arr_x  = np.array(arr_x)
    arr_sx = np.array(arr_sx)
    arr_sy = np.array(arr_sy)

    mu = np.mean(arr_x)
    sx = np.sqrt(np.sum(arr_sx / 4.))
    sy = np.sqrt(np.sum(arr_sy / 4.))

    return mu, sx, sy

def find_salt_angles():
    ic1 = find_peaks(inacl_deg, inacl_c1)
    ic2 = find_peaks(inacl_deg, inacl_c2)
    ic3 = find_peaks(inacl_deg, inacl_c3)
    ic4 = find_peaks(inacl_deg, inacl_c4)

    # Something's fucky with vc1; do not use
    vc2 = find_peaks(vnacl_deg, vnacl_c2)
    vc3 = find_peaks(vnacl_deg, vnacl_c3)
    vc4 = find_peaks(vnacl_deg, vnacl_c4)
    vc5 = find_peaks(vnacl_deg, vnacl_c5)

    concat = np.array([ic1, ic2, ic3, ic4, vc2, vc3, vc4, vc5])

    alpha  = []
    beta   = []

    for i in range(len(concat)):
        alpha.append(concat[i][0][0])
        beta.append(concat[i][1][0])

    alpha = np.array(alpha)
    beta  = np.array(beta)

    mua = np.mean(alpha)
    sda = np.std(alpha)

    mub = np.mean(beta)
    sdb = np.std(beta)

    return mua, sda, mub, sdb

def get_saltangles():
    mua, sda, mub, sdb = find_salt_angles()
    return ('α: %1.3f ± %1.3f°\nβ: %1.3f ± %1.3f°' % (mua, sda, mub, sdb))

def get_currentl():

    t, dt, dy = cutoff_angle(inacl_deg, y=[inacl_c4, inacl_c3], lim=[21, 21], tol=[0.39, 0.3])

    l  = (nacl_a * np.sin((PI / 180.) * t))
    sl = ((PI / 180.) * (nacl_a * np.cos((PI / 180.) * t))) * dt

    return l, sl

def get_h():
    l, sl = get_currentl()

    hm  = ((30e3 * q_e.value) / c.value) * l # hm to not overwrite the constant above
    shm = ((30e3 * q_e.value) / c.value) * sl

    return hm, shm

def plot_saltcurrent():

    plt.figure()
    plt.plot(inacl_deg, inacl_c1, 'b-', label='$0.4 mA$')
    plt.plot(inacl_deg, inacl_c2, 'r-', label='$0.6 mA$')
    plt.plot(inacl_deg, inacl_c3, 'g-', label='$0.8 mA$')
    plt.plot(inacl_deg, inacl_c4, 'm-', label='$1.0 mA$')

    plt.xlabel('Degrees')
    plt.ylabel('Counts / second')
    plt.legend(loc='upper left')

def plot_saltvoltage():

    plt.figure()
    plt.plot(vnacl_deg, vnacl_c1, 'k--', label='$15 keV$', alpha=0.5)
    plt.plot(vnacl_deg, vnacl_c2, 'b-', label='$20 keV$')
    plt.plot(vnacl_deg, vnacl_c3, 'r-', label='$25 keV$')
    plt.plot(vnacl_deg, vnacl_c4, 'g-', label='$30 keV$')
    plt.plot(vnacl_deg, vnacl_c5, 'm-', label='$35 keV$')

    plt.xlabel('Degrees')
    plt.ylabel('Counts / second')
    plt.legend(loc='upper left')

def plot_voltpeaks():

    t = 1 #s
    peak1 = find_peaks(vnacl_deg, vnacl_c1)
    peak2 = find_peaks(vnacl_deg, vnacl_c2)
    peak3 = find_peaks(vnacl_deg, vnacl_c3)
    peak4 = find_peaks(vnacl_deg, vnacl_c4)
    peak5 = find_peaks(vnacl_deg, vnacl_c5)

    alphaPeaks = np.array([peak1[0, 1], peak2[0, 1], peak3[0, 1], peak4[0, 1], peak5[0, 1]])
    alphaerrors = np.sqrt(alphaPeaks / t)

    betaPeaks = np.array([peak1[1, 1], peak2[1, 1], peak3[1, 1], peak4[1, 1], peak5[1, 1]])
    betaerrors = np.sqrt(betaPeaks / t)

    plt.figure()
    plt.errorbar(volts, alphaPeaks, yerr=alphaerrors, fmt='r.', ecolor='k', alpha=0.5, label='$K_{\\alpha}$ Peaks')
    plt.errorbar(volts, betaPeaks, yerr=betaerrors, fmt='b.', ecolor='k', alpha=0.5, label='$K_{\\beta}$ Peaks')


    plt.xlabel('Voltage (keV)')
    plt.ylabel('Counts / second')
    plt.legend(loc='upper left')

def plot_currentpeaks():

    t = 1 #s
    peak1 = find_peaks(inacl_deg, inacl_c1)
    peak2 = find_peaks(inacl_deg, inacl_c2)
    peak3 = find_peaks(inacl_deg, inacl_c3)
    peak4 = find_peaks(inacl_deg, inacl_c4)

    alphaPeaks = np.array([peak1[0, 1], peak2[0, 1], peak3[0, 1], peak4[0, 1]])
    alphaerrors = np.sqrt(alphaPeaks / t)

    betaPeaks = np.array([peak1[1, 1], peak2[1, 1], peak3[1, 1], peak4[1, 1]])
    betaerrors = np.sqrt(betaPeaks / t)

    plt.figure()
    plt.errorbar(currents, alphaPeaks, yerr=alphaerrors, fmt='r.', ecolor='k', alpha=0.5, label='$K_{\\alpha}$ Peaks')
    plt.errorbar(currents, betaPeaks, yerr=betaerrors, fmt='b.', ecolor='k', alpha=0.5, label='$K_{\\beta}$ Peaks')


    plt.xlabel('Current (mA)')
    plt.ylabel('Counts / second')
    plt.legend(loc='upper left')


def plot_al():

    peaks = find_peaks(alcry_deg, alcry_counts)

    plt.figure()
    plt.plot(alcry_deg, alcry_counts, 'b-', label='Al crystal')
    plt.plot(alslab_deg, alslab_counts, 'r-', label='Al slab')

    plt.axvline(peaks[0, 0], color='k', ls='-', label='$K_{\\alpha}$', alpha=0.5)
    plt.axvline(peaks[1, 0], color='k', ls='--', label='$K_{\\beta}$', alpha=0.5)

    plt.xlabel('Degrees')
    plt.ylabel('Counts / second')
    plt.legend(loc='upper left')

def plot_cutoffcurrent():
    plt.figure()
    plt.plot(inacl_deg, inacl_c1, 'k--', alpha=0.3)
    plt.plot(inacl_deg, inacl_c2, 'k--', alpha=0.3)
    plt.plot(inacl_deg, inacl_c3, 'k--', alpha=0.3)
    plt.plot(inacl_deg, inacl_c4, 'k--', alpha=0.3)

    x = np.linspace(3, 6, 1000)
    m1, b1, sy1, sm1, sb1, r1 = find_cutoff(inacl_deg, inacl_c4, limit=21, tolerance=0.39)
    plt.plot(x, m1*x + b1, label=('r: %1.4f' % (r1)))

    m2, b2, sy2, sm2, sb2, r2 = find_cutoff(inacl_deg, inacl_c3, limit=21, tolerance=0.3)
    plt.plot(x, m2*x + b2, label=('r: %1.4f' % (r2)))

    xavg, sxavg, syavg = cutoff_angle(inacl_deg, y=[inacl_c4, inacl_c3], lim=[21, 21], tol=[0.39, 0.3])

    plt.errorbar(xavg, 0, xerr=sxavg, yerr=syavg, fmt='go', ecolor='k', label=(u'%1.3f ± %1.3f°' % (xavg, sxavg)))

    plt.xlabel('Degrees')
    plt.ylabel('Counts / second')
    plt.legend(loc='upper left')
    plt.xlim(xmin=3, xmax=6.5)

def plot_cutoffvoltage():
    plt.figure()
    plt.plot(vnacl_deg, vnacl_c2, 'k--', alpha=0.3)
    plt.plot(vnacl_deg, vnacl_c3, 'k--', alpha=0.3)
    plt.plot(vnacl_deg, vnacl_c4, 'k--', alpha=0.3)
    plt.plot(vnacl_deg, vnacl_c5, 'k--', alpha=0.3)

    x = np.linspace(3, 8, 1000)

    m1, b1, sy1, sm1, sb1, r1 = find_cutoff(vnacl_deg, vnacl_c2, limit=40, tolerance=0.95)
    plt.plot(x, m1*x + b1, label=('r: %1.4f' % (r1)))

    m2, b2, sy2, sm2, sb2, r2 = find_cutoff(vnacl_deg, vnacl_c3, limit=29, tolerance=0.5)
    plt.plot(x, m2*x + b2, label=('r: %1.4f' % (r2)))

    m3, b3, sy3, sm3, sb3, r3 = find_cutoff(vnacl_deg, vnacl_c4, limit=20, tolerance=0.5)
    plt.plot(x, m3*x + b3, label=('r: %1.4f' % (r3)))

    m4, b4, sy4, sm4, sb4, r4 = find_cutoff(vnacl_deg, vnacl_c5, limit=15, tolerance=0.25)
    plt.plot(x, m4*x + b4, label=('r: %1.4f' % (r4)))

    xavg, sxavg, syavg = cutoff_angle(vnacl_deg, [vnacl_c2], lim=[40], tol=[0.95])
    plt.errorbar(xavg, 0, xerr=sxavg, yerr=syavg, fmt='bo', ecolor='k', label=(u'%1.3f ± %1.3f°' % (xavg, sxavg)))

    xavg, sxavg, syavg = cutoff_angle(vnacl_deg, [vnacl_c3], lim=[29], tol=[0.5])
    plt.errorbar(xavg, 0, xerr=sxavg, yerr=syavg, fmt='o', color='#f39131', ecolor='k', label=(u'%1.3f ± %1.3f°' % (xavg, sxavg)))

    xavg, sxavg, syavg = cutoff_angle(vnacl_deg, [vnacl_c4], lim=[20], tol=[0.5])
    plt.errorbar(xavg, 0, xerr=sxavg, yerr=syavg, fmt='go', ecolor='k', label=(u'%1.3f ± %1.3f°' % (xavg, sxavg)))

    xavg, sxavg, syavg = cutoff_angle(vnacl_deg, [vnacl_c5], lim=[15], tol=[0.25])
    plt.errorbar(xavg, 0, xerr=sxavg, yerr=syavg, fmt='ro', ecolor='k', label=(u'%1.3f ± %1.3f°' % (xavg, sxavg)))

    plt.xlabel('Degrees')
    plt.ylabel('Counts / second')
    plt.legend(loc='upper left')
    plt.xlim(xmin=3, xmax=8)

def plot_calibrated():
    peaky = find_peaks(deg, calibrated)
    plt.figure()
    plt.plot(deg, calibrated, 'b-', label='Calibration Data')
    plt.axvline(peaky[0, 0], color='k', ls='-', label='$K_{\\alpha}$', alpha=0.5)
    plt.axvline(peaky[1, 0], color='k', ls='--', label='$K_{\\beta}$', alpha=0.5)

    plt.xlabel('Degrees')
    plt.ylabel('Counts / second')
    plt.legend(loc='upper left')
