# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import matplotlib.pyplot as plt

import decimal as dec
import gc
import math
import os
import shutil
import sys

from astropy import units as u
from astropy import constants as const

from matplotlib import animation
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D

from numba import jit

from scipy.misc import factorial
from scipy import linalg as lin
from scipy import optimize as opt
from scipy import stats

import sympy as sp

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

def progress_meter(per, LEN=50):
    val = per * LEN

    bar = '['

    for i in range(LEN):
        if i <= val:
            bar += '#'
        else:
            bar += '_'

    return (bar + '] %2.0f%%' % (per * 100))

#############################################################
# 4. Data & Globals
#############################################################

data_N = 20
data_m = m_e.value
data_L = 0.1e-3

data_X = np.linspace(0, 2. * data_L, 1000)
data_T = np.linspace(0, 200e-6, 500)

CACHE  = {}

#############################################################
# 5. Lab-Specific Functions
#############################################################

@jit
def coeff(N, L):
    if 'coeff' in CACHE:
        return CACHE['coeff']

    ret   = np.empty(N)
    sqrt2 = np.sqrt(2) / PI

    for n in np.arange(1, N + 1):
        if n == 2:
            foo = PI / 2.
        else:
            foo = ((n - 2.)**(-1.)) * np.sin(((n - 2.) * PI) / 2.)

        bar = ((n + 2.)**(-1.)) * np.sin(((n + 2.) * PI) / 2.)

        ret[n - 1] = sqrt2 * (foo - bar)

    CACHE['coeff'] = ret
    return ret

@jit
def energies(t, N, L, m):
    ret = np.empty(N, dtype=complex)
    hb  = hbar.value

    c   = coeff(N, L)
    val = (2. * m)**(-1.) * (hb * PI / L)**(2.)

    for n in np.arange(1, N + 1):
        E = val * (n**2.)

        re = c[n - 1] * np.cos(-1. * (E / hb) * t)
        im = c[n - 1] * np.sin(-1. * (E / hb) * t)
        ret[n - 1] = re + im*1j

    return ret

@jit
def basis(x, N, L):
    ret = np.empty(N)

    for n in np.arange(1, N + 1):
        ret[n - 1] = np.sqrt(1. / L) * np.sin(((PI * n) / (2. * L)) * x)

    return ret

@jit
def psi_T(t, x, N, L, m):
    key = t

    if key in CACHE:
        return CACHE[key]

    ret = np.empty(len(x), dtype=complex)

    E   = energies(t, N, L, m)

    for i in range(len(x)):
        b      = basis(x[i], N, L)
        psi_x  = np.multiply(c, b)

        ret[i] = np.dot(psi_x, E)

    CACHE[key] = ret
    return ret

@jit
def probability(wav):
    p = np.real(np.multiply(np.conjugate(wav), wav))
    return p / np.sum(p)

# Interactivity below here ---------

# Get the coefficients of the wavefunction
def get_coeff():
    return coeff(data_N, data_L)

# Check the probability normalization condition
def get_probcheck():
    an    = coeff(data_N, data_L)
    spec = np.empty(len(an))

    for i in range(len(an)):
        spec[i] = an[i]**(2.)

    return np.sum(spec)

# Show which eigenstates show up
def plot_probspec():
    an    = coeff(data_N, data_L)
    spec = np.empty(len(an))

    for i in range(len(an)):
        spec[i] = an[i]**(2.)

    plt.figure()

    n = np.arange(1, data_N + 1)
    plt.plot(n, spec, 'ro')

# Plot the expected postion over time
def plot_expvalue():
    X = data_X
    T = data_T

    expect = np.empty(len(T))

    # Firstly, generate the coefficients
    an = coeff(data_N, data_L)

    # Now draw the frames
    print('Building...')
    for i in range(len(T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(T)))))

        # Get wavefunction
        wav  = psi_T(T[i], X, data_N, data_L, data_m)
        prob = probability(wav)

        expect[i] = np.sum(np.multiply(X, prob))

    print('') #dummy spacer
    fig, ax = plt.subplots()

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    ax.plot(T, expect, 'r-')

    plt.xlabel('Time ($s$)')
    plt.ylabel('Position ($m$)')

# Plot the uncertainty in postion over time
def plot_deviation():
    X = data_X
    T = data_T

    dev = np.empty(len(T))

    # Firstly, generate the coefficients
    an = coeff(data_N, data_L)

    # Now draw the frames
    print('Building...')
    for i in range(len(T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(T)))))

        # Get wavefunction
        wav  = psi_T(T[i], X, data_N, data_L, data_m)
        prob = probability(wav)

        exp  = np.sum(np.multiply(X, prob))
        exp2 = np.sum(np.multiply(X**2., prob))

        dev[i] = np.sqrt(exp2 - exp**(2.))

    print('') #dummy spacer
    fig, ax = plt.subplots()

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    ax.plot(T, dev, 'r-')

    plt.xlabel('Time ($s$)')
    plt.ylabel('Uncertainty ($m$)')

# Check the cache size (~ 20KB)
def get_cacheload():
    return '%.0f B' % (float(sys.getsizeof(CACHE)) / 8.)

# Builds the cache of coefficients and wavefunctions
def run_buildcache():
    print('Coefficients..')
    an = coeff(data_N, data_L)

    print('Wavefunctions..')
    X = data_X
    T = data_T

    for i in range(len(T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(T)))))

        # Get wavefunction
        wav = psi_T(T[i], X, data_N, data_L, data_m)

    print('\nCache built.')

# Renders the probability evolving over time
def run_animate():
    # Create cache directory
    directory = 'wavebox'
    cache     = directory + '/dump'

    if not os.path.exists(cache):
        os.makedirs(cache)

    X = data_X
    T = data_T

    # Firstly, build cache if it hasn't already been built
    run_buildcache()

    # Now draw the frames
    print('Frames..')
    for i in range(len(T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(T)))))

        # Get wavefunction
        wav  = psi_T(T[i], X, data_N, data_L, data_m)
        prob = probability(wav)

        # Draw
        fig, ax = plt.subplots()

        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

        ax.plot(X, prob)
        plt.ylim(0, 5e-3)

        plt.xlabel('$x$')
        plt.ylabel('$\\mathcal{P}(x)$')

        plt.savefig('%s/%d.png' % (cache, i))
        plt.clf()
        plt.close()

    print('\nVideo..')
    os.system('ffmpeg -framerate 100 -pattern_type glob -i \'%s/*.png\' -c:v libx264 -pix_fmt yuv420p -preset slower %s/output.mp4' % (cache, directory))

    print('Cleanup..')
    shutil.rmtree(cache)
    plt.close('all') # just in case
    gc.collect()

    print('Done')
