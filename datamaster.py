# -*- coding: utf-8 -*-

# an attempt to unify all the lab scripts into one executable so that we don't need to
# rewrite old code and shit

import sys, getopt

# Allows LaTeX output in Jupyter and in matplotlib
# Importing this because it may be an environmental trigger
from sympy import init_printing
init_printing(use_latex=True, use_unicode=True)

LABS         = ['halflife', 'michelson']

current_labs = {}
selected_lab = None

'''
fetch_lab( name )

Finds a lab analysis script, loads, and returns script
'''
def fetch_lab(name):
    obj = None

    if name not in LABS:
        print('Invalid lab name')
        return

    if name in current_labs:
        obj = current_labs[name]
    else:
        try:
            obj = import ('%s/lab' % (name))
        except Exception as err:
            print('Lab analysis script not found')
            print(str(err))

    return obj

def plot_var(var):
    if selected_lab is not None:
        obj = current_labs[selected_lab]

        try:
            getattr(obj, ('plot_%s' % (var)))()
        except Exception as err:
            print(str(err))
    else:
        print('No selected lab')
        usage()

def get_var(var):
    if selected_lab is not None:
        obj = current_labs[selected_lab]

        try:
            print(getattr(obj, ('get_%s' % (var)))())
        except Exception as err:
            print(str(err))
    else:
        print('No selected lab')
        usage()
        return None

def usage():
    # print a list of commands available with examples
