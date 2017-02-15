# -*- coding: utf-8 -*-

# an attempt to unify all the lab scripts into one executable so that we don't need to
# rewrite old code and shit

import signal
import sys
import getopt
from importlib import import_module

# Allows LaTeX output in Jupyter and in matplotlib
# Importing this because it may be an environmental trigger
from sympy import init_printing
init_printing(use_latex=True, use_unicode=True)

cli_thread = True

LABS = ['halflife', 'michelson', 'photoelectric']

current_labs = {}
selected_lab = None


def fetch_lab(name):
    obj = None

    if name not in LABS:
        print('Invalid lab name')
        return

    if name in current_labs:
        obj = current_labs[name]
    else:
        try:
            obj = __import__(name)
            current_labs[name] = obj
        except Exception as err:
            print('Lab analysis script not found')
            print(str(err))

    return obj


def select_lab(name):
    lab = fetch_lab(name)

    if lab is not None:
        global selected_lab
        selected_lab = name
        print('Selected Lab: %s\n' % (selected_lab))


def plot_var(var):
    if selected_lab is not None:
        obj = current_labs[selected_lab]

        try:
            getattr(obj.lab, ('plot_%s' % (var)))()
        except Exception as err:
            print(str(err))
    else:
        print('No selected lab')
        usage()


def get_var(var):
    if selected_lab is not None:
        obj = current_labs[selected_lab]

        try:
            print(getattr(obj.lab, ('get_%s' % (var)))())
        except Exception as err:
            print(str(err))
    else:
        print('No selected lab')
        usage()


def usage():
    print('Usage: datamaster.py -s <name> [-g, -p] <data name>')
    print('\nCommands:\n\t-h, --help: Prints out this help section')
    print('\t-s, --select <name>: Selects lab to compute data from')
    print('\t-p, --plot <variable>: Calls a plotting function of form \"plot_<variable>\"')
    print('\t-g, --get <variable>: Prints out a value from function of form \"get_<variable>\"')
    print('\t-e, --exit: Explicit command to exit from DataMaster CLI')


def cli():

    while cli_thread:
        args = str(input('> ')).split(' ')
        handle_args(args)

def exit_handle(sig, frame):
    global cli_thread
    cli_thread = False # Safely halts while loop in thread

    print('\nExiting...')
    sys.exit(0)

def handle_args(args):
    try:
        opts, args = getopt.getopt(args, 'hs:p:g:e', ['help', 'select=', 'plot=', 'get=', 'exit'])
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            return
        elif opt in ('-s', '--select'):
            select_lab(arg)
        elif opt in ('-p', '--plot'):
            plot_var(arg)
        elif opt in ('-g', '--get'):
            get_var(arg)
        elif opt in ('-e', '--exit'):
            print('\nExiting...')
            sys.exit(0)
        else:
            usage()

def main(argv):
    if len(argv) == 0:
        # register sigkill event and start looping CLI
        signal.signal(signal.SIGINT, exit_handle)
        cli()

    handle_args(argv)

if __name__ == '__main__':
    main(sys.argv[1:])
