# -*- coding: utf-8 -*-

# an attempt to unify all the lab scripts into one executable so that we don't need to
# rewrite old code and shit

import signal
import sys
import getopt

# importing matplotlib so datamaster prints all requested plots at once
import matplotlib.pyplot as plt

# Allows LaTeX output in Jupyter and in matplotlib
# Importing this because it may be an environmental trigger
from sympy import init_printing
init_printing(use_latex=True, use_unicode=True)

cli_thread = True

LABS = ['halflife', 'michelson', 'photoelectric', 'franckhertz', 'twoslit']

current_labs = {}
selected_lab = None

def fetch_lab(name, load):
    obj = None

    if name not in LABS:
        print('Invalid lab name')
        return

    if name in current_labs and not load:
        if current_labs[name] is not None:
            print('%s loaded from memory' % (name))
            obj = current_labs[name]

    elif name in current_labs and load:
        try:
            unload_lab(name)

            obj = load_lab(name)
            current_labs[name] = obj
        except Exception as err:
            print('Reload of lab failed')
            print(str(err))
    else:
        obj = load_lab(name)

        if obj is not None:
            current_labs[name] = obj
        else:
            print('Could not load \'%s\'' % (name))

    return obj

def load_lab(name):
    obj = None

    try:
        obj = __import__(name)
        current_labs[name] = obj
    except Exception as err:
        print('Lab analysis could not be loaded')
        print(str(err))
        obj = None

    return obj

def unload_lab(name):
    # this is so dirty I feel so uncomfortable
    rm = []

    global selected_lab

    if str(selected_lab) is name:
        selected_lab = ''

    for mod in sys.modules.keys():
        if mod.startswith('%s.' % (name)):
            rm.append(mod)

    for i in rm:
        del sys.modules[i]

    del sys.modules[name]
    del current_labs[name]

def select_lab(name, load=False):
    lab = fetch_lab(name, load)

    if lab is not None:
        global selected_lab
        selected_lab = name
        print('Selected Lab: %s\n' % (selected_lab))


def plot_var(var):
    if selected_lab is not None:
        obj = current_labs[selected_lab]

        if str(var) == 'all':

            for e in dir(obj.lab):
                if e.startswith('plot_') and callable(getattr(obj.lab, str(e))):
                    getattr(obj.lab, str(e))()

            return True

        try:
            getattr(obj.lab, ('plot_%s' % (var)))()
            return True
        except Exception as err:
            print(str(err))
    else:
        print('No selected lab')
        usage()
        return False


def get_var(var):
    if selected_lab is not None:
        obj = current_labs[selected_lab]

        if str(var) == 'all':

            for e in dir(obj.lab):
                if e.startswith('get_') and callable(getattr(obj.lab, str(e))):
                    print(getattr(obj.lab, str(e))())

            return

        try:
            print(getattr(obj.lab, ('get_%s' % (var)))())
        except Exception as err:
            print(str(err))
    else:
        print('No selected lab')
        usage()


def usage():
    print('Usage: datamaster.py -s <name> [-g, -p] <data name>')
    print('\nCommands:\n  -h, --help: Prints out this help section')
    print('  -l, --list: Lists all the available labs')
    print('  -s, --select <name>: Selects lab to compute data from')
    print('  -r, --reload: Reloads the selected lab from file')
    print('  -p, --plot <variable>: Calls a plotting function of form \"plot_<variable>\"')
    print('  -g, --get <variable>: Prints out a value from function of form \"get_<variable>\"')
    print('  -e, --exit: Explicit command to exit from DataMaster CLI')

def cli():
    legacy = False

    if sys.version_info < (3, 0):
        legacy = True

    while cli_thread:
        if legacy:
            # Py ≤ 2.7
            args = str(raw_input('> ')).split(' ')
            handle_args(args)
        else:
            # Py 3
            args = str(input('> ')).split(' ')
            handle_args(args)

def exit_handle(sig, frame):
    global cli_thread
    cli_thread = False # Safely halts while loop in thread

    print('\nExiting...')
    sys.exit(0)

def handle_args(args):
    try:
        opts, args = getopt.getopt(args, 'hls:rp:g:e', ['help', 'list', 'reload', 'select=', 'plot=', 'get=', 'exit'])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        return

    plotting = False

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            return

        elif opt in ('-l', '--list'):
            print('Available labs:')

            for s in LABS:
                print('  * %s' % (s))

        elif opt in ('-s', '--select'):
            select_lab(arg)

        elif opt in ('-r', '--reload'):
            if selected_lab is not None:
                select_lab(selected_lab, True)
            else:
                print('No selected lab to reload')
                return

        elif opt in ('-p', '--plot'):
            if plot_var(arg):
                plotting = True

        elif opt in ('-g', '--get'):
            get_var(arg)

        elif opt in ('-e', '--exit'):
            print('\nExiting...')
            sys.exit(0)

        else:
            usage()

    if plotting:
        plt.show()

def main(argv):
    if len(argv) == 0:
        # register sigkill event and start looping CLI
        signal.signal(signal.SIGINT, exit_handle)
        cli()

    handle_args(argv)

if __name__ == '__main__':
    main(sys.argv[1:])
