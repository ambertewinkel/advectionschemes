"""This file produces the L2 norm of the error over the Courant number for the different schemes, which are in the file call.\
Author: Amber te Winkel
Email: a.j.tewinkel@pgr.reading.ac.uk
Date: 2025-02-17
Call as "py produce_L2_over_C.py 
(inside the code are the settings on what to produce)
"""


import numpy as np
import matplotlib.pyplot as plt
import schemes as sch
import experiments as epm
import analytic as an
import os
import datetime as dati
import logging
import inspect
import main

def plot():
    print('Plotting L2 over C')

    # Cases to analyse
    cases = [\
        {'scheme': 'RK2QC'},
        #{'scheme': 'aiUpwind'}, # already defaults to a theta blend
        #{'scheme': 'aiMPDATA', 'do_beta': 'blend'},
        {'scheme': 'ImExRK', 'RK': 'UJ31e32', 'SD':'fifth302', 'blend':'sm'},
        #{'scheme': 'PPM'},
        ###{'scheme': 'ImExRK', 'RK': 'ARS3233', 'SD':'fifth302', 'blend':'off'},
        ###{'scheme': 'ImExRK', 'RK': 'ARS3233', 'SD':'fifth302', 'blend':'sm'},
        ###{'scheme': 'ImExRK', 'RK': 'ARS3233', 'SD':'fifth302', 'blend':'Ex'},        
        ###{'scheme': 'ImExRK', 'RK': 'ARS3233', 'SD':'fifth302', 'blend':'Im'},
        ##{'scheme': 'SSP3', 'SD':'fourth301'},
        ##{'scheme': 'SSP3', 'SD':'fifth302'},
        ##{'scheme': 'SSP3', 'SD':'fifth401'},
        ##{'scheme': 'ARS3', 'SD':'fourth301'},
        ##{'scheme': 'ARS3', 'SD':'fifth302'},
        ##{'scheme': 'ARS3', 'SD':'fifth401'},
        ##{'scheme': 'UJ3', 'SD':'fourth301'},
        ##{'scheme': 'UJ3', 'SD':'fifth302'},
        ##{'scheme': 'UJ3', 'SD':'fifth401'},
        #{'scheme': 'SSP3QC'},
        #{'scheme': 'SSP3C4'},
        #{'scheme': 'ARS3QC'},
        #{'scheme': 'ARS3C4'},
        #{'scheme': 'UJ3QC'},
        #{'scheme': 'UJ3C4'},
        ]
    
    plot_args = [\
        {'label': 'WKS24', 'color':'magenta', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImEx Upwind', 'color':'darkgreen', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImEx MPDATA', 'color':'orange', 'marker':'x', 'linestyle':'-'},  
        {'label':'AdImEx Strang', 'color':'darkgreen', 'marker':'x', 'linestyle':'-'},
        #{'label':'PPM', 'color':'red', 'marker':'+', 'linestyle':'-'},
        ###{'label':'ImEx ARS3 5th302 noblend', 'color':'green', 'marker':'o', 'linestyle':'-'},
        ###{'label':'ImEx ARS3 5th302 sm', 'color':'orange', 'marker':'x', 'linestyle':'-'},
        ###{'label':'ImEx ARS3 5th302 Ex', 'color':'blue', 'marker':'+', 'linestyle':'-'},
        ###{'label':'ImEx ARS3 5th302 Im', 'color':'purple', 'marker':'+', 'linestyle':'-'}
        ##{'label':'Im SSP3fourth301', 'color':'cyan', 'marker':'x', 'linestyle':'-'},
        ##{'label':'Im SSP3fifth302', 'color':'dodgerblue', 'marker':'+', 'linestyle':'-'},
        ##{'label':'Im SSP3fifth401', 'color':'blue', 'marker':'x', 'linestyle':'-'},
        ##{'label':'Im ARS3fourth301', 'color':'springgreen', 'marker':'x', 'linestyle':'-'},
        ##{'label':'Im ARS3fifth302', 'color':'darkgreen', 'marker':'+', 'linestyle':'-'},
        ##{'label':'Im ARS3fifth401', 'color':'mediumseagreen', 'marker':'x', 'linestyle':'-'},
        ##{'label':'ImEx UJ3fourth301', 'color':'pink', 'marker':'x', 'linestyle':'-'},
        ##{'label':'ImEx UJ3fifth302', 'color':'purple', 'marker':'+', 'linestyle':'-'},
        ##{'label':'ImEx UJ3fifth401', 'color':'magenta', 'marker':'x', 'linestyle':'-'}
        #{'label':'Im SSP3QC', 'color':'cyan', 'marker':'x', 'linestyle':'-'},
        #{'label':'Im SSP3C4', 'color':'green', 'marker':'+', 'linestyle':'-'},
        #{'label':'Im ARS3QC', 'color':'purple', 'marker':'x', 'linestyle':'-'},
        #{'label':'Im ARS3C4', 'color':'orange', 'marker':'+', 'linestyle':'-'},
        #{'label':'ImEx UJ3QC', 'color':'pink', 'marker':'x', 'linestyle':'-'},
        #{'label':'ImEx UJ3C4', 'color':'blue', 'marker':'+', 'linestyle':'-'}
        ]    

    # Setup 
    schemenames = [case["scheme"] for case in cases]
    analytic = an.sine
    u_setting = 'constant'
    dt, nt, u, xmin, xmax = 0.01, 1, 1., 0., 1. 
    c = np.arange(0.075, 3., 0.05) # array of Courant numbers
    dx = u*dt/c # array of grid spacings
    nx = np.array([int((xmax-xmin)/dx_) for dx_ in dx]) # array of number of grid points
    domainsize = nx*dx # array of domain sizes

    # Setup output directory in output/L2_over_C directory/{date}/{schemenames}
    date = dati.date.today().strftime("%Y%m%d")                   # date of the run
    datetime = dati.datetime.now().strftime("%d%m%Y-%H%M%S")      # date and time of the run
    if not os.path.exists('./output/L2_over_C/' + date + '/'):
        os.mkdir('./output/L2_over_C/' + date + '/')
        print("Folder %s created" % date)
    plotname = f'./output/L2_over_C/{date}/{'L2_over_C-' + "-".join(schemenames)}.pdf' 
    filename = f'./output/L2_over_C/{date}/{'L2_over_C-' + "-".join(schemenames)}.log'
    i = 0 
    while os.path.exists(plotname):
        print("Plot file %s already exists" % plotname)
        i += 1
        plotname = f'./output/L2_over_C/{date}/{'L2_over_C-' + "-".join(schemenames)}_{i}.pdf'
        filename = f'./output/L2_over_C/{date}/{'L2_over_C-' + "-".join(schemenames)}_{i}.log'

    # Create L2 arrays for each scheme
    for case in range(len(cases)):
        locals()[f'L2_{plot_args[case]["label"]}'] = np.zeros(len(c))

    # Loop over Courant numbers
    for ic in range(len(c)):
        # Calculate initial condition
        xc = np.zeros(nx[ic])
        for i in range(nx[ic]):
            xc[i] = xmin + (0.5 + i)*dx[ic] # assumes uniform grid
        psi_in = analytic(xc, domainsize[ic], u)   
        # Calculate analytic solution
        psi_an = analytic(xc, domainsize[ic], u, nt*dt)
        # Loop over schemes
        for case in range(len(cases)):
            # Calculate numerical solution
            psi_num = main.callscheme(cases[case], nt, dt, np.full(nx[ic], u), np.full(nx[ic], dx[ic]), psi_in, u_setting=u_setting, verbose=False)[-1]
            # Calculate L2 norm
            locals()[f'L2_{plot_args[case]["label"]}'][ic] = epm.l2norm(psi_num, psi_an, dx[ic])
        
    # Plot L2 over C for each scheme into a single plot
    plt.figure()
    for case in range(len(cases)):
        plt.plot(c, locals()[f'L2_{plot_args[case]["label"]}'], label=plot_args[case]["label"], color=plot_args[case]["color"], marker=plot_args[case]["marker"], linestyle=plot_args[case]["linestyle"])
    plt.xlabel('Courant number')
    plt.ylabel('L2 norm')
    plt.yscale('log')
    plt.legend()
    plt.title(f'L2 norm over C for dt={dt} and u={u}')
    plt.tight_layout()
    plt.savefig(plotname)
    plt.close()

    # Log everything into out.log file
    print(f'See output file {filename}')    
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
    logging.info(f'Date and time: {datetime}')
    logging.info(f'Plot file: {plotname}')
    logging.info('')
    logging.info(f'Analytic function: {analytic.__name__}')
    logging.info(f'Domain size: {xmin} to {xmax} m')
    logging.info(f'Number of time steps: {nt}')
    logging.info(f'Time step: {dt} s')
    logging.info(f'Total simulated time: {nt*dt:.4f} s')
    logging.info(f'Velocity: {u}')     
    logging.info('')
    logging.info(f'Number of grid points: {nx}') #!!! adjust nx along with dx and C
    logging.info(f'Grid spacing: {dx} m')
    logging.info(f'Courant numbers: {c}')
    logging.info(f'Domain sizes: {domainsize} m')
    logging.info('')
    logging.info(f'Schemes included are: {schemenames}')
    logging.info(f'Cases:')
    for case in cases:
        logging.info(case)
    logging.info('')
    logging.info('Default values:')
    for case in cases:
        fn = getattr(sch, f'{case["scheme"]}')
        logging.info(f'{fn.__name__}: {inspect.signature(fn)}')
    logging.info('')
    logging.info('Plotting details:')
    for case in plot_args:
        logging.info(case)
    logging.info('')
    logging.info('')
    logging.info('Results - L2 values for each scheme')
    logging.info('')
    for case in range(len(cases)):
        logging.info(f'Case: {cases[case]}')
        logging.info('L2 values:')
        logging.info(locals()[f'L2_{plot_args[case]["label"]}'])
        logging.info('')
    
    logging.info('========================= Done =========================')


if __name__ == '__main__':
    plot()
    print('Done')

