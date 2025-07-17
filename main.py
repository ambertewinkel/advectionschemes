# Main code with various numerical schemes to solve the advection equation
# Author:   Amber te Winkel
# Email:    a.j.tewinkel@pgr.reading.ac.uk


import numpy as np
import matplotlib.pyplot as plt
import schemes as sch
import experiments as epm
import utils as ut
import analytic as an
import grid as gr
import animation as anim
import os
import datetime as dati
import logging
import inspect
import timeit


logger = logging.getLogger(__name__)


def main():
    """
    This function computes and plots the results of various numerical schemes 
    with 1D periodic space and time. Results are compared to the analytic soln. 
    Two initial conditions are considered: a Gaussian distribution and a step 
    function, both defined on a subdomain. 
    Schemes included: FTBS, FTFS, FTCS, CTBS, CTFS, CTCS, Upwind, BTBS, BTFS, BTCS, CNBS, MPDATA, three hybrid schemes and Jacobi and Gauss-Seidel iterations (and more, see schemes.py).
    """

    #############################
    #### Input and testcases ####
    #############################

    # Test or save output in name-specified folder
    save_as = 'store'             # 'test' or 'store'; determines how the output is saved
    
    # Input booleans
    limitCto1 = False
    create_animation = True
    check_orderofconvergence = False # Be careful with this setting with the varying velocity field in space (and time). Has not been adjusted to work with this yet, if that is necessary (probably not necessary for the varying in space and time as you only compare after a ful revolution in time; probably necessary for the varying in space as we can compare the behaviour after 1/2/4 time steps as it is independent of time).
    accuracy_in = 'space with C const' # 'space with dt const' or 'time with dx const' or 'space with C const'; (relevant only if check_orderofconvergence == True)
    date = dati.date.today().strftime("%Y%m%d")                   # date of the run
    datetime = dati.datetime.now().strftime("%d%m%Y-%H%M%S")      # date and time of the run

    # Input cases
    cases = [\
        #{'scheme':'RK2QC'},
        #{'scheme': 'aiUpwind'},
        {'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm'},#, 'output_substages':True},
        #{'scheme': 'ImExRK', 'RK':'aiUpwind', 'SD':'BS', 'blend':'sm'},#, 'output_substages':True},
        #{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':1},#, 'output_substages':True},
        #{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':2},#, 'output_substages':True},
        #{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':3},#, 'output_substages':True},
        #{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':4},#, 'output_substages':True},
        #!{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':5},#, 'output_substages':True},
        #{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':6},#, 'output_substages':True},
        ####{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':7},#, 'output_substages':True},
        ####{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':8},#, 'output_substages':True},
        ####{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':9},#, 'output_substages':True},
        ####{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'iterFCT':True, 'nIter':10},#, 'output_substages':True},
        ####{'scheme': 'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'FCT':True},#, 'output_substages':True},
        #{'scheme':'PPM'},
        #{'scheme':'PPM', 'iterFCT':True, 'nIter':1},
        ##{'scheme':'PPM', 'iterFCT':True, 'nIter':2},
        ##{'scheme':'PPM', 'iterFCT':True, 'nIter':3},
        #{'scheme':'ImExRK', 'RK':'UJ31e32', 'SD':'fifth302', 'blend':'sm', 'FCT_HW':True},#, 'output_substages':True},
        ]
    
    plot_args = [\
        #{'label':'WKS24', 'color':'magenta', 'marker':'+', 'linestyle':'-'},
        #{'label':'aiUpwind', 'color':'red', 'marker':'x', 'linestyle':'-'},
        {'label':'AdImEx Strang', 'color':'darkgreen', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImEx Upwind', 'color':'cyan', 'marker':'', 'linestyle':':'},
        #{'label':'AdImEx Strang FCT', 'color':'darkorange', 'marker':'x', 'linestyle':'--'},
        #{'label':'AdImEx Strang FCT2', 'color':'magenta', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImEx Strang FCT3', 'color':'navy', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImEx Strang FCT4', 'color':'purple', 'marker':'x', 'linestyle':'-'},
        #!{'label':'AdImEx Strang FCT5', 'color':'brown', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImEx Strang FCT6', 'color':'crimson', 'marker':'x', 'linestyle':'-'},
        ####{'label':'AdImEx Strang FCT7', 'color':'darkgreen', 'marker':'x', 'linestyle':'-'},
        ####{'label':'AdImEx Strang FCT8', 'color':'darkviolet', 'marker':'x', 'linestyle':'-'},
        ####{'label':'AdImEx Strang FCT9', 'color':'gold', 'marker':'x', 'linestyle':'-'},
        ####{'label':'AdImEx Strang FCT10', 'color':'darkred', 'marker':'x', 'linestyle':'-'},
        ####{'label':'AdImEx Strang FCT', 'color':'blue', 'marker':'x', 'linestyle':':'},
        #{'label':'PPM', 'color':'blue', 'marker':'o', 'linestyle':'-'},
        #{'label':'PPM FCT1', 'color':'red', 'marker':'o', 'linestyle':'-'},
        ##{'label':'PPM FCT2', 'color':'green', 'marker':'o', 'linestyle':'-'},
        ##{'label':'PPM FCT3', 'color':'purple', 'marker':'o', 'linestyle':'-'},
        #{'label':'AdImEx Strang FCT_HW1', 'color':'darkblue', 'marker':'x', 'linestyle':'-'},
        ]

    # Initial conditions
    ymin, ymax = 0., 30.#8., 13.#-0.1, 1.1#1.1#8., 13.#0., 30.#2.#30.         # for plotting purposes (animation)
    nx = 40                     # number of points in space
    xmax = 1.                   # physical domain parameters
    nt = 1#16#32#100#10#50                     # number of time steps # needs to be 1 when output_substages is True for ImExRK scheme
    dt = 0.01                   # time step
    coords = 'uniform'          # 'uniform' or 'stretching' # note: stretching won't work with a varying velocity field
    schemenames = [case["scheme"] for case in cases]
    analytic = an.sine_yshift#analytic_constant # initial condition, options: sine, cosbell, tophat, or combi, halfwave, revhalfwave, and more for varying velocity field
    u_setting = 'varying_space5' # 'constant' or various 'varying_space..' options
    time1rev = False            # This boolean is set by hand - determines whether, for a varying velocity field in space and time, the u ~ cos(wt) has gone through a full revolution in time (and space?). It determines whether the analytic solution is plotted for a certain number of time steps or not. # Note: This is currently (21-04-2025) only applied to the .pdf final field output, not to the animation .gif file.
    if u_setting == 'constant':
        uconstant = 6.25#3.125#1.        # constant velocity # should only apply when u_setting == 'constant' # is used in the analytic function and for the title in the final.pdf plot for the constant velocity field
        schemenames_settings = str(analytic.__name__) + f'_t{nt*dt:.4f}_u{uconstant}_' + "-".join(schemenames)
    else:
        schemenames_settings = str(analytic.__name__) + f'_t{nt*dt:.4f}_u{u_setting}_' + "-".join(schemenames)
    
    ##################################
    #### Setup output and logging ####
    ##################################

    # Setup output directory
    # Check if ./output/ and outputdir exist, if not create them
    if not os.path.exists('./output/'):
        os.mkdir('./output/')
        print("./output/ created")
    # Determine where to save the output
    if save_as == 'test':
        outputdir = './output/test/'
        filename = outputdir + 'out.log'
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
            print("Folder %s created" % outputdir)
        else:
            filename = outputdir + 'out.log'
            os.remove(filename)
    elif save_as == 'store':
        if not os.path.exists('./output/dated/' + date + '/'):
            os.mkdir('./output/dated/' + date + '/')
            print("Folder %s created" % date)
        outputdir = f'./output/dated/{date}/{schemenames_settings}/' 
        i = 0 
        while os.path.exists(outputdir):
            print("Folder %s already exists" % outputdir)
            i += 1
            outputdir = f'./output/dated/{date}/{schemenames_settings}_{i}/'
        os.mkdir(outputdir)
        print("Folder %s created" % outputdir)
        filename = outputdir + 'out.log'
    plotname = outputdir + 'final.pdf'

    # Set up logging
    print(f'See output file {filename}')    
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
    logging.info(f'Date and time: {datetime}')
    logging.info(f'Output directory: {outputdir}')
    logging.info('')
    logging.info(f'Analytic function: {analytic.__name__}')
    if u_setting == 'constant':
        logging.info(f'Velocity function: {u_setting} with u={uconstant}')
    else:
        logging.info(f'Velocity function: {u_setting}')
    logging.info(f'Number of grid points: {nx}')
    logging.info(f'Number of time steps: {nt}')
    logging.info(f'Time step: {dt} s')
    logging.info(f'Total simulated time: {nt*dt:.4f} s')
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

    #####################
    #### Run schemes ####
    #####################

    # Setup: run schemes for one or three grid spacings (nx*factor, nx/factor, nx)
    if check_orderofconvergence == True: # Run schemes for two extra grid spacings # v!!! change/check how we can get this to work with varying u fields in space (and time?) - currently not implemented - could be implemented in the future, see comment where check_orderofconvergence is set
        factor = 2
        gridlabels = ['fine', 'coarse', 'reg']
        if accuracy_in == 'space with dt const': # 'space with dt const' or 'time with dx const' or 'space with C const'
            nx_arr = np.array([nx*factor, nx/factor, nx], dtype=int)
            dx_arr = np.array([xmax/nx_arr[0], xmax/nx_arr[1], xmax/nx_arr[2]], dtype=float)
            nt_arr = np.full(len(nx_arr), nt)
            dt_arr = np.full(len(nx_arr), dt)
            c_arr = np.array([uconstant*dt_arr[0]/dx_arr[0], uconstant*dt_arr[1]/dx_arr[1], uconstant*dt_arr[2]/dx_arr[2]], dtype=float)
            resolution = dx_arr.copy()
            print('Courant numbers for the [fine, coarse, reg] grid spacings:', c_arr)
            var_acc = f'dx with dt={dt_arr[-1]:.4f}'
        elif accuracy_in == 'time with dx const':
            nt_arr = np.array([nt*factor, nt/factor, nt], dtype=int)
            dt_arr = np.array([dt/factor, dt*factor, dt], dtype=float)
            nx_arr = np.full(len(nt_arr), nx)
            dx_arr = np.full(len(nt_arr), xmax/nx)
            c_arr = np.array([uconstant*dt_arr[0]/dx_arr[0], uconstant*dt_arr[1]/dx_arr[1], uconstant*dt_arr[2]/dx_arr[2]], dtype=float)
            resolution = dt_arr.copy()
            print('Courant numbers for the [fine, coarse, reg] grid spacings:', c_arr)
            var_acc = f'dt with dx={dx_arr[-1]:.4f}'
        elif accuracy_in == 'space with C const': # dx and dt vary keeping Courant number constant
            nx_arr = np.array([nx*factor, nx/factor, nx], dtype=int)
            dx_arr = np.array([xmax/nx_arr[0], xmax/nx_arr[1], xmax/nx_arr[2]], dtype=float)
            nt_arr = np.array([nt*factor, nt/factor, nt], dtype=int)
            dt_arr = np.array([dt/factor, dt*factor, dt], dtype=float)
            c_arr = np.array([uconstant*dt_arr[0]/dx_arr[0], uconstant*dt_arr[1]/dx_arr[1], uconstant*dt_arr[2]/dx_arr[2]], dtype=float)
            resolution = dx_arr.copy()
            for i in range(len(nx_arr)): # Loop over 1 or 3 grid spacings
                logging.info(f'Courant number for the {gridlabels[i]} grid spacing: {c_arr[i]:.4f}')
                logging.info(f'nt, dt, nt*dt for the {gridlabels[i]} grid spacing: {nt_arr[i]}, {dt_arr[i]:.4f}, {nt_arr[i]*dt_arr[i]:.4f}')
                logging.info(f'nx, dx, xmax for the {gridlabels[i]} grid spacing: {nx_arr[i]}, {dx_arr[i]:.4f}, {xmax}')
                logging.info('')
            var_acc = f'dx with C={c_arr[-1]:.4f}'
        else:
            logging.info('Error: invalid accuracy_in')
    else: # Run schemes for only the one grid spacing defined above
        nx_arr = np.array([nx], dtype=int)
        dt_arr = np.array([dt], dtype=float) # change this? !!!
        nt_arr = np.array([nt], dtype=int)
        gridlabels = ['reg']

    # Calculate numerical results
    for xi in range(len(nx_arr)): # Loop over 1 or 3 grid spacings
        nx = nx_arr[xi]
        dt = dt_arr[xi]
        nt = nt_arr[xi]
            
        # Setup grid for each of the grid spacings
        if coords == 'stretching': # Potentially adjusted later if limitCto1 == True
            if u_setting == 'varying_space' or u_setting == 'varying_space_time':
                print('Error: stretching coordinates not implemented for varying velocity field')
                logging.info('Error: stretching coordinates not implemented for varying velocity field')
                raise ValueError('Error: stretching coordinates not implemented for varying velocity field')
            xf, dxc, xc, dxf = gr.coords_stretching(xmax, nx, nx/2, dxcmin=0.) # points in space, length of spatial step        
        elif coords == 'uniform':
            xf, dxc, xc, dxf = gr.coords_uniform(xmax, nx) # points in space, length of spatial step
        elif coords == 'weller':
            xf, dxc, xc, dxf = gr.coords_welleretal2022(xmax, nx) # points in space, length of spatial step
        else: 
            logging.info('Error: invalid coordinates')

        if u_setting == 'constant':
            uf = np.full(nx, uconstant)
        elif u_setting == 'varying_space':
            uf = an.velocity_varying_space(xf)
        elif u_setting == 'varying_space2':
            uf = an.velocity_varying_space2(xf)
        elif u_setting == 'varying_space3':
            uf = an.velocity_varying_space3(xf)     
        elif u_setting == 'varying_space4':
            uf = an.velocity_varying_space4(xf) 
        elif u_setting == 'varying_space5':
            uf = an.velocity_varying_space5(xf)
        elif u_setting == 'varying_space6':
            uf = an.velocity_varying_space6(xf)
        else:
            logging.info('Error: invalid velocity setting')
            print('Error: invalid velocity setting')

        l = gridlabels[xi]

        # Check whether to limit the Courant number by limiting the grid spacing
        if limitCto1 == True: 
            cmax = 1.
            dxcmin = np.min(0.5*dt*(np.roll(uf,-1) + uf)/cmax)
        else:
            dxcmin = 0.

        # Reset with potentially the new dxcmin (i.e. connected to limitCby1)
        if coords == 'stretching':
            xf, dxc, xc, dxf = gr.coords_stretching(xmax, nx, nx/2, dxcmin=dxcmin) # points in space, length of spatial step

        # Calculate velocity and Courant number at cell centers  # v!!! check if this is correct! 21-04-2025: somehow figured out a way around it. I think it is correct now but double check whether I want to improve the code though as it might be a bit hacky.
        # v!!! means that I have found a (temporary) solution on 21-04-2025 - but could use a think of a better solution in the future.
        if u_setting == 'constant': # !!! dt will not always be the same for all grids
            uc = gr.linear(xc, xf, uf)       # velocity at cell centers
            cc = 0.5*dt*(np.roll(abs(uf),-1) + abs(uf))/dxc # Courant number at cell centers # This C is not actually used directly in the scheme nor analytic solution - it is recalculated in those functions (and sometimes also at cell faces instead)! It is only used to calculate the cmax/cmin below. 
            cmax = np.max(cc)
            cmin = np.min(cc)
            logging.info(f'Min Courant number: {cmin:.4f}')
            logging.info(f'Max Courant number: {cmax:.4f}')  
        if u_setting == 'varying_space' or u_setting == 'varying_space_time' or u_setting == 'varying_space2' or u_setting == 'varying_space3' or u_setting == 'varying_space4' or u_setting == 'varying_space5' or u_setting == 'varying_space6':
            # FYI - for the nonuniform velocity schemes I need to calculate the Courant number at cell faces
            logging.info('The Courant numbers values and plot wont be exactly correct for varying velocity, as the ones I would calculate here are defined at cell centers -> hence I have set them to be NaNs.')
            uc = np.full(nx, np.nan)
            cc = np.full(nx, np.nan)

        # Print and plot grid and Courant number (solely for the regular grid spacing) # v!!! remove feature or move it to cell faces (varying velocity)? -> 21-04-2025: I have simply set cc to be NaN for the varying velocity field in space and time, so that it doesn't plot anything I think. I wouldn't want to remove the actual plotting action as it wouldn't update the plot in testing mode otherwise.
        if gridlabels[xi] == 'reg':
            logging.info('The (cell center) points and Courant numbers are:')
            for i in range(nx):
                logging.info(f'{i}: {xc[i]:.4f} -- {cc[i]:.4f}')
            logging.info('')
            ut.plot_Courant(xc, cc, outputdir)
            ut.plot_grid(xc, dxc, outputdir)

        # Calculate analytic solutions for each time step
        locals()[f'psi_an_{l}'] = np.zeros((nt+1, nx))
        for it in range(nt+1):
            locals()[f'psi_an_{l}'][it] = analytic(xc, xmax, uc, it*dt) # analytic solution uses uc. For the varying velocity fields, I can pass on uc but the analytic solution function doesn't actually need it as the velocity is basically already prescribed in the equation it calculates, that is, if an analytic solution exists.
        a = locals()[f'psi_an_{l}'][-1].copy()
        if u_setting == 'varying_space' or u_setting == 'varying_space2' or u_setting == 'varying_space3' or u_setting == 'varying_space4' or u_setting == 'varying_space5' or u_setting == 'varying_space6':
            logging.info("NOTE: the analytic solution is only sensible for a variable velocity field in space for a couple of time steps into the simulation due to accummulation of the field.")
        elif u_setting == 'varying_space_time':
            logging.info("NOTE: the analytic solution is only sensible for a variable velocity field in space and time after a full revolution in time.")
        logging.info(f"Analytic solution for nx={nx}, nt={nt}, dt={dt}: {a}")
        logging.info('')

        # Calculate initial condition
        psi_in = analytic(xc, xmax, 0., 0.)#locals()[f'psi_an_{l}'][0] # this adjustment allows the initial condition to be nonzero for the velocity varying space settings. 
        
        # Calculate numerical solutions for each scheme through time
        # Output is 2D field ([1d time, 1d space])
        for c in range(len(cases)):
            s = plot_args[c]['label']
            locals()[f'psi_{s}_{l}'] = callscheme(cases[c], nt, dt, uf, dxc, psi_in, u_setting)

    ##########################
    #### Plotting schemes ####
    ##########################
    
    plt.figure(figsize=(7,4))
    # Plotting the final time step for each scheme in the same plot
    if u_setting != 'varying_space_time' or ( u_setting == 'varying_space_time' and time1rev == True ): # if it is varying_space_time, the analytic solution is only valid for a full revolution in time
        plt.plot(xc, locals()['psi_an_reg'][nt], label='Analytic', linestyle='-', color='k')

    for c in range(len(cases)):        
        s = plot_args[c]['label']
        plt.plot(xc, locals()[f'psi_{s}_reg'][nt], **plot_args[c])
    if u_setting == 'constant':
        cconstant = uconstant*dt/(xmax/nx)  # Courant number # only used for title in final.pdf
        ut.design_figure(plotname, f'$\\Psi$ at t={nt*dt} with C={cconstant}', \
                     'x', '$\\Psi$', 0., xmax, True, -0.1, 1.1)
    elif u_setting == 'varying_space':
        ut.design_figure(plotname, f'$\\Psi$ at t={nt*dt} with $u$ varying in space', \
                     'x', '$\\Psi$', 0., xmax, True, 0.1, ymax)
    elif u_setting == 'varying_space_time':
        ut.design_figure(plotname, f'$\\Psi$ at t={nt*dt} with $u$ varying in space and time', \
                     'x', '$\\Psi$', 0., xmax, True, -0.1, 1.1)
    elif u_setting == 'varying_space2' or u_setting == 'varying_space3' or u_setting == 'varying_space4' or u_setting == 'varying_space5' or u_setting == 'varying_space6':
        ut.design_figure(plotname, f'$\\Psi$ at t={nt*dt} with $u$ {u_setting}', \
                     'x', '$\\Psi$', 0., xmax, True, ymin, ymax)

    #####################
    #### Experiments ####
    #####################

    # Print experiment results in .log
    logging.info('')
    logging.info('========== Data at the final time step ==========')
    logging.info('')
    
    # Conservation, boundedness and total variation Psi
    csv_psi_analytic = epm.check_conservation(psi_in, locals()['psi_an_reg'][nt], dxc)
    logging.info(f'Analytic - Mass gained: {csv_psi_analytic:.4E}')    
    bdn_psi_analytic = epm.check_boundedness(psi_in, locals()['psi_an_reg'][nt])
    logging.info(f'Analytic - Boundedness: {bdn_psi_analytic}')    
    logging.info('')
    for c in range(len(cases)):        
        s = plot_args[c]['label']
        locals()[f'csv_psi_{s}'] = epm.check_conservation(psi_in, locals()[f'psi_{s}_reg'][nt], dxc)
        logging.info(f"{plot_args[c]['label']} - Mass gained: {locals()[f'csv_psi_{s}']:.4E}")
        locals()[f'bdn_psi_{s}'] = epm.check_boundedness(psi_in, locals()[f'psi_{s}_reg'][nt])
        logging.info(f"{plot_args[c]['label']} - Boundedness: {locals()[f'bdn_psi_{s}']}")         
        locals()[f'TV_psi_{s}'] = epm.totalvariation(locals()[f'psi_{s}_reg'][nt])
        logging.info(f"{plot_args[c]['label']} - Variation: {locals()[f'TV_psi_{s}']:.4E}")
        logging.info('')

    ##########################
    #### Plot experiments ####
    ##########################
    
    logging.info('')
    logging.info('========== Data during the time integration ==========')

    # Setup plot for results (mass, min/max, RMSE) over time
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(17, 7))

    # Mass over time (ax1)
    for c in range(len(cases)):        
        s = plot_args[c]['label']
        mass = np.zeros(nt+1)
        for it in range(nt+1):
            mass[it] = epm.totalmass(locals()[f'psi_{s}_reg'][it], dxc)
        ax1.plot(np.arange(0,nt+1), mass, **plot_args[c])
    ax1.set_title('Mass')
    ax1.set_xlabel('Time')
    ax1.legend()

    # Boundedness (min/max) over time (ax2)
    for c in range(len(cases)):        
        s = plot_args[c]['label']
        minarr = np.min(locals()[f'psi_{s}_reg'], axis=1)
        maxarr = np.max(locals()[f'psi_{s}_reg'], axis=1)
        logging.info('')
        logging.info(f'{plot_args[c]['label']} - Minimum during the time integration: {np.min(minarr)}')
        logging.info(f'{plot_args[c]['label']} - Maximum during the time integration: {np.max(maxarr)}')
        ax2.plot(np.arange(0,nt+1), minarr, **plot_args[c])
        ax2.plot(np.arange(0,nt+1), maxarr, **plot_args[c])
    ax2.set_title('Bounds')
    ax2.set_xlabel('Time')
    ax2.legend()
        
    # Error over time (ax3)
    for c in range(len(cases)):        
        s = plot_args[c]['label']
        rmse_time = np.zeros(nt+1)
        for it in range(nt+1):     
            rmse_time[it] = epm.l2norm(locals()[f'psi_{s}_reg'][it], locals()['psi_an_reg'][it], dxc) 
        logging.info('')
        logging.info(f'{plot_args[c]['label']} - Max L2 norm during the time integration: {np.max(rmse_time)}')
        ax3.plot(np.arange(0,nt+1), rmse_time, **plot_args[c])
    ax3.set_yscale('log')
    ax3.set_title('$l_2$')
    ax3.set_xlabel('Time')
    ax3.legend()

    # Save plot for results (mass, min/max, RMSE) over time
    plt.savefig(outputdir + f'experiments.pdf')
    plt.tight_layout()
    plt.close()

    # Calculate and plot error over grid spacing (for the final timestep) if check_orderofconvergence is True
    if check_orderofconvergence == True:
        # Setup plot
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        gridscale = np.logspace(0, np.log10(2*factor), num=10)
        gridsizes = resolution[0]*gridscale
        for c in range(len(cases)):        
            s = plot_args[c]['label']
            # Calculate error for each grid spacing (one or three)
            rmse = np.zeros(len(nx_arr))
            dxc_arr = np.zeros(len(nx_arr))
            for xi in range(len(nx_arr)):
                l = gridlabels[xi]
                nx = nx_arr[xi]
                nt = nt_arr[xi]                
                # Calculate RMSE for each grid spacing at the final time, assume uniform grid
                rmse[xi] = epm.l2norm(locals()[f'psi_{s}_{l}'][nt], locals()[f'psi_an_{l}'][nt], resolution[xi]) # Calculate RMSE for each grid spacing at the final time            

            # Plot error over grid spacing
            ax1.scatter(resolution, rmse, marker=plot_args[c]['marker'], label=f'$\\Psi$ {plot_args[c]['label']}', color=plot_args[c]['color'])
            logging.info('')
            logging.info(f'{cases[c]['scheme']} - L2 norm array for the different resolutions (fine, coarse, reg): {rmse}')

            # Order of accuracy lines in the plot for reference
            firstorder = rmse[0]*gridscale
            secondorder = rmse[0]*gridscale*gridscale
            thirdorder = rmse[0]*gridscale*gridscale*gridscale
            fourthorder = rmse[0]*gridscale*gridscale*gridscale*gridscale
            ax1.plot(gridsizes, firstorder, color='grey', linestyle=':', linewidth=0.5)
            ax1.plot(gridsizes, secondorder, color='grey', linestyle='-', linewidth=0.5)
            ax1.plot(gridsizes, thirdorder, color='black', linestyle=':', linewidth=0.5)
            ax1.plot(gridsizes, fourthorder, color='black', linestyle='-', linewidth=0.5)
        
        # Plot details
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title(f'$l_2$ vs {var_acc} at t={nt*dt}')
        ax1.set_ylabel('$l_2$')
        ax1.set_xlabel(var_acc)
        ax1.legend()

        # Save plot of error over grid spacing
        plt.tight_layout()
        if save_as == 'test':
            plt.savefig(outputdir + 'L2.pdf')
        elif save_as == 'store':
            plt.savefig(outputdir + f'L2_{var_acc}.pdf')
        plt.close()

    ###########################
    #### Create animations ####
    ###########################

    fields, colors = [], []
    # Create animation from the data
    if create_animation == True:
        for c in range(len(cases)):        
            s = plot_args[c]['label']
            fields.append(locals()[f'psi_{s}_reg'])
        anim.create_animation_from_data(fields, len(schemenames), locals()['psi_an_reg'], psi_in, nt, dt, xc, outputdir, plot_args, xmax, ymax=ymax)

    print('Done')
    logging.info('')
    logging.info('================================= Done ===================================')
    logging.info('')


def callscheme(case, nt, dt, uf, dxc, psi_in, u_setting, verbose=True):
    """Takes all the input variables and the scheme name and calls the scheme with the appropriate input arguments."""

    # Tranlate the scheme name to a function in schemes.py
    sc = case["scheme"]
    fn = getattr(sch, f'{sc}')

    # Remove 'scheme' key from dictionary
    exclude = {"scheme"}
    params = ut.without_keys(case, exclude)

    # Call the scheme
    if verbose == True: print(f'Running {sc} with parameters {params}')
    #startscheme = timeit.default_timer()
    #print(f'--> Starting runtime for {sc}, nt, nx: {timeit.default_timer() - startscheme:.4f} s, {nt}, {len(psi_in)}')
    psi = fn(psi_in.copy(), nt, dt, uf, dxc, u_setting, **params) # check whether the scheme is called correctly!!! with the u_setting parameters etc
    ##print(case)
    ##print(psi[-1])
    ##print()
    #plt.plot(psi[-1])
    #plt.title('After exiting the scheme function')
    #plt.show()
    if verbose == True:
        logging.info(f'Final psi for {sc} with parameters {params} and nx={len(psi_in)}: {psi[-1]}')
        logging.info('')
    #print(f'--> Runtime for {sc}, nt, nx: {timeit.default_timer() - startscheme:.4f} s, {nt}, {len(psi[-1])}')

    return psi

    
if __name__ == "__main__": 
    starttime = timeit.default_timer()
    
    main()

    print('Total runtime: ', timeit.default_timer() - starttime)