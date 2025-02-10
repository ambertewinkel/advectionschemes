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
    Schemes included: FTBS, FTFS, FTCS, CTBS, CTFS, CTCS, Upwind, BTBS, BTFS, BTCS, CNBS, MPDATA, three hybrid schemes and Jacobi and Gauss-Seidel iterations.
    """

    #############################
    #### Input and testcases ####
    #############################

    # Test or save output in name-specified folder
    save_as = 'test'             # 'test' or 'store'; determines how the output is saved
    
    # Input booleans
    limitCto1 = False
    create_animation = False
    check_orderofconvergence = False
    accuracy_in = 'space with C const' # 'space with dt const' or 'time with dx const' or 'space with C const'; (relevant only if check_orderofconvergence == True)
    date = dati.date.today().strftime("%Y%m%d")                   # date of the run
    datetime = dati.datetime.now().strftime("%d%m%Y-%H%M%S")      # date and time of the run

    # Input cases
    cases = [\
        #{'scheme': 'PPM'}
        #{'scheme': 'PPM'},
        #{'scheme': 'PPM', 'FCT': True},
        #{'scheme': 'PPM', 'multiFCT': True},
        {'scheme': 'PPM', 'multiFCT': True, 'nFCT': 9},
        #{'scheme': 'RK2QC_noPC', 'multiFCT': True, 'nFCT': 3},
        #{'scheme': 'RK2QC_noPC', 'multiFCT': True, 'nFCT': 4},
        #{'scheme': 'RK2QC_noPC', 'multiFCT': True, 'nFCT': 5},
        #{'scheme': 'RK2QC_noPC', 'multiFCT': True, 'nFCT': 6},
        #{'scheme': 'RK2QC_noPC', 'multiFCT': True, 'nFCT': 7},
        #{'scheme': 'RK2QC_noPC', 'multiFCT': True, 'nFCT': 8},
        #{'scheme': 'RK2QC_noPC', 'multiFCT': True, 'nFCT': 9},
        ##{'scheme': 'RK2QC_noPC', 'set_alpha': 'half', 'nonnegative': True},
        ##{'scheme': 'RK2QC_noPC', 'set_alpha': 'half', 'doubleFCT': True},
        ##{'scheme': 'RK2QC_noPC', 'set_alpha': 'half', 'doubleFCT_noupdate': True},
        ##{'scheme': 'RK2QC_noPC', 'set_alpha': 'half', 'FCTnonneg': True},
        
        #{'scheme': 'IRK3QC'},
        ]
    
    plot_args = [\
        #{'label':'PPM', 'color':'blue', 'marker':'x', 'linestyle':'-'},
        #{'label':'PPM', 'color':'blue', 'marker':'o', 'linestyle':'-'},
        #{'label':'PPM', 'color':'red', 'marker':'X', 'linestyle':'-'},
        #{'label':'PPM_mFCT1', 'color':'green', 'marker':'x', 'linestyle':'-'},
        {'label':'PPM_mFCT9', 'color':'yellow', 'marker':'+', 'linestyle':'-'},
        #{'label':'AdImExCubic_mFCT3', 'color':'limegreen', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImExCubic_mFCT4', 'color':'springgreen', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImExCubic_mFCT5', 'color':'aquamarine', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImExCubic_mFCT6', 'color':'turquoise', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImExCubic_mFCT7', 'color':'cyan', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImExCubic_mFCT8', 'color':'lightblue', 'marker':'x', 'linestyle':'-'},
        #{'label':'AdImExCubic_mFCT9', 'color':'royalblue', 'marker':'x', 'linestyle':'-'},
        ##{'label':'AdImExCubic_nn', 'color':'orange', 'marker':'x', 'linestyle':':'},
        ##{'label':'AdImExCubic_dFCT', 'color':'purple', 'marker':'+', 'linestyle':':'},
        ##{'label':'AdImExCubic_dFCT_noupdate', 'color':'green', 'marker':'x', 'linestyle':'-'},
        ##{'label':'AdImExCubic_FCTnonneg', 'color':'black', 'marker':'x', 'linestyle':'-'},
        #{'label':'IRK3QC', 'color':'blue', 'marker':'x', 'linestyle':'-'},
        ]

    # Initial conditions
    analytic = an.combi         # initial condition, options: sine, cosbell, tophat, or combi
    nx = 80                     # number of points in space
    xmax = 1.                   # physical domain parameters
    uconstant = 1.#3.125#10.1#1.26#3.125#6.125#3.125#6.25           # constant velocity
    nt = 1#32#int(100/uconstant)                  # number of time steps
    dt = 0.03125                   # time step
    coords = 'uniform'          # 'uniform' or 'stretching'
    cconstant = uconstant*dt/(xmax/nx)  # Courant number # only used for title in final.pdf

    schemenames = [case["scheme"] for case in cases]
    schemenames_settings = str(analytic.__name__) + f'_t{nt*dt:.4f}_u{uconstant:.4f}_' + 'RK2QC_noPC_FCT' #"-".join(schemenames)
    
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
    logging.info(f'Number of grid points: {nx}')
    logging.info(f'Number of time steps: {nt}')
    logging.info(f'Time step: {dt} s')
    logging.info(f'Total simulated time: {nt*dt:.4f} s')
    logging.info(f'Velocity: {uconstant}')     
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
    if check_orderofconvergence == True: # Run schemes for two extra grid spacings
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
        dt_arr = np.array([dt], dtype=float)
        nt_arr = np.array([nt], dtype=int)
        gridlabels = ['reg']

    # Calculate numerical results
    for xi in range(len(nx_arr)): # Loop over 1 or 3 grid spacings
        nx = nx_arr[xi]
        dt = dt_arr[xi]
        nt = nt_arr[xi]
        uf = np.full(nx, uconstant)
        l = gridlabels[xi]

        # Check whether to limit the Courant number by limiting the grid spacing
        if limitCto1 == True: 
            cmax = 1.
            dxcmin = np.min(0.5*dt*(np.roll(uf,-1) + uf)/cmax)
        else:
            dxcmin = 0.

        # Setup grid for each of the grid spacings
        if coords == 'stretching':
            xf, dxc, xc, dxf = gr.coords_stretching(xmax, nx, nx/2, dxcmin=dxcmin) # points in space, length of spatial step
        elif coords == 'uniform':
            xf, dxc, xc, dxf = gr.coords_uniform(xmax, nx) # points in space, length of spatial step
        elif coords == 'weller':
            xf, dxc, xc, dxf = gr.coords_welleretal2022(xmax, nx) # points in space, length of spatial step
        else: 
            logging.info('Error: invalid coordinates')

        # Calculate velocity and Courant number at cell centers 
        uc = gr.linear(xc, xf, uf)       # velocity at cell centers
        cc = 0.5*dt*(np.roll(abs(uf),-1) + abs(uf))/dxc # Courant number at cell centers
        cmax = np.max(cc)
        cmin = np.min(cc)

        logging.info(f'Min Courant number: {cmin:.4f}')
        logging.info(f'Max Courant number: {cmax:.4f}')  

            # Print and plot grid and Courant number (solely for the regular grid spacing)
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
            locals()[f'psi_an_{l}'][it] = analytic(xc, xmax, uc, it*dt)
        a = locals()[f'psi_an_{l}'][-1].copy()
        logging.info(f"Analytic solution for nx={nx}, nt={nt}, dt={dt}: {a}")
        logging.info('')

        # Calculate initial condition
        psi_in = locals()[f'psi_an_{l}'][0]
        
        # Calculate numerical solutions for each scheme through time
        # Output is 2D field ([1d time, 1d space])
        for c in range(len(cases)):
            s = plot_args[c]['label']
            locals()[f'psi_{s}_{l}'] = callscheme(cases[c], nt, dt, uf, dxc, psi_in)

    ##########################
    #### Plotting schemes ####
    ##########################
    
    plt.figure(figsize=(7,4))
    # Plotting the final time step for each scheme in the same plot
    plt.plot(xc, locals()['psi_an_reg'][nt], label='Analytic', linestyle='-', color='k')
    for c in range(len(cases)):        
        s = plot_args[c]['label']
        plt.plot(xc, locals()[f'psi_{s}_reg'][nt], **plot_args[c])
    ut.design_figure(plotname, f'$\\Psi$ at t={nt*dt} with C={cconstant}', \
                     'x', '$\\Psi$', 0., xmax, True, -0.1, 1.1)
        
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
            ax1.plot(gridsizes, firstorder, color='black', linestyle=':')
            ax1.plot(gridsizes, secondorder, color='black', linestyle=':')
            ax1.plot(gridsizes, thirdorder, color='black', linestyle=':')
        
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
        anim.create_animation_from_data(fields, len(schemenames), locals()['psi_an_reg'], nt, dt, xc, outputdir, plot_args, xmax)

    print('Done')
    logging.info('')
    logging.info('================================= Done ===================================')
    logging.info('')


def callscheme(case, nt, dt, uf, dxc, psi_in):
    """Takes all the input variables and the scheme name and calls the scheme with the appropriate input arguments."""

    # Tranlate the scheme name to a function in schemes.py
    sc = case["scheme"]
    fn = getattr(sch, f'{sc}')

    # Remove 'scheme' key from dictionary
    exclude = {"scheme"}
    params = ut.without_keys(case, exclude)

    # Call the scheme
    print(f'Running {sc} with parameters {params}')
    startscheme = timeit.default_timer()
    #print(f'--> Starting runtime for {sc}, nt, nx: {timeit.default_timer() - startscheme:.4f} s, {nt}, {len(psi_in)}')
    psi = fn(psi_in.copy(), nt, dt, uf, dxc, **params)
    #plt.plot(psi[-1])
    #plt.title('After exiting the scheme function')
    #plt.show()
    logging.info(f'Final psi for {sc} with parameters {params} and nx={len(psi_in)}: {psi[-1]}')
    logging.info('')
    #print(f'--> Runtime for {sc}, nt, nx: {timeit.default_timer() - startscheme:.4f} s, {nt}, {len(psi[-1])}')

    return psi

    
if __name__ == "__main__": 
    starttime = timeit.default_timer()
    
    main()

    print('Total runtime: ', timeit.default_timer() - starttime)