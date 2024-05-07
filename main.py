# Main code with various numerical schemes to solve the advection equation
# Author:   Amber te Winkel
# Email:    a.j.tewinkel@pgr.reading.ac.uk

import numpy as np
import matplotlib.pyplot as plt
import sys
import schemes as sch
import experiments as epm
import utils as ut
import analytic as an
import grid as gr
import animation as anim
import os
import datetime as dati
import logging

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
    plot_timesteps = False #!!!
    create_animation = True
    check_orderofconvergence = False
    date = dati.date.today().strftime("%d%m%Y")                   # date of the run
    datetime = dati.datetime.now().strftime("%d%m%Y-%H%M%S")      # date and time of the run

    # Input cases
    cases = [\
        {'scheme':'hybrid_MPDATA_BTBS1J', 'do_beta':'switch', 'solver':'Jacobi', 'niter':10},
        {'scheme':'hb_imexMPDATA',        'do_beta':'switch', 'solver':'numpy'},
        {'scheme':'imMPDATA',                                 'solver':'numpy'}
        ]
    
    plot_args = [\
        {'color':'red',    'marker':'o', 'linestyle':'-'},
        {'color':'blue',   'marker':'x', 'linestyle':'-'},
        {'color':'orange', 'marker':'+', 'linestyle':'-'}
        ]

    # Initial conditions
    analytic = an.combi         # initial condition, options: cosbell, tophat, or combi
    dt = 0.01                   # time step
    nt = 100                    # number of time steps
    nx = 40                     # number of points in space
    xmax = 1.                   # physical domain parameters
    uconstant = 1.              # constant velocity
    coords = 'uniform'          # 'uniform' or 'stretching

    schemenames = [case["scheme"] for case in cases]
    str_settings = '_' + str(analytic.__name__) + '_t'+ f"{nt*dt:.2f}" + '_b' + cases[0]['do_beta'][0] + '_g' + coords + '_u' + f'{uconstant:.1f}' #!!!
    str_schemenames_settings = "-".join(schemenames) + str_settings
    filebasename = [s  + str_settings for s in schemenames] # name of the directory to save the animation and its corresponding plots in
            # !!! To do: when option to include niter in hybrid scheme, add niter to the filebasename
    
    ##################################
    #### Setup output and logging ####
    ##################################

    # Setup output directory
    # Check if ./output/ and outputdir exist, if not create them, if so, !!! to do: if so give error message and choice to overwrite or n
    if not os.path.exists('./output/'):
        os.mkdir('./output/')
        print("Output folder created!")
    # Determine where to save the output
    if save_as == 'test':
        outputdir = './output/test/' #
        filename = outputdir + 'out.log'
        plotname = 'final'
    elif save_as == 'store':
        if not os.path.exists('./output/' + date + '/'):
            os.mkdir('./output/' + date + '/')
            print("Folder %s created!" % date)
        outputdir = './output/' + date + '/' + str_schemenames_settings + '/' 
        filename = outputdir + 'out_' + str_schemenames_settings + '.out' 
        plotname = 'final_' + str_schemenames_settings
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        print("Folder %s created!" % outputdir)

    # Set up logging
    print(f'See output file {filename}')    
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
    logging.info(f'Date and time: {datetime}')
    logging.info(f'Output directory: {outputdir}')
    logging.info('')
    logging.info(f'Analytic function: {analytic}')
    logging.info(f'Number of grid points: {nx}')
    logging.info(f'Number of time steps: {nt}')
    logging.info(f'Time step: {dt}')
    logging.info(f'Total runtime: {nt*dt:.2f} s')
    logging.info(f'Velocity:, {uconstant}')     
    logging.info(f'Schemes included are: {schemenames}')
    logging.info(f'Cases:')
    for case in cases:
        logging.info(case)
    logging.info('')

    #####################
    #### Run schemes ####
    #####################

    # Setup: run schemes for one or three grid spacings (nx*factor, nx/factor, nx)
    if check_orderofconvergence == True: # Run schemes for two extra grid spacings
        factor = 2
        nx_arr = np.array([nx*factor, nx/factor, nx], dtype=int)
        dt_arr = np.array([dt/factor, dt*factor, dt], dtype=float)
        nt_arr = np.array([nt*factor, nt/factor, nt], dtype=int)
        gridlabels = ['fine', 'coarse', 'reg']
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
        else: 
            logging.info('Error: invalid coordinates')

        # Calculate velocity and Courant number at cell centers 
        uc = gr.linear(xc, xf, uf)       # velocity at cell centers
        cc = 0.5*dt*(np.roll(abs(uf),-1) + abs(uf))/dxc # Courant number at cell centers
        cmax = np.max(cc)
        cmin = np.min(cc)

        logging.info(f'Min Courant number: {cmin:.2f}')
        logging.info(f'Max Courant number: {cmax:.2f}')  

            # Print and plot grid and Courant number (solely for the regular grid spacing)
        if gridlabels[xi] == 'reg':
            logging.info('The (cell center) points and Courant numbers are:')
            for i in range(nx):
                logging.info(i, "%.2f" %xc[i], "%.2f" %cc[i]) #!!!
            logging.info('')
            ut.plot_Courant(xc, cc, outputdir)
            ut.plot_grid(xc, dxc, outputdir)
            
        # Calculate analytic solutions for each time step
        locals()[f'psi_an_{l}'] = np.zeros((nt+1, nx))
        for it in range(nt+1):
            locals()[f'psi_an_{l}'][it] = analytic(xc, xmax, uc, it*dt)

        # Calculate initial condition
        psi_in = locals()[f'psi_an_{l}'][0]

        # Calculate numerical solutions for each scheme through time
        # Output is 2D field ([1d time, 1d space])
        for c in range(len(cases)):
            locals()[f'psi_{cases[c]["scheme"]}_{l}'] = callscheme(case, nt, dt, uf, dxc, psi_in)

    ##########################
    #### Plotting schemes ####
    ##########################
    
    # Plotting the final time step for each scheme in the same plot
    plt.plot(xc, psi_in, label='Initial', linestyle='-', color='grey')
    plt.plot(xc, locals()['psi_an_reg'][nt], label='Analytic', linestyle='-', color='k')
    for s in schemenames:
        si = schemenames.index(s)
        if 'Jacobi' in s or 'GaussSeidel' in s:
            slabel = f'{s}, it={case[si]['niter']}'
        elif s == 'BTBS':
            slabel = 'BTBS_numpy'
        else: 
            slabel = s
        plt.plot(xc, locals()[f'psi_{s}_reg'][nt], label=f'{slabel}', **plot_args[si])
    ut.design_figure(plotname + '.pdf', outputdir, f'$\\Psi$ at t={nt*dt}', \
                     'x', '$\\Psi$', True, -1.5, 1.5)

    #####################
    #### Experiments ####
    #####################

    # Print experiment results in .out
    logging.info('')
    logging.info('========== Data at the final time step ==========')
    logging.info('')
    
    # Conservation, boundedness and total variation Psi
    logging.info('')
    logging.info('========== Conservation, boundedness and total variation ==========')
    logging.info('')
    csv_psi_analytic = epm.check_conservation(psi_in, locals()['psi_an_reg'][nt], dxc)
    logging.info(f'Analytic - Mass gained: {csv_psi_analytic:.2E}')    
    bdn_psi_analytic = epm.check_boundedness(psi_in, locals()['psi_an_reg'][nt])
    logging.info(f'Analytic - Boundedness: {bdn_psi_analytic}')    
    logging.info('')
    for s in schemenames:
        locals()[f'csv_psi_{s}'] = epm.check_conservation(psi_in, locals()[f'psi_{s}_reg'][nt], dxc)
        logging.info(f"{s} - Mass gained: {locals()[f'csv_psi_{s}']:.2E}")
        locals()[f'bdn_psi_{s}'] = epm.check_boundedness(psi_in, locals()[f'psi_{s}_reg'][nt])
        logging.info(f"{s} - Boundedness: {locals()[f'bdn_psi_{s}']}")         
        locals()[f'TV_psi_{s}'] = epm.totalvariation(locals()[f'psi_{s}_reg'][nt])
        logging.info(f"{s} - Variation: {locals()[f'TV_psi_{s}']:.2E}")
        logging.info('')

    ##########################
    #### Plot experiments ####
    ##########################
    
    logging.info('')
    logging.info('========== Data during the time integration ==========')

    # Setup plot for results (mass, min/max, RMSE) over time
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(17, 7))

    # Mass over time (ax1)
    for s in schemenames:
        si = schemenames.index(s)
        mass = np.zeros(nt+1)
        for it in range(nt+1):
            mass[it] = epm.totalmass(locals()[f'psi_{s}_reg'][it], dxc)
        ax1.plot(np.arange(0,nt+1), mass, label=s, **plot_args[si])
    ax1.set_title('Mass')
    ax1.set_xlabel('Time')
    ax1.legend()

    # Boundedness (min/max) over time (ax2)
    for s in schemenames:
        si = schemenames.index(s)
        minarr, maxarr = np.zeros(nt+1), np.zeros(nt+1)
        for it in range(nt+1):     
            minarr[it] = np.min(locals()[f'psi_{s}_reg'][it]) # !!! np max can perhaps introduce axis and for loop is not necessary?
            maxarr[it] = np.max(locals()[f'psi_{s}_reg'][it])
        logging.info('')
        logging.info(f'Scheme - {s}')
        logging.info(f'Minimum bound during the time integration: {np.min(minarr)}')
        logging.info(f'Maximum bound during the time integration: {np.max(maxarr)}')
        ax2.plot(np.arange(0,nt+1), minarr, label=f'Min {s}', **plot_args[si])
        ax2.plot(np.arange(0,nt+1), maxarr, label=f'Max {s}', **plot_args[si])
    ax2.set_title('Bounds')
    ax2.set_xlabel('Time')
    ax2.legend()
        
    # Error over time (ax3)
    for s in schemenames:
        si = schemenames.index(s)
        rmse_time = np.zeros(nt+1)
        for it in range(nt+1):     
            rmse_time[it] = epm.rmse(locals()[f'psi_{s}_reg'][it], locals()['psi_an_reg'][it], dxc) 
        logging.info('')
        logging.info(f'Scheme - {s}')
        logging.info(f'Max RMSE during the time integration: {np.max(rmse_time)}')
        ax3.plot(np.arange(0,nt+1), rmse_time, label=s, **plot_args[si])
    ax3.set_yscale('log')
    ax3.set_title('RMSE')
    ax3.set_xlabel('Time')
    ax3.legend()

    # Save plot for results (mass, min/max, RMSE) over time
    plt.savefig(outputdir + f'epm_over_time_' + str_schemenames_settings + '.pdf')
    plt.tight_layout()
    plt.close()

    # Calculate and plot error over grid spacing (for the final timestep) if check_orderofconvergence is True
    if check_orderofconvergence == True:
        # Setup plot
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        for s in schemenames:
            si = schemenames.index(s)
            # Calculate error for each grid spacing (one or three)
            rmse_x = np.zeros(len(nx_arr))
            dxc_arr = np.zeros(len(nx_arr))
            for xi in range(len(nx_arr)):
                l = gridlabels[xi]
                nx = nx_arr[xi]
                nt = nt_arr[xi]                
                if coords == 'stretching':
                    xf, dxc, xc, dxf = gr.coords_stretching(xmax, nx, nx/2, dxcmin=dxcmin) # points in space, length of spatial step
                elif coords == 'uniform':
                    xf, dxc, xc, dxf = gr.coords_uniform(xmax, nx) # points in space, length of spatial step
                else: 
                    print('Error: invalid coordinates')
                rmse_x[xi] = epm.rmse(locals()[f'psi_{s}_{l}'][nt], locals()[f'psi_an_{l}'][nt], dxc) # Calculate RMSE for each grid spacing at the final time
                dxc_arr[xi] = np.mean(dxc)

            # Plot error over grid spacing
            ax1.scatter(dxc_arr, rmse_x, marker='+', label=f'Psi {s}')

        # Plot details
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title(f'RMSE at t_final')
        ax1.set_xlabel('Mean dx')
        ax1.legend()

        # Save plot of error over grid spacing
        plt.savefig(outputdir + f'RMSE_over_dx_' + str_schemenames_settings + '.pdf')
        plt.tight_layout()
        plt.close()

        # Calculate order of convergence
        # !!! To do: calculate order of convergence

    """
    #### Error analysis for a single scheme
    scheme = 'Upwind'
    fn = getattr(sch, f'{scheme}')
    nx_arr = np.array([nx*2, nx, nx/2], dtype=int)
    dx_arr = xmax/nx_arr
    dt_arr = c[0]*dx_arr/u[0]   # This assumes a constant c throughout the domain
    nt_arr = np.array(nt*dt/dt_arr, dtype=int)  # nt*dt is total time above
    rmse_arr = np.zeros(len(dx_arr))

    for i in range(len(nx_arr)):
        c_error = np.full(nx_arr[i], c[0])
        x_error = np.linspace(xmin, xmax, nx_arr[i], endpoint=False)
        psi_in_error = an.cosinebell(x_error)
        psi_Upwind_error = fn(psi_in_error.copy(), nt_arr[i], c_error)
        psi_an_error = an.cosinebell(x_error, nt_arr[i], c_error)
        rmse_arr[i] = epm.rmse(psi_an_error, psi_Upwind_error, dx_arr[i])

    # log-log plot of RMSE
    plt.loglog(dx_arr, rmse_arr, '-x', label=f'{scheme}')
    plt.loglog(dx_arr, dx_arr, color='green', label='O(dx) accurate')
    ut.design_figure(f'loglog_{scheme}.pdf', f'RMSE for {scheme} scheme', 'dx', 'RMSE')
    """

    ###########################
    #### Create animations ####
    ###########################

    fields, colors = [], []
    # Create animation from the data
    if create_animation == True:
        animdir = outputdir + 'animations/'
        if not os.path.exists(animdir):
            os.mkdir(animdir)
        for s in schemenames:
            fields.append(locals()[f'psi_{s}_reg'])
        anim.create_animation_from_data('Psi', fields, len(schemenames), schemenames, locals()['psi_an_reg'], nt, dt, xc, animdir, plot_args)

    print('Done')
    logging.info('')
    logging.info('================================= Done ===================================')
    logging.info('')

def callscheme(case, nt, dt, uf, dxc, psi_in): #!!! Is this correct? Generalize this function
    """Takes all the input variables and the scheme name and calls the scheme with the appropriate input arguments."""

    s = case["scheme"]
    fn = getattr(sch, f'{s}')
    if 'Jacobi' in s or 'GaussSeidel' in s:
        if 'hybrid' in s:
            psi = fn(psi_in.copy(), nt, dt, uf, dxc, case['niter'], case['do_beta'])
        else:
            psi = fn(psi_in.copy(), nt, dt, uf, dxc, case['niter'])
    else:
        if 'hybrid' in s:
            psi = fn(psi_in.copy(), nt, dt, uf, dxc, case['do_beta'])
        else:
            psi = fn(psi_in.copy(), nt, dt, uf, dxc)
            
    return psi
    
if __name__ == "__main__": main()
