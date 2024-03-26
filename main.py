# Main code with various numerical schemes to solve the advection equation
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

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

# !!! future: allow the wind to change over time?

def main():
    """
    This function computes and plots the results of various numerical schemes 
    with 1D periodic space and time. Results are compared to the analytic soln. 
    Two initial conditions are considered: a Gaussian distribution and a step 
    function, both defined on a subdomain. 
    Schemes included: FTBS, FTFS, FTCS, CTBS, CTFS, CTCS, Upwind, BTBS, BTFS, BTCS, CNBS, MPDATA
    """
    # Input booleans
    schemenames = ['hybrid_Upwind_BTBS1J', 'hybrid_Upwind_Upwind1J']
    predefined_output_file = True
    keep_model_stable = False
    create_animation = True
    check_orderofconvergence = True
    do_beta = 'switch'          # 'switch' or 'blend'
    coords = 'stretching'       # 'uniform' or 'stretching'
    niter = 1                   # number of iterations (for Jacobi or Gauss-Seidel)
    # !!! implement criterion for convergence with Jacobi and Gauss-Seidel iterations?

    # Saving the reference of the standard output
    original_stdout = sys.stdout 

    #######################
    #### Run the model ####
    #######################

    # Initial conditions
    dt = 0.1                    # time step
    nt = 100                    # number of time steps
    nx = 40                     # number of points in space
    xmax = 2.0                  # physical domain parameters

    # Setup output
    str_settings = '_t'+ f"{nt*dt:.2f}" + '_ks' + str(keep_model_stable)[0] + '_b' + do_beta[0] + '_g' + coords[0]
    str_schemenames_settings = "-".join(schemenames) + str_settings
    filebasename = [s  + str_settings for s in schemenames] # name of the directory to save the animation and its corresponding plots in
            # !!! To do: when option to include niter in hybrid scheme, add niter to the filebasename
    outputdir = './output/' + str_schemenames_settings + '/'
    # Check if outputdir exists, if not create it, if so, !!! to do: if so give error message and choice to overwrite or n
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        print("Folder %s created!" % outputdir)
    
    filename = outputdir + 'outputdata_' + str_schemenames_settings + '.out' 
    plotname1 = 'Psi1_' + str_schemenames_settings
    plotname2 = 'Psi2_' + str_schemenames_settings
    
    #################
    #### Schemes ####
    #################

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
        uf = np.full(nx, 0.2)
        l = gridlabels[xi]

        # Check whether to limit the Courant number by limiting the grid spacing
        if keep_model_stable == True: 
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
            print('Error: invalid coordinates')

        # Calculate velocity and Courant number at cell centers 
        uc = gr.linear(xc, xf, uf)       # velocity at cell centers
        cc = 0.5*dt*(np.roll(abs(uf),-1) + abs(uf))/dxc # Courant number at cell centers
        cmax = np.max(cc)
        cmin = np.min(cc)

        # Calculate analytic solutions for each time step
        locals()[f'psi1_an_{l}'], locals()[f'psi2_an_{l}'] = np.zeros((nt+1, nx)), np.zeros((nt+1, nx))
        for it in range(nt+1):
            locals()[f'psi1_an_{l}'][it] = an.analytic1(xc, xmax, uc, it*dt)
            locals()[f'psi2_an_{l}'][it] = an.analytic2(xc, xmax, uc, it*dt)

        # Calculate initial functions
        psi1_in = locals()[f'psi1_an_{l}'][0]
        psi2_in = locals()[f'psi2_an_{l}'][0]

        # Calculate numerical solutions for each scheme through time
        # Output is 2D field ([1d time, 1d space])
        for s in schemenames:
            fn = getattr(sch, f'{s}')
            if 'Jacobi' in s or 'GaussSeidel' in s:
                if 'hybrid' in s:
                    locals()[f'psi1_{s}_{l}'] = fn(psi1_in.copy(), nt, dt, uf, dxc, niter, do_beta)
                    locals()[f'psi2_{s}_{l}'] = fn(psi2_in.copy(), nt, dt, uf, dxc, niter, do_beta)
                else:
                    locals()[f'psi1_{s}_{l}'] = fn(psi1_in.copy(), nt, dt, uf, dxc, niter)
                    locals()[f'psi2_{s}_{l}'] = fn(psi2_in.copy(), nt, dt, uf, dxc, niter)
            else:
                if 'hybrid' in s:
                    locals()[f'psi1_{s}_{l}'] = fn(psi1_in.copy(), nt, dt, uf, dxc, do_beta)
                    locals()[f'psi2_{s}_{l}'] = fn(psi2_in.copy(), nt, dt, uf, dxc, do_beta)
                else:
                    locals()[f'psi1_{s}_{l}'] = fn(psi1_in.copy(), nt, dt, uf, dxc)
                    locals()[f'psi2_{s}_{l}'] = fn(psi2_in.copy(), nt, dt, uf, dxc)

    ##########################
    #### Plotting schemes ####
    ##########################
    
    # Plotting setup
    markers = ['x', '+', '+', '', '', '']
    linestyle = ['-','-','-', '--', '-', '--']
    colors = ['red', 'blue', 'orange', 'red', 'lightblue', 'gray']

    # Psi 1: Plotting the final time step for each scheme in the same plot
    plt.plot(xc, psi1_in, label='Initial', linestyle='-', color='grey')
    plt.plot(xc, locals()['psi1_an_reg'][nt], label='Analytic', linestyle='-', color='k')
    for s in schemenames:
        si = schemenames.index(s)
        if 'Jacobi' in s or 'GaussSeidel' in s:
            slabel = f'{s}, it={niter}'
        elif s == 'BTBS':
            slabel = 'BTBS_numpy'
        else: 
            slabel = s
        plt.plot(xc, locals()[f'psi1_{s}_reg'][nt], label=f'{slabel}', marker=markers[si], linestyle=linestyle[si], color=colors[si])
    ut.design_figure(plotname1 + '.png', outputdir, f'$\\Psi_1$ at t={nt*dt}', \
                     'x', '$\\Psi_1$', True, -1.5, 1.5)

    # Psi 2: Plotting the final time step for each scheme in the same plot
    plt.plot(xc, psi2_in, label='Initial', linestyle='-', color='grey')
    plt.plot(xc, locals()['psi2_an_reg'][nt], label='Analytic', linestyle='-', color='k')
    for s in schemenames:
        si = schemenames.index(s)
        if 'Jacobi' in s or 'GaussSeidel' in s:
            slabel = f'{s}, it={niter}'
        else: 
            slabel = s
        plt.plot(xc, locals()[f'psi2_{s}_reg'][nt], label=f'{slabel}', marker=markers[si], linestyle=linestyle[si], color=colors[si])
    ut.design_figure(plotname2 + '.png', outputdir, f'$\\Psi_2$ at t={nt*dt}', \
                     'x', '$\\Psi_2$', True,  -1.5, 1.5)
    plt.close()

    #####################
    #### Experiments ####
    #####################

    with open(filename, 'w') as f:
        # Redirect the standard output to the output file if predefined_output_file is True
        if predefined_output_file == True:
            print(f'See output file {filename}')
            sys.stdout = f
        else: # Output in terminal
            sys.stdout = original_stdout 
        
        # Print results in output file/terminal for the final time step
        print('========== Output file for data at the final time step ==========')
        print()
        print('Schemes included are:', schemenames)
        print(f'Number of timesteps: {nt}')
        print(f'Total runtime: {nt*dt:.2f} s')
        print(f'Min Courant number: {cmin:.2f}')
        print(f'Max Courant number: {cmax:.2f}')        
        print()

        # Print and plot grid and Courant number (solely for the regular grid spacing)
        if gridlabels[xi] == 'reg':
            print('The (cell center) points and Courant numbers are:')
            for i in range(nx):
                print(i, "%.2f" %xc[i], "%.2f" %cc[i])
            print()
            ut.plot_Courant(xc, cc, outputdir)
            ut.plot_grid(xc, dxc, outputdir)
        
        # Conservation, boundedness and total variation Psi1
        print()
        print('========== Psi 1 ==========')
        print()
        csv_psi1_analytic = epm.check_conservation(psi1_in, locals()['psi1_an_reg'][nt], dxc)
        print(f'Analytic - Mass gained: {csv_psi1_analytic:.2E}')    
        bdn_psi1_analytic = epm.check_boundedness(psi1_in, locals()['psi1_an_reg'][nt])
        print(f'Analytic - Boundedness: {bdn_psi1_analytic}')    
        print()
        for s in schemenames:
            locals()[f'csv_psi1_{s}'] = epm.check_conservation(psi1_in, locals()[f'psi1_{s}_reg'][nt], dxc)
            print(f'{s} - Mass gained: {locals()[f'csv_psi1_{s}']:.2E}')
            locals()[f'bdn_psi1_{s}'] = epm.check_boundedness(psi1_in, locals()[f'psi1_{s}_reg'][nt])
            print(f'{s} - Boundedness: {locals()[f'bdn_psi1_{s}']}')         
            locals()[f'TV_psi1_{s}'] = epm.totalvariation(locals()[f'psi1_{s}_reg'][nt])
            print(f'{s} - Variation: {locals()[f'TV_psi1_{s}']:.2E}')
            print()

        # Conservation, boundedness and total variation Psi2
        print()
        print('========== Psi 2 ==========')
        print()
        csv_psi2_analytic = epm.check_conservation(psi2_in, locals()['psi2_an_reg'][nt], dxc)
        print(f'Analytic - Mass gained: {csv_psi2_analytic:.2E}')   
        bdn_psi2_analytic = epm.check_boundedness(psi2_in, locals()['psi2_an_reg'][nt])
        print(f'Analytic - Boundedness: {bdn_psi2_analytic}')   
        print()
        for s in schemenames:
            locals()[f'csv_psi2_{s}'] = epm.check_conservation(psi2_in, locals()[f'psi2_{s}_reg'][nt], dxc)
            print(f'{s} - Mass gained: {locals()[f'csv_psi2_{s}']:.2E}')
            locals()[f'bdn_psi2_{s}'] = epm.check_boundedness(psi2_in, locals()[f'psi2_{s}_reg'][nt])
            print(f'{s} - Boundedness: {locals()[f'bdn_psi2_{s}']}')
            locals()[f'TV_psi2_{s}'] = epm.totalvariation(locals()[f'psi2_{s}_reg'][nt])
            print(f'{s} - Variation: {locals()[f'TV_psi2_{s}']:.2E}')
            print()    

        ##########################
        #### Plot experiments ####
        ##########################
        
        print()
        print('========== Data during the time integration ==========')

        # Setup plot for results (mass, min/max, RMSE) over time
        fig, ((ax1, ax2, ax3),(ax5, ax6, ax7)) = plt.subplots(2, 3, figsize=(17, 10))

        # Mass over time (ax1, ax5)
        for s in schemenames:
            si = schemenames.index(s)
            mass1, mass2 = np.zeros(nt+1), np.zeros(nt+1)
            for it in range(nt+1):
                mass1[it] = epm.totalmass(locals()[f'psi1_{s}_reg'][it], dxc)
                mass2[it] = epm.totalmass(locals()[f'psi2_{s}_reg'][it], dxc)
            ax1.plot(np.arange(0,nt+1), mass1, marker=markers[si], color=colors[si], label=s)
            ax5.plot(np.arange(0,nt+1), mass2, marker=markers[si], color=colors[si], label=s)
        ax1.set_title('Mass Psi1')
        ax5.set_title('Mass Psi2')
        ax1.set_xlabel('Time')
        ax5.set_xlabel('Time')     
        ax1.legend()
        ax5.legend()

        # Boundedness (min/max) over time (ax2, ax6)
        for s in schemenames:
            si = schemenames.index(s)
            minarr1, maxarr1, minarr2, maxarr2 = np.zeros(nt+1), np.zeros(nt+1), np.zeros(nt+1), np.zeros(nt+1)
            for it in range(nt+1):     
                minarr1[it] = np.min(locals()[f'psi1_{s}_reg'][it]) # !!! np max can perhaps introduce axis and for loop is not necessary?
                maxarr1[it] = np.max(locals()[f'psi1_{s}_reg'][it])
                minarr2[it] = np.min(locals()[f'psi2_{s}_reg'][it])
                maxarr2[it] = np.max(locals()[f'psi2_{s}_reg'][it])
            print()
            print(f'Scheme - {s}')
            print(f'Psi 1 - Minimum bound during the time integration: {np.min(minarr1)}')
            print(f'Psi 1 - Maximum bound during the time integration: {np.max(maxarr1)}')
            print(f'Psi 2 - Minimum bound during the time integration: {np.min(minarr2)}')
            print(f'Psi 2 - Maximum bound during the time integration: {np.max(maxarr2)}')
            ax2.plot(np.arange(0,nt+1), minarr1, marker=markers[si], color=colors[si], label=f'Min {s}')
            ax2.plot(np.arange(0,nt+1), maxarr1, marker=markers[si], color=colors[si], label=f'Max {s}')
            ax6.plot(np.arange(0,nt+1), minarr2, marker=markers[si], color=colors[si], label=f'Min {s}')
            ax6.plot(np.arange(0,nt+1), maxarr2, marker=markers[si], color=colors[si], label=f'Max {s}')
        ax2.set_title('Bounds Psi1')
        ax6.set_title('Bounds Psi2')   
        ax2.set_xlabel('Time')
        ax6.set_xlabel('Time')     
        ax2.legend()
        ax6.legend()
            
        # Error over time (ax3, ax7)
        for s in schemenames:
            si = schemenames.index(s)
            rmse_time1, rmse_time2 = np.zeros(nt+1), np.zeros(nt+1)
            for it in range(nt+1):     
                rmse_time1[it] = epm.rmse(locals()[f'psi1_{s}_reg'][it], locals()['psi1_an_reg'][it], dxc) 
                rmse_time2[it] = epm.rmse(locals()[f'psi2_{s}_reg'][it], locals()['psi2_an_reg'][it], dxc)
            print()
            print(f'Scheme - {s}')
            print(f'Psi 1 - Max RMSE during the time integration: {np.max(rmse_time1)}')
            print(f'Psi 2 - Max RMSE during the time integration: {np.max(rmse_time2)}')
            ax3.plot(np.arange(0,nt+1), rmse_time1, marker=markers[si], color=colors[si], label=s)
            ax7.plot(np.arange(0,nt+1), rmse_time2, marker=markers[si], color=colors[si], label=s)
        ax3.set_yscale('log')
        ax7.set_yscale('log')
        ax3.set_title('RMSE Psi1')
        ax7.set_title('RMSE Psi2')    
        ax3.set_xlabel('Time')
        ax7.set_xlabel('Time')         
        ax3.legend()
        ax7.legend()

        # Save plot for results (mass, min/max, RMSE) over time
        plt.savefig(outputdir + f'epm_over_time_' + str_schemenames_settings + '.png')
        plt.tight_layout()
        plt.close()

        # Calculate and plot error over grid spacing (for the final timestep) if check_orderofconvergence is True
        if check_orderofconvergence == True:
            # Setup plot
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
            for s in schemenames:
                # Calculate error for each grid spacing (one or three)
                rmse_x1, rmse_x2 = np.zeros(len(nx_arr)), np.zeros(len(nx_arr))
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
                    rmse_x1[xi] = epm.rmse(locals()[f'psi1_{s}_{l}'][nt], locals()[f'psi1_an_{l}'][nt], dxc) # Calculate RMSE for each grid spacing at the final time
                    rmse_x2[xi] = epm.rmse(locals()[f'psi2_{s}_{l}'][nt], locals()[f'psi1_an_{l}'][nt], dxc)
                    dxc_arr[xi] = np.mean(dxc)

                # Plot error over grid spacing
                ax1.scatter(dxc_arr, rmse_x1, marker='+', label=f'Psi1 {s}')
                ax1.scatter(dxc_arr, rmse_x2, marker='x', label=f'Psi2 {s}')

            # Plot details
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_title(f'RMSE at t_final')
            ax1.set_xlabel('Mean dx')
            ax1.legend()

            # Save plot of error over grid spacing
            plt.savefig(outputdir + f'RMSE_over_dx_' + str_schemenames_settings + '.png')
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
            psi1_in_error = an.analytic1(x_error)
            psi1_Upwind_error = fn(psi1_in_error.copy(), nt_arr[i], c_error)
            psi1_an_error = an.analytic1(x_error, nt_arr[i], c_error)
            rmse_arr[i] = epm.rmse(psi1_an_error, psi1_Upwind_error, dx_arr[i])

        # log-log plot of RMSE
        plt.loglog(dx_arr, rmse_arr, '-x', label=f'{scheme}')
        plt.loglog(dx_arr, dx_arr, color='green', label='O(dx) accurate')
        ut.design_figure(f'loglog_{scheme}.pdf', f'RMSE for {scheme} scheme', 'dx', 'RMSE')
        """

        ###########################
        #### Create animations ####
        ###########################
    
        # Create animation from the data
        if create_animation == True:
            animdir = outputdir + 'animations/'
            if not os.path.exists(animdir):
                os.mkdir(animdir)
            for s in schemenames:
                anim.create_animation_from_data('Psi1_' + filebasename[schemenames.index(s)], locals()[f'psi1_{s}_reg'], locals()['psi1_an_reg'], nt, dt, xc, animdir)
                anim.create_animation_from_data('Psi2_' + filebasename[schemenames.index(s)], locals()[f'psi2_{s}_reg'], locals()['psi2_an_reg'], nt, dt, xc, animdir)

    # Reset the standard output
    sys.stdout = original_stdout 
    print('Done')

if __name__ == "__main__": main()