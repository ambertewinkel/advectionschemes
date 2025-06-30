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
    Schemes included: FTBS, FTFS, FTCS, CTBS, CTFS, CTCS, Upwind, BTBS, BTFS, BTCS, CNBS, MPDATA, three hybrid schemes and Jacobi and Gauss-Seidel iterations (and more, ese schemes.py).
    """

    #############################
    #### Input and testcases ####
    #############################

    # Test or save output in name-specified folder
    save_as = 'store'             # 'test' or 'store'; determines how the output is saved
    
    # We need to set: dt, nx, nt, u_setting, analytic
    xmax = 1.0
    ymax = 25.#200. # for plotting purposes
    dt_LRES_AdImEx = 0.01 # largest time step of the two dt's for the animation dt_LRES before
    nt_LRES_AdImEx = 1#50#100 # largest number of time steps of the two nt's for the animation
    nx_LRES = 40
    dx_LRES = xmax/nx_LRES

    dtfactor_HRESLRES = 10
    dtfactor_ExAdImEx = 10#5

    dt_HRES_AdImEx = dt_LRES_AdImEx/dtfactor_HRESLRES # used to be dt_HRES
    nt_HRES_AdImEx = nt_LRES_AdImEx*dtfactor_HRESLRES
    dx_HRES = dx_LRES/dtfactor_HRESLRES
    nx_HRES = nx_LRES*dtfactor_HRESLRES

    dt_LRES_Ex = dt_LRES_AdImEx/dtfactor_ExAdImEx 
    nt_LRES_Ex = nt_LRES_AdImEx*dtfactor_ExAdImEx

    dt_HRES_Ex = dt_LRES_Ex/dtfactor_HRESLRES 
    nt_HRES_Ex = nt_LRES_Ex*dtfactor_HRESLRES

    u_setting = 'varying_space5'
    analytic = an.analytic_constant
    total_time = nt_LRES_AdImEx*dt_LRES_AdImEx

    #!!! check the courant numbers for the different options! i.e. plot in a single plot?? when plotting the final fields as well?

    # Input booleans
    create_animation = True
    date = dati.date.today().strftime("%Y%m%d")                   # date of the run
    datetime = dati.datetime.now().strftime("%d%m%Y-%H%M%S")      # date and time of the run

    # Input cases
    cases = [\
        #{'scheme': 'ImExRK', 'RK': 'aiUpwind', 'SD': 'BS', 'blend': 'sm', 'HRES': True, 'AdImEx': False},
        #{'scheme': 'ImExRK', 'RK': 'aiUpwind', 'SD': 'BS', 'blend': 'sm', 'HRES': True, 'AdImEx': True},
        {'scheme': 'ImExRK', 'RK': 'UJ31e32', 'SD': 'fifth302', 'blend': 'sm', 'HRES': False, 'AdImEx': False},
        {'scheme': 'ImExRK', 'RK': 'UJ31e32', 'SD': 'fifth302', 'blend': 'sm', 'HRES': False, 'AdImEx': True},
        ]
    
    plot_args = [\
        #{'label':'Ex Upwind HRES', 'color':'blue', 'marker':'', 'linestyle':'-'},
        #{'label':'AdImEx Upwind HRES', 'color':'red', 'marker':'', 'linestyle':'-'},
        {'label':'Ex Strang', 'color':'darkturquoise', 'marker':'+', 'linestyle':'-'},
        {'label':'AdImEx Strang', 'color':'purple', 'marker':'x', 'linestyle':'-'},
        ]

    schemenames = "-".join([case["scheme"] for case in cases])
    HRESbool, AdImExbool = [], []    
    for c in range(len(cases)):
        HRESbool.append(cases[c]['HRES'])
        AdImExbool.append(cases[c]['AdImEx'])
    
    ##################################
    #### Setup output and logging ####
    ##################################

    # Setup output directory
    # Check if ./output/ and outputdir exist, if not create them
    if not os.path.exists('./output/unique_dtdxu/'):
        os.mkdir('./output/unique_dtdxu/')
        print("./output/unique_dtdxu/ created")
    # Determine where to save the output
    if save_as == 'test':
        outputdir = './output/unique_dtdxu/test/'
        filename = outputdir + 'out.log'
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
            print("Folder %s created" % outputdir)
        else:
            filename = outputdir + 'out.log'
            os.remove(filename)
    elif save_as == 'store':
        if not os.path.exists('./output/unique_dtdxu/dated/' + date + '/'):
            os.mkdir('./output/unique_dtdxu/dated/' + date + '/')
            print("Folder %s created" % date)
        outputdir = f'./output/unique_dtdxu/dated/{date}/{schemenames}/' 
        i = 0 
        while os.path.exists(outputdir):
            print("Folder %s already exists" % outputdir)
            i += 1
            outputdir = f'./output/unique_dtdxu/dated/{date}/{schemenames}_{i}/'
        os.mkdir(outputdir)
        print("Folder %s created" % outputdir)
        filename = outputdir + 'out.log'
    plotname = outputdir + 'final.png'#.pdf'

    # Set up logging
    print(f'See output file {filename}')    
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
    logging.info(f'Date and time: {datetime}')
    logging.info(f'Output directory: {outputdir}')
    logging.info('')
    logging.info(f'Total simulated time: {total_time:.4f} s')
    logging.info(f'Velocity setting: {u_setting}')
    logging.info(f'Analytic function: {analytic.__name__}')
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

    # Print experiment results in .log
    logging.info('')
    logging.info('========== Data at the final time step ==========')
    logging.info('')


    xf_HRES, dxc_HRES, xc_HRES, dxf_HRES = gr.coords_uniform(xmax, nx_HRES) # points in space, length of spatial step
    xf_LRES, dxc_LRES, xc_LRES, dxf_LRES = gr.coords_uniform(xmax, nx_LRES) # points in space, length of spatial step
    psi_in_HRES = analytic(xc_HRES, xmax, u=0., t=0.) # u not relevant here but in function call
    psi_in_LRES = analytic(xc_LRES, xmax, u=0., t=0.) # u not relevant here but in function call
    uf_HRES = getattr(an, 'velocity_' + u_setting)(xf_HRES)
    uf_LRES = getattr(an, 'velocity_' + u_setting)(xf_LRES)
    l = 'reg'

    plt.plot(xf_LRES, uf_LRES, linestyle='-', color='k') 
    plt.title('Velocity')
    plt.xlabel('x')
    plt.ylabel('$u$')
    plt.savefig(outputdir + f'{u_setting}.png')
    plt.close()

    for c in range(len(cases)):
        #if cases[c]['HRES'] == True:
        #    if cases[c]['AdImEx'] == True:
        #        dtplot = dt_HRES_AdImEx
        #        cplot = dtplot*uf_HRES/dx_HRES # courant number for HRES
        #        plt.plot(xf_HRES, cplot, label='AdImEx', color='purple', linestyle='-')            
        #    else:
        #        dtplot = dt_HRES_Ex
        #        cplot = dtplot*uf_HRES/dx_HRES # courant number for HRES
        #        plt.plot(xf_HRES, cplot, label='Ex', color='darkturquoise', linestyle='-')
        if cases[c]['HRES'] == False:
            if cases[c]['AdImEx'] == True:
                dtplot = dt_LRES_AdImEx
                cplot = dtplot*uf_LRES/dx_LRES # courant number for LRES
                logging.info(f'Velocity for {plot_args[c]["label"]} with dt={dtplot:.4f}, dx={dx_LRES:.4f}: {uf_LRES}')
                logging.info(f'Courant number for {plot_args[c]["label"]} with dt={dtplot:.4f}, dx={dx_LRES:.4f}: {cplot}')
                plt.plot(xf_LRES, cplot, label='AdImEx', color='purple', linestyle='-')            
            else:
                dtplot = dt_LRES_Ex
                cplot = dtplot*uf_LRES/dx_LRES # courant number for LRES
                logging.info(f'Velocity for {plot_args[c]["label"]} with dt={dtplot:.4f}, dx={dx_LRES:.4f}: {uf_LRES}')
                logging.info(f'Courant number for {plot_args[c]["label"]} with dt={dtplot:.4f}, dx={dx_LRES:.4f}: {cplot}')
                plt.plot(xf_LRES, cplot, label='Ex', color='darkturquoise', linestyle='-')
    plt.title('Courant number')
    plt.xlabel('x')
    plt.ylabel('$C$')
    plt.legend()
    plt.axhline(1, color='k', linestyle=':')
    plt.savefig(outputdir + 'courant.png')
    plt.close()




    for c in range(len(cases)):
        #if cases[c]['HRES'] == True:
        #    if cases[c]['AdImEx'] == True:
        #        dtplot = dt_HRES_AdImEx
        #        cplot = dtplot*uf_HRES/dx_HRES # courant number for HRES
        #        beta = np.maximum(0., 1. - 1./cplot) # beta for HRES
        #        plt.plot(xf_HRES, cplot, label='AdImEx', color='purple', linestyle='-')            
        #    else:
        #        dtplot = dt_HRES_Ex
        #        cplot = dtplot*uf_HRES/dx_HRES # courant number for HRES
        #        beta = np.maximum(0., 1. - 1./cplot) # beta for HRES
        #        plt.plot(xf_HRES, cplot, label='Ex', color='darkturquoise', linestyle='-')                
        if cases[c]['HRES'] == False:
            if cases[c]['AdImEx'] == True:
                dtplot = dt_LRES_AdImEx
                cplot = dtplot*uf_LRES/dx_LRES # courant number for LRES
                beta = np.maximum(0., 1. - 1./cplot) # beta for LRES
                logging.info(f'Off-centring for {plot_args[c]["label"]} with dt={dtplot:.4f}, dx={dx_LRES:.4f}: {beta}')
                plt.plot(xf_LRES, beta, label='AdImEx', color='purple', linestyle='-')            
            else:
                dtplot = dt_LRES_Ex
                cplot = dtplot*uf_LRES/dx_LRES # courant number for LRES
                beta = np.maximum(0., 1. - 1./cplot) # beta for LRES
                logging.info(f'Off-centring for {plot_args[c]["label"]} with dt={dtplot:.4f}, dx={dx_LRES:.4f}: {beta}')
                plt.plot(xf_LRES, beta, label='Ex', color='darkturquoise', linestyle='-')
    plt.title('Off-centering in time')
    plt.xlabel('x')
    plt.ylabel('$\\theta$')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    #plt.axhline(1, color='k', linestyle=':')
    plt.savefig(outputdir + 'offcentering.png')
    plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(xc_HRES, psi_in_HRES, linestyle='--', color='grey', label='Initial') # plot initial condition

    # Run schemes and plot the final time step for each scheme in the same plot and do experiments
    for c in range(len(cases)):
        s = plot_args[c]['label']
        if cases[c]['HRES'] == True:
            # Set up time step and number of time steps for HRES
            if cases[c]['AdImEx'] == True:
                nt_HRES = nt_HRES_AdImEx
                dt_HRES = dt_HRES_AdImEx
            else:
                nt_HRES = nt_HRES_Ex
                dt_HRES = dt_HRES_Ex

            # Run case
            locals()[f'psi_{s}_{l}'] = callscheme(cases[c], nt_HRES, dt_HRES, uf_HRES, dxc_HRES, psi_in_HRES, u_setting)

            # Plot final time step
            plt.plot(xc_HRES, locals()[f'psi_{s}_reg'][-1], **plot_args[c]) # !!! check whether -1 is correct - I want to make sure that LRES and HRES are plotting at the same point in time!

            # Do experiments
            locals()[f'csv_psi_{s}'] = epm.check_conservation(psi_in_HRES, locals()[f'psi_{s}_reg'][-1], dxc_HRES) # !!! same -> -1 correct?
            logging.info(f"{plot_args[c]['label']} - Mass gained: {locals()[f'csv_psi_{s}']:.4E}")
            locals()[f'bdn_psi_{s}'] = epm.check_boundedness(psi_in_HRES, locals()[f'psi_{s}_reg'][-1]) # !!! same -> -1 correct?
            logging.info(f"{plot_args[c]['label']} - Boundedness: {locals()[f'bdn_psi_{s}']}")         
            locals()[f'TV_psi_{s}'] = epm.totalvariation(locals()[f'psi_{s}_reg'][-1]) # !!! same -> -1 correct?
            logging.info(f"{plot_args[c]['label']} - Variation: {locals()[f'TV_psi_{s}']:.4E}")
            logging.info('')

        elif cases[c]['HRES'] == False:
            # Set up time step and number of time steps for LRES
            if cases[c]['AdImEx'] == True:
                nt_LRES = nt_LRES_AdImEx
                dt_LRES = dt_LRES_AdImEx
            else:
                nt_LRES = nt_LRES_Ex
                dt_LRES = dt_LRES_Ex

            # Run case
            locals()[f'psi_{s}_{l}'] = callscheme(cases[c], nt_LRES, dt_LRES, uf_LRES, dxc_LRES, psi_in_LRES, u_setting)

            # Plot final time step
            plt.plot(xc_LRES, locals()[f'psi_{s}_reg'][-1], **plot_args[c]) # !!! same -> -1 correct?

            # Do experiments
            locals()[f'csv_psi_{s}'] = epm.check_conservation(psi_in_LRES, locals()[f'psi_{s}_reg'][-1], dxc_LRES) # !!! same -> -1 correct?
            logging.info(f"{plot_args[c]['label']} - Mass gained: {locals()[f'csv_psi_{s}']:.4E}")
            locals()[f'bdn_psi_{s}'] = epm.check_boundedness(psi_in_LRES, locals()[f'psi_{s}_reg'][-1]) # !!! same -> -1 correct?
            logging.info(f"{plot_args[c]['label']} - Boundedness: {locals()[f'bdn_psi_{s}']}")         
            locals()[f'TV_psi_{s}'] = epm.totalvariation(locals()[f'psi_{s}_reg'][-1]) # !!! same -> -1 correct?
            logging.info(f"{plot_args[c]['label']} - Variation: {locals()[f'TV_psi_{s}']:.4E}")
            logging.info('')


    ##########################
    #### Plotting schemes ####
    ##########################

    ut.design_figure(plotname, f'$\\Psi$ at t={total_time} with $u$ varying', \
                     'x', '$\\Psi$', 0., xmax, True, -0.1, ymax)

    ###########################
    #### Create animations ####
    ###########################

    fields,  colors = [], []
    # Create animation from the data
    if create_animation == True:
        for c in range(len(cases)):        
            s = plot_args[c]['label']
            fields.append(locals()[f'psi_{s}_reg'])
        #anim.create_animation_from_HRESLRESdata(fields, len(schemenames), psi_in_HRES, nt_LRES, dt_LRES, xc_LRES, dtfactor_HRESLRES, xc_HRES, outputdir, plot_args, xmax, HRESbool, ymax=ymax)
        anim.create_animation_from_HRESLRES_ExAdImExdata(fields, len([case["scheme"] for case in cases]), psi_in_HRES, nt_LRES_AdImEx, dt_LRES_AdImEx, xc_LRES, dtfactor_HRESLRES, xc_HRES, outputdir, plot_args, xmax, HRESbool, dtfactor_ExAdImEx, AdImExbool, ymax=ymax)

    print('Done')
    logging.info('')
    logging.info('================================= Done ===================================')
    logging.info('')


def callscheme(case, nt, dt, uf, dxc, psi_in, u_setting, verbose=True):
    """Takes all the input variables and the scheme name and calls the scheme with the appropriate input arguments."""

    # Translate the scheme name to a function in schemes.py
    sc = case["scheme"]
    fn = getattr(sch, f'{sc}')

    # Remove 'scheme' key from dictionary
    exclude = {"scheme"}
    params = ut.without_keys(case, exclude)

    # Call the scheme
    if verbose == True: print(f'Running {sc} with parameters {params}')
    psi = fn(psi_in.copy(), nt, dt, uf, dxc, u_setting, **params) # check whether the scheme is called correctly!!! with the u_setting parameters etc
    if verbose == True:
        logging.info(f'Final psi for {sc} with parameters {params} and nx={len(psi_in)}: {psi[-1]}')
        logging.info('')
    return psi

    
if __name__ == "__main__": 
    starttime = timeit.default_timer()
    
    main()

    print('Total runtime: ', timeit.default_timer() - starttime)