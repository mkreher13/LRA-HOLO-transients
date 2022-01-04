# This code solves the discrete 2D diffusion equations for the LRA problem
# in a fully implicit scheme with an analytic solution to the precursor equations

import time as runtime

from cross_sections import *
from solver_td import *

from modules.construct import *
from modules.diff_opts import *
from modules.getfilename import *
from modules.material import *
from modules.plotter import *
from modules.solver_ss import *


start = runtime.time()
fn = FileName()
fn.get_filename()
stored_Power = []

results = Plotter()
M = Material()
M.read()
Matrix = Construct()

for f in fn.listfn:

    print(f)
    options = DiffusionOpts1D()
    options.read(f)
    nGrps = options.numGroups
    nBins = options.numBins
    N = nBins*nBins
    
    ############################
    #    Steady-state problem  #
    ############################
    if options.pb_num == 1:

        # Problem setup 
        XS = CrossSections(options)
        XS.build(options, M.data)
        Matrix.construct_ss(options, XS.D, XS.removal, XS.Gscat, XS.fis)
        sol = SolveSS(options)

        # Solver
        if options.method == 'PointJacobi':
            sol.point_jacobi(options, Matrix.D, Matrix.O, Matrix.F, XS)
            print("Source iterations:", sol.source_it)
            print("Flux iterations:", sol.flux_it)
        else:
            sol.gauss_seidel(options, Matrix.D, Matrix.O, Matrix.F, XS)
            print("Source iterations:", sol.source_it)
            print("Source iterations:", sol.flux_it)
        print("Eigenvalue:", sol.k)
        
        # Compute inital values
        InitFlux = copy.copy(sol.flux)
        kcrit = copy.copy(sol.k)
        InitT = np.zeros(N)
        InitT[:] = 300.0
        RRfis = np.zeros(N)
        RRfis[:] = XS.fis[:N]*InitFlux[:N] + XS.fis[N:]*InitFlux[N:]
        PrecursorFactor = XS.Bi[:]/(XS.DECAYi[:]*kcrit)
        InitC = np.zeros([2,N])
        for i in range(2):
            InitC[i,:] = PrecursorFactor[i]*RRfis

        # Plot results
        # results = Plotter()
        # results.flux_fine_mesh_2d(options, InitFlux)
        # results.flux_avg_2d(options, InitFlux, RRfis)

    ############################
    #    Transient problem     #
    ############################
    elif options.pb_num == 2:

        # Initalize variables
        alpha = 0.0  # Factor of material change for transient implementation
        time = 0     # Time in seconds
        t = []       # Stored time
        stored_iterations = []

        # Loop through time
        while time < options.t_end: 

            # Get initial conditions
            if time == 0:
                flux = copy.copy(InitFlux)
                C = copy.copy(InitC)
                T = copy.copy(InitT)
                Doppler = np.zeros([N])
                stored_Power.append(options.ReactorPower)
                t.append(time)
                XS.update(options, M.data, alpha, Doppler, kcrit)
                Matrix.construct_td(options, XS.D, XS.Premoval, XS.Gscat, XS.Pfis)
            else:
                flux = copy.copy(sol.flux)
                C = copy.copy(sol.C) 
                T = copy.copy(sol.T)
                Doppler = sol.DopplerFactor

            # Carry out transient
            time = round(time+options.dt,6)
            if time <= 2.000000001:
                alpha = time
            else:
                alpha = 2.0

            XS.update(options, M.data, alpha, Doppler, kcrit)
            Matrix.update_td(options, XS)
            sol = SolveTD(options, flux, C, kcrit, T)
            sol.gauss_seidel(options, Matrix.D, Matrix.O, XS, flux, time)

            # Update values
            stored_Power.append(sol.Power)
            stored_iterations.append(sol.iterations)
            t.append(time)
            print(time)

            if time == 1.44545:
                results.power_2d(options, sol.flux, XS, name='peak')
                results.temps_2d(options, sol.T, name='peak')

        results.power(t, stored_Power, log=True)
        results.power_2d(options, sol.flux, XS, name='end')
        results.temps_2d(options, sol.T, name='end')
        results.precursors_2d(options, sol.C[0,:], sol.C[1,:])
        results.implicit_iterations(t[1:], stored_iterations)

    else:
        print("ERROR: please specify problem 1 or 2 in the input file. \
Problem 1 is the steady state version and problem 2 is the transient")

end = runtime.time()
print('Runtime:', end-start)