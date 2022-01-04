# This code solves the discrete 2D diffusion equations for the LRA problem
# in a coarse time stepping scheme with an analytic solution to the precursor equations
# and optional repeated time steps 

import time as runtime

from cross_sections import *
from solver_td import *

from modules.construct import *
from modules.diff_opts import *
from modules.getfilename import *
from modules.material import *
from modules.pke import *
from modules.plotter import *
from modules.solver_ss import *


start = runtime.time()
fn = FileName()
fn.get_filename()

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
    time = 0  # Time in seconds
    
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
            sol.point_jacobi(options, Matrix.D, Matrix.Otrans, Matrix.Ftrans, XS)
            AdjFlux = copy.copy(sol.flux)
            sol.point_jacobi(options, Matrix.D, Matrix.O, Matrix.F, XS)
            print("Source iterations:", sol.source_it)
            print("Flux iterations:", sol.flux_it)
        else:
            sol.gauss_seidel(options, Matrix.D, Matrix.Otrans, Matrix.Ftrans, XS)
            AdjFlux = copy.copy(sol.flux)
            sol.gauss_seidel(options, Matrix.D, Matrix.O, Matrix.F, XS)
            print("Source iterations:", sol.source_it)
            print("Source iterations:", sol.flux_it)
        print("Eigenvalue:", sol.k)
        
        # Compute inital values
        InitFlux = copy.copy(sol.flux)
        Flux = copy.copy(sol.flux)
        kcrit = copy.copy(sol.k)
        T = np.zeros(N)
        T[:] = 300.0
        RRfis = np.zeros(N)
        RRfis[:] = XS.fis[:N]*InitFlux[:N] + XS.fis[N:]*InitFlux[N:]
        PrecursorFactor = XS.Bi[:]/(XS.DECAYi[:]*kcrit)
        C = np.zeros([2,N])
        for i in range(2):
            C[i,:] = PrecursorFactor[i]*RRfis

    ############################
    #    Transient problem     #
    ############################
    elif options.pb_num == 2:

        # Initalize variables
        alpha = 0.0  # Factor of material change for transient implementation
        n_steps = int(options.dt/options.subStep)
        t = []
        # t = np.linspace(0, options.t_end, 
        #     int((options.t_end+options.subStep)/options.subStep), endpoint=True)
        t_outer = np.linspace(0, options.t_end, 
            int((options.t_end+options.dt)/options.dt), endpoint=True)
        stored_Power = []
        # stored_Power = np.zeros(len(t))
        # stored_Power[0] = options.ReactorPower
        wp = 0

        Amplitude = PKE(options, XS, InitFlux, kcrit, T, C_int=C, rho=0)
        Amplitude.outer_parameters(options, InitFlux, AdjFlux, Matrix, XS, kcrit, time)

        # Loop through time
        CountOuter = 0
        while time < options.t_end: 

            for r in range(options.repeat_inner+1):

                if time == 0:
                    Doppler = np.zeros([N])
                    stored_Power.append(options.ReactorPower)
                    T = copy.copy(T)
                else:
                    Doppler = sol.DopplerFactor
                    stored_Power.append(sol.Power)
                    T = copy.copy(sol.T)

                # Only store values of rho, frequencies during the last repeition
                if r == options.repeat_inner:
                    store_rho = True
                    store_w = True
                else:
                    store_rho = False
                    store_w = False

                # First iteration: save the starting conditions
                if r == 0:
                    StartTime = copy.copy(time)
                    StartAdj = copy.copy(AdjFlux)
                    StartFlux = copy.copy(sol.flux)
                    StartDoppler = copy.copy(Amplitude.DopplerFactor)
                    StartTemps = copy.copy(Amplitude.T)
                    InnerEnd = round(StartTime+options.dt,6)
                    StartC = copy.copy(C)

                # All other iterations: revert to starting conditions
                else:
                    time = copy.copy(StartTime)
                    AdjFlux = copy.copy(StartAdj)

                CountInner = 0
                while time < InnerEnd:

                    time = round(time+options.subStep,6)
                    if time <= 2.00000001:
                        alpha = time
                    else:
                        alpha = 2.0

                    if CountInner == 0:
                        XS.update(options, M.data, alpha, StartDoppler, kcrit)
                        Amplitude.outer_parameters(options, StartFlux, AdjFlux, Matrix, XS, kcrit, time, C_int=StartC)
                        Amplitude.inner_parameters(options, AdjFlux, XS, kcrit, stored_rho=store_rho)
                        if r == 0:
                            StartConditions = copy.copy(Amplitude.B)
                        Amplitude.solve(options, XS, T=StartTemps, B=StartConditions)
                    else:
                        if r == 0:
                            EndAdj  = copy.copy(StartAdj)
                            EndFlux = copy.copy(StartFlux)
                        AdjFlux[:] = StartAdj[:] + (time-StartTime)*(EndAdj[:]-StartAdj[:])/(InnerEnd-StartTime)
                        Flux[:] = StartFlux[:] + (time-StartTime)*(EndFlux[:]-StartFlux[:])/(InnerEnd-StartTime)
                        
                        XS.update(options, M.data, alpha, Amplitude.DopplerFactor, kcrit)
                        Amplitude.outer_parameters(options, Flux, AdjFlux, Matrix, XS, kcrit, time)
                        Amplitude.inner_parameters(options, AdjFlux, XS, kcrit, stored_rho=store_rho)
                        Amplitude.solve(options, XS, T=Amplitude.T, B=Amplitude.B)

                    # stored_Power[int(CountOuter*n_steps+CountInner)+1] = Amplitude.Power
                    CountInner += 1

                # Solve implicitly for shape update at t+1 with w
                t.append(time)
                XS.update(options, M.data, alpha, Doppler, kcrit, wp, wp, 0, 0)
                Matrix.construct_td(options, XS.D, XS.Premoval, XS.Gscat, XS.Pfis)
                sol = SolveTD(options, Flux, C, kcrit, T)
                sol.gauss_seidel(options, Matrix.D, Matrix.O, XS, Flux, time, 0, 0, wp)
                LastEndFlux = copy.copy(EndFlux)
                EndFlux = copy.copy(sol.flux)
                ERR = max(abs(LastEndFlux[:]-EndFlux[:]))
                C = copy.copy(sol.C)
                CountInner = 0
                print(time)

            CountOuter += 1
            
        results = Plotter()
        results.power(t, stored_Power, log=True)
        # results.reactivity(t, Amplitude.stored_rho)
        # results.power_2d(options, sol.flux, XS)
        # results.temps_2d(options, Amplitude.T)
        # results.precursors_2d(options, sol.C[0,:], sol.C[1,:])

    else:
        print("ERROR: please specify problem 1 or 2 in the input file. \
Problem 1 is the steady state version and problem 2 is the transient")

end = runtime.time()
print('Runtime:', end-start)