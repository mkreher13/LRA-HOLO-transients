# This code solves the discrete 2D diffusion equations for the LRA problem
# in an omega-mode scheme with optional repeated time steps

import time as runtime

from cross_sections import *

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

results = Plotter()
M = Material()
M.read()

for f in fn.listfn:

    print(f)
    options = DiffusionOpts1D()
    options.read(f)
    nGrps = options.numGroups
    nBins = options.numBins
    N = nBins*nBins
    time = 0  # Time in seconds
    
    ############################
    #   Steady-State problem   #
    ############################
    if options.pb_num == 1:

        # Problem setup 
        XS = CrossSections(options)
        XS.build(options, M.data)
        Matrix = Construct()
        Matrix.construct_ss(options, XS.D, XS.Premoval, XS.Gscat, XS.Pfis)
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

        # Compute inital values: InitFlux, InitC, kcrit
        InitFlux = copy.copy(sol.flux)
        Flux = copy.copy(sol.flux)
        kcrit = copy.copy(sol.k)
        InitT = np.zeros(N)
        InitT[:] = 300.0
        RRfis = np.zeros(N)
        RRfis[:] = XS.fis[:N]*InitFlux[:N] + XS.fis[N:]*InitFlux[N:]
        PrecursorFactor = XS.Bi[:]/(XS.DECAYi[:]*kcrit)
        InitC = np.zeros([2,N])
        for i in range(2):
            InitC[i,:] = PrecursorFactor[i]*RRfis

    ############################
    #    Transient problem     #
    ###########################
    elif options.pb_num == 2:

        # Initalize variables
        alpha = 0.0  # Factor of material change for transient implementation
        n_steps = int(options.dt/options.subStep)
        t = np.linspace(0, options.t_end, 
            int((options.t_end+options.subStep)/options.subStep), endpoint=True)
        t_outer = np.linspace(0, options.t_end, 
            int((options.t_end+options.dt)/options.dt), endpoint=True)
        stored_Power = np.zeros(len(t))
        stored_Power[0] = options.ReactorPower
        stored_kbalance = np.zeros(len(t_outer))
        stored_kbalance[0] = 1.0

        Amplitude = PKE(options, XS, InitFlux, kcrit, InitT, C_int=InitC, rho=0)
        Amplitude.outer_parameters(options, InitFlux, AdjFlux, Matrix, XS, kcrit, time)

        # Loop through time
        CountOuter = 0
        CountInner = 0
        while time < options.t_end:

            for r in range(options.repeat_inner+1):

                # Only store values of rho, frequencies during last repetition 
                if r == options.repeat_inner:
                    store_rho = False
                    store_w = False
                    plot_wd = False
                else:
                    store_rho = False
                    store_w = False
                    plot_wd = False

                # First iteration: save the starting conditions
                if r == 0:
                    StartTime = copy.copy(time)
                    StartAdj = copy.copy(AdjFlux)
                    StartFlux = copy.copy(sol.flux)
                    StartDoppler = copy.copy(Amplitude.DopplerFactor)
                    StartTemps = copy.copy(Amplitude.T)
                    InnerEnd = round(StartTime+options.dt,6)
                    StartC = copy.copy(Amplitude.C_int)

                # All other iterations: revert to starting conditions
                else:
                    time = copy.copy(StartTime)
                    AdjFlux = copy.copy(StartAdj)

                # Progress in time along the inner time steps
                CountInner = 0
                while time < InnerEnd:

                    time = round(time+options.subStep,6)
                    if time <= 2.00000001:
                        alpha = time
                    else:
                        alpha = 2.0

                    # First inner time step must use starting conditions
                    if CountInner == 0:
                        XS.update(options, M.data, alpha, StartDoppler)
                        Amplitude.outer_parameters(options, StartFlux, AdjFlux, Matrix, XS, kcrit, time, C_int=StartC)
                        Amplitude.inner_parameters(options, AdjFlux, XS, kcrit, stored_rho=store_rho)
                        if r == 0:
                            StartConditions = copy.copy(Amplitude.B)
                        Amplitude.solve(options, XS, T=StartTemps, B=StartConditions)

                    # Subsequent inner time steps proceed naturally
                    else:
                        if r == 0:
                            EndAdj  = copy.copy(StartAdj)
                            EndFlux = copy.copy(StartFlux)
                        AdjFlux[:] = StartAdj[:] + (time-StartTime)*(EndAdj[:]-StartAdj[:])/(InnerEnd-StartTime)
                        Flux[:] = StartFlux[:] + (time-StartTime)*(EndFlux[:]-StartFlux[:])/(InnerEnd-StartTime)

                        XS.update(options, M.data, alpha, Amplitude.DopplerFactor)
                        Amplitude.outer_parameters(options, Flux, AdjFlux, Matrix, XS, kcrit, time)
                        Amplitude.inner_parameters(options, AdjFlux, XS, kcrit, stored_rho=store_rho)
                        Amplitude.solve(options, XS, T=Amplitude.T, B=Amplitude.B)

                    stored_Power[int(CountOuter*n_steps+CountInner)+1] = Amplitude.Power
                    CountInner += 1

                    # if time == 1.4451 and r == options.repeat_inner:
                    #     results.power_2d(options, Flux, XS, name='peak', amp=Amplitude.B[0])
                    #     results.temps_2d(options, Amplitude.T, name='peak')

                Amplitude.calc_frequencies(options, CountOuter, store=store_w, plot=plot_wd)
                Matrix.update_spatial_freq(options, XS, kcrit, Amplitude.wp, Amplitude.wd1, Amplitude.wd2)
                sol.gauss_seidel(options, Matrix.D, Matrix.Otrans, Matrix.Ftrans, XS)
                AdjFlux = copy.copy(sol.flux)
                sol.gauss_seidel(options, Matrix.D, Matrix.O, Matrix.F, XS)
                stored_kbalance[CountOuter] = sol.k
                LastEndFlux = copy.copy(EndFlux)
                EndAdj = copy.copy(AdjFlux)
                EndFlux = copy.copy(sol.flux)
                ERR = max(abs(LastEndFlux[:]-EndFlux[:]))
                CountInner = 0
                print(time)

            CountOuter += 1

        results.power(t, stored_Power, log=True)
        results.reactivity(t, Amplitude.stored_rho)
        results.kbalance(t_outer, stored_kbalance[:-1], ylim=True)
        results.frequencies(t_outer, Amplitude.stored_wp, Amplitude.stored_wd1, Amplitude.stored_wd2)
        results.power_2d(options, sol.flux, XS, name='end', amp=Amplitude.B[0])
        results.temps_2d(options, Amplitude.T, name='end')
        results.precursors_2d(options, Amplitude.C_int[0,:], Amplitude.C_int[1,:])

    else:
        print("ERROR: please specify problem 1 or 2 in the input file. \
            Problem 1 is the steady state version and problem 2 is the transient")

end = runtime.time()
print('Runtime:', end-start)