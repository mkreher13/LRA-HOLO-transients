# This code solves the discrete 2D diffusion equations for the LRA problem
# in a fixed shape scheme

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

M = Material()
M.read()

for f in fn.listfn:

    print(f)
    options = DiffusionOpts1D()
    options.read(f)
    nGrps = options.numGroups
    nBins = options.numBins
    N = nBins*nBins
    time = 0  # seconds

    ############################
    #      Adjoint problem     #
    ############################
    if options.pb_num == 1:

        # Problem setup 
        XS = CrossSections(options)
        XS.build(options, M.data)
        Matrix = Construct()
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

        # Compute inital values: InitFlux, InitC, kcrit
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

    ############################
    #    Transient problem     #
    ###########################
    elif options.pb_num == 2:

        # Initalize variables
        alpha = 0.0        # Factor of material change for transient implementation
        t = []             # Stored time
        stored_Power = []  # Stored core power
        Amplitude = PKE(options, XS, InitFlux, kcrit, InitT, C_int=InitC)
        Amplitude.outer_parameters(options, InitFlux, AdjFlux, Matrix, XS, kcrit, time)

        # Loop through time
        while time < options.t_end: 
            time = round(time+options.subStep,6)
            t.append(time)
            print(time)

            if time <= 2.00000001:
                alpha = time
            else:
                alpha = 2.0

            XS.update(options, M.data, alpha, Amplitude.DopplerFactor)
            Amplitude.inner_parameters(options, AdjFlux, XS, kcrit, stored_rho=True)
            Amplitude.solve(options, XS)
            stored_Power.append(Amplitude.Power)

        results = Plotter()
        results.power(t, stored_Power, log=True)
        results.reactivity(t, Amplitude.stored_rho)
        results.power_2d(options, sol.flux, XS)
        results.temps_2d(options, Amplitude.T)
        results.precursors_2d(options, Amplitude.C_int[0,:], Amplitude.C_int[1,:])

    else:
        print("ERROR: please specify problem 1 or 2 in the input file. \
Problem 1 is the steady state version and problem 2 is the transient")

end = runtime.time()
print('Runtime:', end-start)