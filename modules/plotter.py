# Class to plot results

import copy 
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from scipy.signal import argrelextrema
import seaborn as sb


class Plotter:

####################################################################

    def __init__(self):
    	self

####################################################################

    def flux_avg_2d(self, opts, flux, RRfis):

        nBins = opts.numBins
        nGrps = opts.numGroups
        N = nBins*nBins
        subBins = int(15/opts.delta)

        AltFlux = np.zeros([121*nGrps,subBins**2])
        subBinPower = np.zeros([121,subBins**2])
        subBinTemp = np.zeros([121,subBins**2])
        xcount = 0
        ycount = 0
        n = 0
        m = 0
        for g in range(nGrps):
            for y in range(nBins):
                for x in range(nBins):
                    i = g*N+x+y*nBins

                    if x*opts.delta < 15*(n+1) and y*opts.delta < 15*(m+1):
                        AltFlux[m*11+n+121*g, ycount*subBins+xcount] = flux[i]
                        if g == 0:
                            subBinPower[m*11+n, ycount*subBins+xcount] = RRfis[i]*opts.delta**2

                    xcount = xcount + 1
                    if xcount == subBins:
                        n = n + 1
                        xcount = 0
                n = 0
                ycount = ycount + 1
                if ycount == subBins:
                    m = m + 1
                    ycount = 0
            m = 0

        AvgFlux = np.zeros(121*nGrps)
        AvgPower = np.zeros(121)
        Norm = (sum(sum(subBinPower))/(78*15**2))
        for i in range(121*nGrps):
            AvgFlux[i] = np.average(AltFlux[i,:])
        for i in range(121):
            AvgPower[i] = (sum(subBinPower[i,:])/15**2)/Norm

        fluxG1 = np.reshape(AvgFlux[:121],(11,11))
        fluxG2 = np.reshape(AvgFlux[121:],(11,11))
        cell_labels = np.reshape(AvgPower,(11,11))
        axis_labels = range(0,165,15)

        heat_map1 = sb.heatmap(fluxG1, annot=cell_labels, annot_kws={"size": 5}, xticklabels=axis_labels, yticklabels=axis_labels) #vmin=0, vmax=0.5, annot=cell_labels, annot_kws={"size": 5}
        heat_map1.invert_yaxis()
        plt.yticks(rotation=0)
        plt.savefig("./G1flux_AVG")
        plt.clf()

        heat_map2 = sb.heatmap(fluxG2, xticklabels=axis_labels, yticklabels=axis_labels) #vmin=0, vmax=0.01, annot=cell_labels, annot_kws={"size": 5}
        heat_map2.invert_yaxis()
        plt.yticks(rotation=0)
        plt.savefig("./G2flux_AVG")
        plt.clf()

        np.savetxt("start_AssemblyPower.txt", cell_labels)

####################################################################

    def flux_fine_mesh_2d(self, opts, flux):

        nBins = opts.numBins
        nGrps = opts.numGroups
        N = nBins*nBins

        fluxG1 = np.reshape(flux[:N],(nBins,nBins))
        fluxG2 = np.reshape(flux[N:],(nBins,nBins))
        # cell_labels = np.reshape(Power,(nBins,nBins))
        axis_labels = range(0,int(opts.length),int(opts.length))

        heat_map1 = sb.heatmap(fluxG1, xticklabels=axis_labels, yticklabels=axis_labels) #vmin=0, vmax=0.5, annot=cell_labels, annot_kws={"size": 5}
        heat_map1.invert_yaxis()
        plt.yticks(rotation=0)
        plt.savefig("./G1flux_FineMesh")
        plt.clf()

        heat_map2 = sb.heatmap(fluxG2, xticklabels=axis_labels, yticklabels=axis_labels) #vmin=0, vmax=0.01, annot=cell_labels, annot_kws={"size": 5}
        heat_map2.invert_yaxis()
        plt.yticks(rotation=0)
        plt.savefig("./G2flux_FineMesh")
        plt.clf()

####################################################################

    def power_2d(self, opts, flux, XS, name='end', amp=None):

        nBins = opts.numBins
        nGrps = opts.numGroups
        N = nBins*nBins

        if amp:
            shape = flux[:]/sum(flux)
            bin_power = (XS.fis[:N]*shape[:N]*amp+XS.fis[N:]*shape[N:]*amp)*opts.delta**2
        else:
            bin_power = (XS.fis[:N]*flux[:N]+XS.fis[N:]*flux[N:])*opts.delta**2
        norm = sum(bin_power)/(78*15**2)
        assembly_power = np.reshape(bin_power[:]/15**2/norm,(nBins,nBins))

        axis_labels = range(0,int(opts.length),int(opts.delta))

        heat_map1 = sb.heatmap(assembly_power, annot=True, annot_kws={"size": 5}, xticklabels=axis_labels, yticklabels=axis_labels) #vmin=0, vmax=0.5, annot=cell_labels, annot_kws={"size": 5}
        heat_map1.invert_yaxis()
        plt.yticks(rotation=0)
        plt.savefig("./"+name+"_AssemblyPowers")
        plt.clf()

        np.savetxt(name+"_AssemblyPowers.txt", assembly_power)

####################################################################

    def temps_2d(self, opts, T, name='end'):

        nBins = opts.numBins
        temperatures = np.reshape(T,(nBins,nBins))

        axis_labels = range(0,int(opts.length),int(opts.delta))

        heat_map1 = sb.heatmap(temperatures, annot=True, annot_kws={"size": 5}, xticklabels=axis_labels, yticklabels=axis_labels) #vmin=0, vmax=0.5, annot=cell_labels, annot_kws={"size": 5}
        heat_map1.invert_yaxis()
        plt.yticks(rotation=0)
        plt.savefig("./"+name+"_Temperatures")
        plt.clf()

        np.savetxt(name+"_Temperatures.txt", temperatures)

####################################################################

    def precursors_2d(self, opts, C1, C2):

        nBins = opts.numBins
        Grp1 = np.reshape(C1,(nBins,nBins))
        Grp2 = np.reshape(C2,(nBins,nBins))

        axis_labels = range(0,int(opts.length),int(opts.delta))

        heat_map1 = sb.heatmap(Grp1, annot=True, annot_kws={"size": 5}, xticklabels=axis_labels, yticklabels=axis_labels) #vmin=0, vmax=0.5, annot=cell_labels, annot_kws={"size": 5}
        heat_map1.invert_yaxis()
        plt.yticks(rotation=0)
        plt.savefig("./3s_Grp1")
        plt.clf()

        heat_map2 = sb.heatmap(Grp2, annot=True, annot_kws={"size": 5}, xticklabels=axis_labels, yticklabels=axis_labels) #vmin=0, vmax=0.5, annot=cell_labels, annot_kws={"size": 5}
        heat_map2.invert_yaxis()
        plt.yticks(rotation=0)
        plt.savefig("./3s_Grp2")
        plt.clf()

        np.savetxt("3s_Precursors_1.txt", Grp1)
        np.savetxt("3s_Precursors_2.txt", Grp2)

####################################################################

    def frequencies_2d(self, opts, C1, C2, t):

        nBins = opts.numBins
        Grp1 = copy.copy(np.reshape(C1,(nBins,nBins)))
        Grp2 = copy.copy(np.reshape(C2,(nBins,nBins)))
        Grp1[Grp1 == 0] = np.nan
        Grp2[Grp2 == 0] = np.nan

        axis_labels = range(0,int(opts.length),int(opts.delta))

        heat_map1 = sb.heatmap(Grp1, annot=False, annot_kws={"size": 5}, xticklabels=axis_labels, yticklabels=axis_labels)#, annot=cell_labels, annot_kws={"size": 5}
        heat_map1.invert_yaxis()
        plt.yticks(rotation=0)
        name = "frequencies/Grp1/"+str(t)
        plt.savefig(name)
        plt.clf()

        heat_map2 = sb.heatmap(Grp2, annot=False, annot_kws={"size": 5}, xticklabels=axis_labels, yticklabels=axis_labels)# vmin=0, vmax=80, annot=cell_labels, annot_kws={"size": 5}
        heat_map2.invert_yaxis()
        plt.yticks(rotation=0)
        name = "frequencies/Grp2/"+str(t)
        plt.savefig(name)
        plt.clf()

####################################################################

    def matrix(self, opts, A, F):

        pl.spy(A)
        pl.savefig('./spy/spyA')
        pl.clf()

        pl.spy(F)
        pl.savefig('./spy/spyF')
        pl.clf()

####################################################################

    def power(self, t, core_power, log=False, ylim=False):

        np.savetxt("PowerChange.txt", core_power)

        plt.rcParams.update({'font.size': 15})
        if log:
            plt.yscale("log")
        if ylim:
            plt.ylim([0,1e-5])
        plt.plot(t, core_power)
        plt.xlabel('time [s]')
        plt.ylabel('Core Power [W/cc]')
        plt.tight_layout()
        plt.savefig('./PowerChange')
        plt.clf()

        print('Power at t=3.0s:', core_power[-1])

        maximum = argrelextrema(np.array(core_power), np.greater)
        maxpower1 = core_power[maximum[0][0]]
        time1 = t[maximum[0][0]]
        print('Power at first peak [W/cc]:', maxpower1)
        print('Time to first peak [s]:', time1)

        for i in range(len(maximum[0])):
            print("Power: ", core_power[maximum[0][i]], " at time: ", t[maximum[0][i]])

        # if len(maximum[0]) > 1:
        #     maxpower2 = core_power[maximum[0][1]]
        #     print('Power at second peak [W/cc]:', maxpower2)
            # time2 = t[maximum[0][1]]
            # for i in range(len(maximum[0])):
            #   print(core_power[maximum[0][i]], t[maximum[0][i]])

####################################################################

    def reactivity(self, t, rho, log=False, ylim=False):

        np.savetxt("Reactivity.txt", rho)

        plt.plot(t, rho)
        plt.xlabel('time [s]')
        plt.ylabel('Reactivity')
        plt.tight_layout()
        plt.savefig('./Reactivity')
        plt.clf()

####################################################################

    def frequencies(self, t_outer, wp, wd1=None, wd2=None, log=False, ylim=False, iqs=False):

        np.savetxt("PromptFrequencies.txt", wp)

        plt.plot(t_outer[1:], wp)
        plt.xlabel('time [s]')
        if iqs == False:
            plt.ylabel('Prompt frequency')
        else:
            plt.ylabel('Amplitude derivative term')
        plt.tight_layout()
        plt.savefig('./Prompt')
        plt.clf()

        if wd1 is not None and wd2 is not None:

            np.savetxt("DelayedFrequenciesGrp1.txt", wd1)
            np.savetxt("DelayedFrequenciesGrp2.txt", wd2)

            plt.plot(t_outer[1:], wd1, label='Grp 1')
            plt.plot(t_outer[1:], wd2, label='Grp 2')
            plt.xlabel('time [s]')
            plt.ylabel('Delayed frequency')
            plt.tight_layout()
            plt.legend()
            plt.savefig('./Delayed')
            plt.clf()

####################################################################

    def frequencies_ftm(self, t, wp1, wp2, wd1=None, wd2=None, log=False, ylim=False):

        np.savetxt("PromptFrequenciesGrp1.txt", wp1)
        np.savetxt("PromptFrequenciesGrp2.txt", wp2)
        
        plt.plot(t[1:], wp1, label='Grp 1')
        plt.plot(t[1:], wp2, label='Grp 2')
        plt.xlabel('time [s]')
        plt.ylabel('Prompt frequency')
        plt.tight_layout()
        plt.legend()
        plt.savefig('./Prompt')
        plt.clf()

        if wd1 is not None and wd2 is not None:

            np.savetxt("DelayedFrequenciesGrp1.txt", wd1)
            np.savetxt("DelayedFrequenciesGrp2.txt", wd2)

            plt.plot(t[1:], wd1, label='Grp 1')
            plt.plot(t[1:], wd2, label='Grp 2')
            plt.xlabel('time [s]')
            plt.ylabel('Delayed frequency')
            plt.tight_layout()
            plt.legend()
            plt.savefig('./Delayed')
            plt.clf()

####################################################################

    def kbalance(self, t_outer, kbalance, log=False, ylim=False):

        np.savetxt("Kbalance.txt", kbalance)

        if ylim:
            plt.ylim([0.99,1.01])
        plt.plot(t_outer[1:], kbalance)
        plt.xlabel('time [s]')
        plt.ylabel('balance k')
        plt.tight_layout()
        plt.savefig('./kbalance')
        plt.clf()

####################################################################

    def a(self, t_outer, a, log=False, ylim=False):

        np.savetxt("Alpha.txt", a)

        if ylim:
            plt.ylim([0.99,1.01])
        plt.plot(t_outer[1:], a)
        plt.xlabel('time [s]')
        plt.ylabel('alpha/v')
        plt.tight_layout()
        plt.savefig('./alpha')
        plt.clf()

####################################################################

    def iterations(self, t_outer, iterations, log=False, ylim=False):

        np.savetxt("Iterations.txt", iterations)

        plt.plot(t_outer, iterations)
        plt.xlabel('time [s]')
        plt.ylabel('Iterations')
        plt.tight_layout()
        plt.savefig('./Iterations')
        plt.clf()

####################################################################

    def implicit_iterations(self, t, iterations, log=False, ylim=False):

        np.savetxt("Iterations.txt", iterations)

        plt.plot(t, iterations)
        plt.xlabel('time [s]')
        plt.ylabel('Iterations')
        plt.tight_layout()
        plt.savefig('./Iterations')
        plt.clf()

####################################################################

    def omega_change(self, inner_omegas, count_outer, options):

        plt.plot(inner_omegas)
        plt.xlabel('Inner iterations')
        plt.ylabel('Omega')
        plt.tight_layout()
        plt.savefig('OmegaChange at '+str(count_outer*options.dt))
        plt.clf()

#end class 