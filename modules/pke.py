# class to solve amplitude by PKEs

import scipy.linalg as la
import numpy as np
import copy

from modules.plotter import *

class PKE:

###############################################################
	
	def __init__(self, opts, XS, flux, kcrit, InitT, C_int=None, rho=None):
		
		N = opts.numBins*opts.numBins
		rank = N*opts.numGroups
		self.subBinPower = np.zeros(N)
		self.stored_power = []

		# Initalize production matrix
		nRows = len(XS.Bi)+1
		self.A0 = np.zeros([nRows, nRows])
		for i in range(1,nRows):
			self.A0[i,0] = XS.Bi[i-1]
			self.A0[i,i] = -XS.DECAYi[i-1]
			self.A0[0,i] = XS.DECAYi[i-1]

		# Initalize power and precursor matrix
		self.B = np.zeros(nRows)
		self.B[0] = sum(flux)
		for i in range(1,nRows):
			self.B[i] = XS.Bi[i-1]*self.B[0]/XS.DECAYi[i-1]
		# print(self.B)

		# Temperature variables
		self.dTdt = np.zeros(N)
		self.T = copy.copy(InitT)
		self.DopplerFactor = np.zeros(N)

		# Storing rho for plotting
		self.stored_rho = []
		if rho is not None:
			self.stored_rho.append(rho)

		# Spatial precursors
		self.C_int = copy.copy(C_int)
		self.wd1 = np.zeros(N)
		self.wd2 = np.zeros(N)
		self.wp = 0.0
		self.stored_wd1 = []
		self.stored_wd2 = []
		self.stored_wp = []

###############################################################

	def outer_parameters(self, opts, flux, adjoint, Matrix, XS, kcrit, time, C_int=None):

		delta = opts.delta
		nBins = opts.numBins
		N = nBins*nBins
		self.Shape = flux[:]/sum(flux)

		if C_int is not None:
			self.C_int = copy.copy(C_int)

		# prompt neutron lifetime
		RRshape = XS.fis[:N]*self.Shape[:N] + XS.fis[N:]*self.Shape[N:]
		ShapeIntegral = adjoint[:N]*1/XS.VEL1*self.Shape[:N]+adjoint[N:]*1/XS.VEL2*self.Shape[N:]
		# self.PNL = sum(ShapeIntegral[:])/sum(adjoint[:N]*RRshape[:])*kcrit
		self.PNL = sum(ShapeIntegral[:])/sum(adjoint[:N]*RRshape[:])/kcrit
		if time == 0:
			self.B[1:] = self.B[1:]/self.PNL

		self.J = np.zeros(N*opts.numGroups)
		self.J[:] = -(Matrix.dLeft[:]+Matrix.dRight[:]+Matrix.dUp[:]+Matrix.dDown[:])*self.Shape[:]

		for y in range(nBins):
			for x in range(nBins):
				i = x+y*nBins

				if x == 0:
					if y == 0:
						self.J[i] = self.J[i] + Matrix.dRight[i]*self.Shape[i+1] + Matrix.dUp[i]*self.Shape[i+nBins]
						self.J[i+N] = self.J[i+N] + Matrix.dRight[i+N]*self.Shape[i+1+N] + Matrix.dUp[i+N]*self.Shape[i+nBins+N]
					elif y == (nBins-1):
						self.J[i] = self.J[i] + Matrix.dRight[i]*self.Shape[i+1] + Matrix.dDown[i]*self.Shape[i-nBins]
						self.J[i+N] = self.J[i+N] + Matrix.dRight[i+N]*self.Shape[i+1+N] + Matrix.dDown[i+N]*self.Shape[i-nBins+N]
					else:
						self.J[i] = self.J[i] + Matrix.dRight[i]*self.Shape[i+1] + Matrix.dUp[i]*self.Shape[i+nBins] + Matrix.dDown[i]*self.Shape[i-nBins]
						self.J[i+N] = self.J[i+N] + Matrix.dRight[i+N]*self.Shape[i+1+N] + Matrix.dUp[i+N]*self.Shape[i+nBins+N] + Matrix.dDown[i+N]*self.Shape[i-nBins+N]

				elif x == (nBins-1):
					if y == 0:
						self.J[i] = self.J[i] + Matrix.dLeft[i]*self.Shape[i-1] + Matrix.dUp[i]*self.Shape[i+nBins]
						self.J[i+N] = self.J[i+N] + Matrix.dLeft[i+N]*self.Shape[i-1+N] + Matrix.dUp[i+N]*self.Shape[i+nBins+N]
					elif y == (nBins-1):
						self.J[i] = self.J[i] + Matrix.dLeft[i]*self.Shape[i-1] + Matrix.dDown[i]*self.Shape[i-nBins]
						self.J[i+N] = self.J[i+N] + Matrix.dLeft[i+N]*self.Shape[i-1+N] + Matrix.dDown[i+N]*self.Shape[i-nBins+N]
					else:
						self.J[i] = self.J[i] + Matrix.dLeft[i]*self.Shape[i-1] + Matrix.dUp[i]*self.Shape[i+nBins] + Matrix.dDown[i]*self.Shape[i-nBins]
						self.J[i+N] = self.J[i+N] + Matrix.dLeft[i+N]*self.Shape[i-1+N] + Matrix.dUp[i+N]*self.Shape[i+nBins+N] + Matrix.dDown[i+N]*self.Shape[i-nBins+N]

				elif y == 0:
					self.J[i] = self.J[i] + Matrix.dLeft[i]*self.Shape[i-1] + Matrix.dRight[i]*self.Shape[i+1] + Matrix.dUp[i]*self.Shape[i+nBins]
					self.J[i+N] = self.J[i+N] + Matrix.dLeft[i+N]*self.Shape[i-1+N] + Matrix.dRight[i+N]*self.Shape[i+1+N] + Matrix.dUp[i+N]*self.Shape[i+nBins+N]

				elif y == (nBins-1):
					self.J[i] = self.J[i] + Matrix.dLeft[i]*self.Shape[i-1] + Matrix.dRight[i]*self.Shape[i+1] + Matrix.dDown[i]*self.Shape[i-nBins]
					self.J[i+N] = self.J[i+N] +Matrix.dLeft[i+N]*self.Shape[i-1+N] + Matrix.dRight[i+N]*self.Shape[i+1+N] + Matrix.dDown[i+N]*self.Shape[i-nBins+N]

				else:
					self.J[i] = self.J[i] + Matrix.dLeft[i]*self.Shape[i-1] + Matrix.dRight[i]*self.Shape[i+1] + Matrix.dUp[i]*self.Shape[i+nBins] + Matrix.dDown[i]*self.Shape[i-nBins]
					self.J[i+N] = self.J[i+N] + Matrix.dLeft[i+N]*self.Shape[i-1+N] + Matrix.dRight[i+N]*self.Shape[i+1+N] + Matrix.dUp[i+N]*self.Shape[i+nBins+N] + Matrix.dDown[i+N]*self.Shape[i-nBins+N]

###############################################################

	def inner_parameters(self, opts, adjoint, XS, kcrit, stored_rho=False):

		delta = opts.delta
		nBins = opts.numBins
		N = nBins*nBins

		# Calculate rho
		integral1 = np.zeros(N)
		integral2 = np.zeros(N)
		integral3 = np.zeros(N)

		integral1[:] = (self.J[:N]-np.array(XS.removal[:N])*delta*self.Shape[:N]
			+np.array(XS.fis[:N]*self.Shape[:N])*delta/kcrit
			+np.array(XS.fis[N:]*self.Shape[N:])*delta/kcrit)
		integral2[:] = (self.J[N:]-np.array(XS.removal[N:])*delta*self.Shape[N:]
			+XS.Gscat[:,1]*self.Shape[:N]*delta)
		integral3[:] = (XS.fis[:N]*self.Shape[:N]
			+XS.fis[N:]*self.Shape[N:])*delta/kcrit

		self.rho = (sum(adjoint[:N]*integral1)+sum(adjoint[N:]*integral2))/sum(adjoint[:N]*integral3)/XS.B_TOT
		# self.rho = 0
		if stored_rho:
			self.stored_rho.append(self.rho)

###############################################################
		
	def solve(self, opts, XS, kcrit=1, T=None, B=None): 

		N = opts.numBins*opts.numBins
		KAPPA = 3.204e-11 # [J/fission]
		ALPHA = 3.83e-11  # [K cm3]
		GAMMA = 3.034e-3  # [K^(-1/2)]
		# self.rho = 0
	
		# Update production matrix
		self.A = copy.copy(self.A0)
		self.A0[0,0] = (self.rho*XS.B_TOT-XS.B_TOT)
		self.A[:,0]  = self.A0[:,0]/self.PNL

		# Solve
		EXPM = la.expm(self.A*opts.subStep)
		if B is not None:
			self.B = np.matmul(EXPM,B)
		else:
			self.B = np.matmul(EXPM,self.B)
		# print('Amplitude:', self.B[0])

		# Update parameters for feedback and plotting
		RRfis = (XS.fis[:N]*self.Shape[:N]*self.B[0]
			+XS.fis[N:]*self.Shape[N:]*self.B[0])
		self.subBinPower[:] = RRfis*KAPPA*(opts.delta**2)/2.43
		self.Power = sum(self.subBinPower)/(78*15**2)
		self.stored_power.append(self.Power)

		self.dTdt[:] = RRfis*ALPHA/2.43
		if T is not None:
			self.T[:] = T[:] + self.dTdt[:]*opts.subStep
		else:
			self.T[:] = self.T[:] + self.dTdt[:]*opts.subStep
		self.DopplerFactor[:] = GAMMA*(np.sqrt(self.T[:])-np.sqrt(300))

		# Update precursor integral
		self.C_int_prev = copy.copy(self.C_int)
		if self.C_int is not None:
			for i in range(2):
				self.C_int[i,:] = self.C_int[i,:] + (-XS.DECAYi[i]*self.C_int[i,:] + 
					XS.Bi[i]/kcrit*RRfis[:])*opts.subStep

###############################################################

	def calc_frequencies(self, opts, count, store=False, plot=False):

		N = opts.numBins*opts.numBins
		self.wp = np.log(self.stored_power[-1]/self.stored_power[-2])/opts.subStep
		for i in range(N):
			if self.C_int_prev[0,i] != 0:
				self.wd1[i] = np.log(self.C_int[0,i]/self.C_int_prev[0,i])/opts.subStep
				self.wd2[i] = np.log(self.C_int[1,i]/self.C_int_prev[1,i])/opts.subStep

		if store:
			self.stored_wp.append(self.wp)
			self.stored_wd1.append(self.wd1[0])
			self.stored_wd2.append(self.wd2[0])

		if plot:
			results = Plotter()
			results.frequencies_2d(opts, self.wd1, self.wd2, count)

#end class