# Class to fill cross section vectors

import copy
import numpy as np


class CrossSections:

###############################################################

	def __init__(self, opts, kcrit=1.0):

		# Delayed neutron data
		self.Bi = np.array([0.0054, 0.001087])
		self.B_TOT = sum(self.Bi)
		self.DECAYi = np.array([0.0654, 1.35])
		self.VEL1 = 3.0e7
		self.VEL2 = 3.0e5

		self.Ffactor = (1-self.B_TOT)/kcrit+sum(
			(self.Bi[:]*self.DECAYi[:])/self.DECAYi[:]/kcrit * 
			(1-(1-np.exp(-self.DECAYi[:]*opts.dt))/(self.DECAYi[:]*opts.dt)))

		self.Rfactor = [1/(self.VEL1*opts.dt), 1/(self.VEL2*opts.dt)]

###############################################################

	def build(self, opts, data):

		nGrps = opts.numGroups
		nBins = opts.numBins
		N = nBins*nBins
		delta = opts.delta
		Bz2 = 1.0e-4

		self.abs = []
		self.fis = []
		self.removal = []
		self.D = []
		self.Gscat = np.zeros((opts.numBins*opts.numBins,
			opts.numGroups*opts.numGroups))

		for g in range(1,nGrps+1):
			if g == 1:
				gIn = 2
			else:
				gIn = 1
			for y in range(nBins):
				for x in range(nBins):
					if y*delta >= 135:
						self.fis.append(data['reflector']['fisXS'][g])
						self.abs.append(data['reflector']['absXS'][g])
						self.removal.append(data['reflector']['absXS'][g]
							+data['reflector']['Ex'+str(g)][gIn]
							+Bz2*data['reflector']['D'][g])
						self.D.append(data['reflector']['D'][g])
					elif y*delta >=120:
						if x*delta >= 105:
							self.fis.append(data['reflector']['fisXS'][g])
							self.abs.append(data['reflector']['absXS'][g])
							self.removal.append(data['reflector']['absXS'][g]
							+data['reflector']['Ex'+str(g)][gIn]
							+Bz2*data['reflector']['D'][g])
							self.D.append(data['reflector']['D'][g])
						else:
							self.fis.append(data['fuel3']['fisXS'][g])
							self.abs.append(data['fuel3']['absXS'][g])
							self.removal.append(data['fuel3']['absXS'][g]
							+data['fuel3']['Ex'+str(g)][gIn]
							+Bz2*data['fuel3']['D'][g])
							self.D.append(data['fuel3']['D'][g])
					elif y*delta >= 105:
						if x*delta >= 120:
							self.fis.append(data['reflector']['fisXS'][g])
							self.abs.append(data['reflector']['absXS'][g])
							self.removal.append(data['reflector']['absXS'][g]
							+data['reflector']['Ex'+str(g)][gIn]
							+Bz2*data['reflector']['D'][g])
							self.D.append(data['reflector']['D'][g])
						elif x*delta >= 105:
							self.fis.append(data['fuel4']['fisXS'][g])
							self.abs.append(data['fuel4']['absXS'][g])
							self.removal.append(data['fuel4']['absXS'][g]
							+data['fuel4']['Ex'+str(g)][gIn]
							+Bz2*data['fuel4']['D'][g])
							self.D.append(data['fuel4']['D'][g])
						else:
							self.fis.append(data['fuel3']['fisXS'][g])
							self.abs.append(data['fuel3']['absXS'][g])
							self.removal.append(data['fuel3']['absXS'][g]
							+data['fuel3']['Ex'+str(g)][gIn]
							+Bz2*data['fuel3']['D'][g])
							self.D.append(data['fuel3']['D'][g])
					elif y*delta >= 75:
						if x*delta >=135:
							self.fis.append(data['reflector']['fisXS'][g])
							self.abs.append(data['reflector']['absXS'][g])
							self.removal.append(data['reflector']['absXS'][g]
							+data['reflector']['Ex'+str(g)][gIn]
							+Bz2*data['reflector']['D'][g])
							self.D.append(data['reflector']['D'][g])

						#################################
						#           Region R            #
						#################################
						elif x*delta >= 105: 
							self.fis.append(data['fuel3']['fisXS'][g])
							self.abs.append(data['fuel3']['absXS'][g])
							self.removal.append(data['fuel3']['absXS'][g]
							+data['fuel3']['Ex'+str(g)][gIn]
							+Bz2*data['fuel3']['D'][g])
							self.D.append(data['fuel3']['D'][g])

						elif x*delta >= 75:
							self.fis.append(data['fuel2']['fisXS'][g])
							self.abs.append(data['fuel2']['absXS'][g])
							self.removal.append(data['fuel2']['absXS'][g]
							+data['fuel2']['Ex'+str(g)][gIn]
							+Bz2*data['fuel2']['D'][g])
							self.D.append(data['fuel2']['D'][g])
						elif x*delta >= 15:
							self.fis.append(data['fuel1']['fisXS'][g])
							self.abs.append(data['fuel1']['absXS'][g])
							self.removal.append(data['fuel1']['absXS'][g]
							+data['fuel1']['Ex'+str(g)][gIn]
							+Bz2*data['fuel1']['D'][g])
							self.D.append(data['fuel1']['D'][g])
						else:
							self.fis.append(data['fuel2']['fisXS'][g])
							self.abs.append(data['fuel2']['absXS'][g])
							self.removal.append(data['fuel2']['absXS'][g]
							+data['fuel2']['Ex'+str(g)][gIn]
							+Bz2*data['fuel2']['D'][g])
							self.D.append(data['fuel2']['D'][g])
					elif y*delta >= 15:
						if x*delta >= 135:
							self.fis.append(data['reflector']['fisXS'][g])
							self.abs.append(data['reflector']['absXS'][g])
							self.removal.append(data['reflector']['absXS'][g]
							+data['reflector']['Ex'+str(g)][gIn]
							+Bz2*data['reflector']['D'][g])
							self.D.append(data['reflector']['D'][g])
						elif x*delta >= 105:
							self.fis.append(data['fuel3']['fisXS'][g])
							self.abs.append(data['fuel3']['absXS'][g])
							self.removal.append(data['fuel3']['absXS'][g]
							+data['fuel3']['Ex'+str(g)][gIn]
							+Bz2*data['fuel3']['D'][g])
							self.D.append(data['fuel3']['D'][g])
						else:
							self.fis.append(data['fuel1']['fisXS'][g])
							self.abs.append(data['fuel1']['absXS'][g])
							self.removal.append(data['fuel1']['absXS'][g]
							+data['fuel1']['Ex'+str(g)][gIn]
							+Bz2*data['fuel1']['D'][g])
							self.D.append(data['fuel1']['D'][g])
					else:
						if x*delta >= 135:
							self.fis.append(data['reflector']['fisXS'][g])
							self.abs.append(data['reflector']['absXS'][g])
							self.removal.append(data['reflector']['absXS'][g]
							+data['reflector']['Ex'+str(g)][gIn]
							+Bz2*data['reflector']['D'][g])
							self.D.append(data['reflector']['D'][g])
						elif x*delta >= 105:
							self.fis.append(data['fuel3']['fisXS'][g])
							self.abs.append(data['fuel3']['absXS'][g])
							self.removal.append(data['fuel3']['absXS'][g]
							+data['fuel3']['Ex'+str(g)][gIn]
							+Bz2*data['fuel3']['D'][g])
							self.D.append(data['fuel3']['D'][g])
						elif x*delta >= 75:
							self.fis.append(data['fuel2']['fisXS'][g])
							self.abs.append(data['fuel2']['absXS'][g])
							self.removal.append(data['fuel2']['absXS'][g]
							+data['fuel2']['Ex'+str(g)][gIn]
							+Bz2*data['fuel2']['D'][g])
							self.D.append(data['fuel2']['D'][g])
						elif x*delta >= 15:
							self.fis.append(data['fuel1']['fisXS'][g])
							self.abs.append(data['fuel1']['absXS'][g])
							self.removal.append(data['fuel1']['absXS'][g]
							+data['fuel1']['Ex'+str(g)][gIn]
							+Bz2*data['fuel1']['D'][g])
							self.D.append(data['fuel1']['D'][g])
						else:
							self.fis.append(data['fuel2']['fisXS'][g])
							self.abs.append(data['fuel2']['absXS'][g])
							self.removal.append(data['fuel2']['absXS'][g]
							+data['fuel2']['Ex'+str(g)][gIn]
							+Bz2*data['fuel2']['D'][g])
							self.D.append(data['fuel2']['D'][g])

		# Gscat matrix
		for y in range(nBins):
			for x in range(nBins):
				i = x+(y*nBins)
				count = 1
				gcount = 1
				for g in range(nGrps*nGrps):
					if y*delta >= 135:
						self.Gscat[i,g] = data['reflector']['Ex'+str(gcount)][count]
					elif y*delta >=120:
						if x*delta >= 105:
							self.Gscat[i,g] = data['reflector']['Ex'+str(gcount)][count]
						else:
							self.Gscat[i,g] = data['fuel3']['Ex'+str(gcount)][count]
					elif y*delta >= 105:
						if x*delta >= 120:
							self.Gscat[i,g] = data['reflector']['Ex'+str(gcount)][count]
						elif x*delta >= 105:
							self.Gscat[i,g] = data['fuel4']['Ex'+str(gcount)][count]
						else:
							self.Gscat[i,g] = data['fuel3']['Ex'+str(gcount)][count]
					elif y*delta >= 75:
						if x*delta >=135:
							self.Gscat[i,g] = data['reflector']['Ex'+str(gcount)][count]

						#################################
						#           Region R            #
						#################################
						elif x*delta >= 105: 
							self.Gscat[i,g] = data['fuel3']['Ex'+str(gcount)][count]

						elif x*delta >= 75:
							self.Gscat[i,g] = data['fuel2']['Ex'+str(gcount)][count]
						elif x*delta >= 15:
							self.Gscat[i,g] = data['fuel1']['Ex'+str(gcount)][count]
						else:
							self.Gscat[i,g] = data['fuel2']['Ex'+str(gcount)][count]
					elif y*delta >= 15:
						if x*delta >= 135:
							self.Gscat[i,g] = data['reflector']['Ex'+str(gcount)][count]
						elif x*delta >= 105:
							self.Gscat[i,g] = data['fuel3']['Ex'+str(gcount)][count]
						else:
							self.Gscat[i,g] = data['fuel1']['Ex'+str(gcount)][count]
					else:
						if x*delta >= 135:
							self.Gscat[i,g] = data['reflector']['Ex'+str(gcount)][count]
						elif x*delta >= 105:
							self.Gscat[i,g] = data['fuel3']['Ex'+str(gcount)][count]
						elif x*delta >= 75:
							self.Gscat[i,g] = data['fuel2']['Ex'+str(gcount)][count]
						elif x*delta >= 15:
							self.Gscat[i,g] = data['fuel1']['Ex'+str(gcount)][count]
						else:
							self.Gscat[i,g] = data['fuel2']['Ex'+str(gcount)][count]

					count = count+1
					if count == nGrps+1:
						count = 1
						gcount = gcount+1

		self.ORIGINALabs = copy.copy(self.abs)
		self.removal = np.array(self.removal)    # For sparse implementation
		self.fis = np.array(self.fis)            # For sparse implementation

###############################################################

	def update(self, opts, data, alpha, Doppler, kcrit, wp1=0, wp2=0, wd1=0.0, wd2=0.0):

		wd = [wd1, wd2]

		self.Ffactor = (1-self.B_TOT)/kcrit

		self.Rfactor = [wp1/self.VEL1+1/(self.VEL1*opts.dt), wp2/self.VEL2+1/(self.VEL2*opts.dt)]

		nGrps = opts.numGroups
		nBins = opts.numBins
		delta = opts.delta
		N = nBins*nBins
		Bz2 = 1.0e-4

		self.abs = copy.copy(self.ORIGINALabs)
		self.abs[:N] = self.abs[:N]*(1+Doppler[:N])

		x_lower_bound = int(nBins*7/11)
		x_upper_bound = int(nBins*9/11)
		y_lower_bound = int(nBins*5/11)
		y_upper_bound = int(nBins*7/11)

		for y in range(y_lower_bound,y_upper_bound):
			for x in range(x_lower_bound,x_upper_bound):
				self.abs[N+x+y*nBins]=self.abs[N+x+y*nBins]*(1-(0.0606184*alpha))

		self.removal[:]  = self.abs[:]+Bz2*np.array(self.D[:])
		self.removal[:N] = self.removal[:N]+self.Gscat[:,1]
		self.Premoval = copy.copy(self.removal)
		self.Premoval[:N] = np.array(self.Premoval[:N])+self.Rfactor[0]
		self.Premoval[N:] = np.array(self.Premoval[N:])+self.Rfactor[1]
		self.Pfis = np.array(self.fis[:])*self.Ffactor

#end class