import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

core_power = np.loadtxt('PowerChange.txt')

print("MAX: ", max(core_power))


maximum = argrelextrema(np.array(core_power), np.greater)
maxpower1 = core_power[maximum[0][0]]
# print(core_power[maximum[0][0]])
for i in range(len(maximum[0])):
	print(core_power[maximum[0][i]])