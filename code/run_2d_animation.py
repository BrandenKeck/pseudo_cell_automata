'''
This Script is a RUN function which shows an animated progression of cell growth in the cellular automation defined in 'biosystem.py'.  Data is drawn randomly over a rectangular space.  Then, the function y=x**2 is used as a boundary to classify data into two categories.  Cellular automation is run with the initial dataset to determine how accurately the y=x**2 boundary can be reproduced.
'''

# Standard Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split

# Biosystem Import
import biosystem
from biosystem import culture

# Update function for the animation
def display2D(frameNum, mf, luca):

	# Iteration of the simulation and update status to console
	a = luca.ferment()
	print(a)

	# Set a cell variable
	bins = luca.cells
	
	# Determine 2D dimensions
	xdim = len(bins)
	ydim = len(bins[0])
	
	# Construct the animation image
	img = biosystem.empties([xdim, ydim])
	for i in range(0, xdim):
		for j in range(0, ydim):
			if bins[i][j]:
				s = bins[i][j][0].species
				f = bins[i][j][0].fitness
				
				f = f/luca.totalfitness
				if f > 0.0022: f = 0.0022
				if f < 0.0005: f = 0.0005
				
				rgb = (np.ones(3)*(255 - 100000*f)).tolist()
				rgb[int(s)] = 255
				
				img[ydim-1-j][i] = rgb
			else:
				img[ydim-1-j][i] = [255,255,255]
	
	# Iterate the animation
	mf.set_data(np.array(img, dtype='uint8'))
	return mf,
	
if __name__ == "__main__":

	# Ignore warnings
	import warnings
	warnings.simplefilter("ignore")
	
	# Important simulation parameters
	n = 100 # Number of datapoints in initial set
	m = 30 # Size of cell grid
	
	# Create system
	luca = culture([m, m])
	
	# Construct parabolic data set
	XT = np.concatenate(((np.random.rand(n)*8 - 4)[:,None], (np.random.rand(n)*8-2)[:,None]), 1)
	YT = []
	for i, x in enumerate(XT):
		if x[1] < x[0]**2:
			YT.append(0)
			plt.plot(XT[i,0], XT[i,1], 'ro')
		else:
			YT.append(1)
			plt.plot(XT[i,0], XT[i,1], 'go')
	YT = np.array(YT)

	# Initialization of cell positions
	luca.inoculate(XT,YT)
	
	# Driving forward of animation also drives the simulation forward
	fig, ax = plt.subplots() 
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	img = ax.imshow(np.zeros([m,m,3]))
	ani = animation.FuncAnimation(fig, display2D, fargs=(img, luca, ), interval=1)
	plt.show()
	