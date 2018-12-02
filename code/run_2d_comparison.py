'''
This Script is a RUN function for the comparison of image results from the "gradient" method proposed in this project vs. standard von Neumann and Moore growth models.  Data is drawn from three separate 2d Gaussians which have small regions of overlap between them.  Data from each gaussian is assigned to a unique class.
'''

# Standard Imports
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split

# Separate Biosystems for the purposes of these tests
import biosystem
from biosystem import culture
from vonneumann_biosystem import culture as vnCult
from moore_biosystem import culture as mCult

# main
if __name__ == "__main__":
	
	# Ignore warnings
	import warnings
	warnings.simplefilter("ignore")
	
	# Start timer for benchmarking
	import time
	start_me = time.time()
	
	# Model params
	n = 75 # Number of data points per class
	m = 30 # Number of cells per dimension
	iter = 100 # Number of iterations in the simulation
	
	# Create all of the systems
	luca = culture([m, m])
	lucaVN = vnCult([m, m])
	lucaM = mCult([m, m])
	
	# Construct dataset and class vector
	X1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n)
	X2 = np.random.multivariate_normal([1.5,-1.5], [[1,0.5],[0.5,1]], n)
	X3 = np.random.multivariate_normal([-2.3,2.3], [[2,1],[1,2]], n)
	XT = np.concatenate((X1, X2, X3), 0)
	Y1 = np.ones(n)-1
	Y2 = np.ones(n)*2-1
	Y3 = np.ones(n)*3-1
	YT = np.concatenate((Y1, Y2, Y3), 0)
	
	# Initialize cell positions in each system
	luca.inoculate(XT,YT)
	lucaVN.inoculate(XT,YT)
	lucaM.inoculate(XT,YT)
	
	# Fermentation steps - iterations predefined
	for i in range(iter):
		a = luca.ferment()
		b = lucaVN.ferment()
		c = lucaM.ferment()
		
		# Update status in console
		if i%10 == 0:
			print("Cycle Number: " + str(i))
			print(a)
			print(b)
			print(c)
			print("")
	
	# Create image arrays
	img = deepcopy(biosystem.empties([m, m]))
	imgVN = deepcopy(biosystem.empties([m, m]))
	imgM = deepcopy(biosystem.empties([m, m]))
	
	# Set variables to model results
	bins = luca.cells
	binsVN = lucaVN.cells
	binsM = lucaM.cells
	
	# Build images for each simulation type
	for i in range(0, m):
		for j in range(0, m):
			
			if bins[i][j]:
				s = bins[i][j][0].species
				rgb = (np.zeros(3)).tolist()
				rgb[int(s)] = 255
				img[m-1-j][i] = rgb
			else:
				img[m-1-j][i] = [255,255,255]
				
			if binsVN[i][j]:
				s = binsVN[i][j][0].species
				rgb = (np.zeros(3)).tolist()
				rgb[int(s)] = 255
				imgVN[m-1-j][i] = rgb
			else:
				imgVN[m-1-j][i] = [255,255,255]
				
			if binsM[i][j]:
				s = binsM[i][j][0].species
				rgb = (np.zeros(3)).tolist()
				rgb[int(s)] = 255
				imgM[m-1-j][i] = rgb
			else:
				imgM[m-1-j][i] = [255,255,255]
	
	# Convert image arrays to appropriate data types
	img = np.array(img, dtype='uint8')
	imgVN = np.array(imgVN, dtype='uint8')
	imgM = np.array(imgM, dtype='uint8')
	
	# Show the results
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 3))
	
	# Data Subplot
	ax1.plot(X1[:,0], X1[:,1], 'r.')
	ax1.plot(X2[:,0], X2[:,1], 'g.')
	ax1.plot(X3[:,0], X3[:,1], 'b.')
	ax1.title.set_text('Dataset')
	
	# Method 1 Subplot
	ax2.set_yticklabels([])
	ax2.set_xticklabels([])
	ax2.title.set_text('Gradient Method')
	ax2.imshow(img)
	
	# Method 2 Subplot
	ax3.set_yticklabels([])
	ax3.set_xticklabels([])
	ax3.title.set_text('Von Neumann Method')
	ax3.imshow(imgVN)
	
	# Method 3 Subplot
	ax4.set_yticklabels([])
	ax4.set_xticklabels([])
	ax4.title.set_text('Moore Method')
	ax4.imshow(imgM)
	
	plt.show()
	
	# Print final running time
	end_me = time.time() - start_me
	print("Run Time:")
	print(end_me)
	