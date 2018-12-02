# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Function that creates random cell boundaries via unit steps in a given direction
# Probability of "stepping" in each of four directions in defined by the input NESW
def create_boundary(points, p, NESW):
	for i in range(len(points), len(points)+p):
		currentPt = deepcopy(points[i-1])
		draw = np.random.rand(1)

		if draw < sum(NESW[:1]):
			currentPt[1] += 1
		elif draw < sum(NESW[:2]):
			currentPt[0] += 1
		elif draw < sum(NESW[:3]):
			currentPt[1] -= 1
		elif draw < sum(NESW[:4]):
			currentPt[0] -= 1

		points.append(currentPt)

        
if __name__ == "__main__":
    
	# Starting point
	points = [[0,0]]
	
	# Add boundary points
	p = 20
	NESW = [0, 0.75, 0.25, 0]
	create_boundary(points, p, NESW)
	
	# Add boundary points
	NESW = [0.25, 0.75, 0, 0]
	create_boundary(points, p, NESW)

	# Add boundary points
	NESW = [0, 0.75, 0.25, 0]
	create_boundary(points, p, NESW)
	
	# Convert to numpy array
	points = np.array(points)
	
	# Calculate polyfit coefficients
	p2 = np.polyfit(points[:,0], points[:,1], 2)
	p3 = np.polyfit(points[:,0], points[:,1], 3)
	p5 = np.polyfit(points[:,0], points[:,1], 5)

	# Compute polynomials based on coefficients
	x = np.arange(np.min(points[:,0]), np.max(points[:,0]), 0.01)
	y2 = p2[0]*x**2 + p2[1]*x + p2[2]
	y3 = p3[0]*x**3 + p3[1]*x**2 + p3[2]*x + p3[3]
	y5 = p5[0]*x**5 + p5[1]*x**4 + p5[2]*x**3 + p5[3]*x**2 + p5[4]*x + p5[5]
	
	# Plot the results
	plt.plot(points[:,0], points[:,1], 'k--')
	plt.plot(x, y2, 'r-')
	plt.plot(x, y3, 'b-')
	plt.plot(x, y5, 'm-')
	
	plt.show()