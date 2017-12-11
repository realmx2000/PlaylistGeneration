import matplotlib.pyplot as plt
import numpy as np

def make_heatmap(matrix = np.random.random((100,100))):
	plt.figure()
	plt.imshow(matrix, cmap='hot', interpolation='nearest')
	plt.savefig('heatmap.png')

