import numpy as np
import matplotlib.pyplot as plt

# Define the range for each dimension
x_min, x_max = 0.5, 1
y_min, y_max = 5, 10

# Number of samples per dimension (sqrt of 30 rounded)
num_samples_per_dim = int(np.sqrt(30))
if num_samples_per_dim ** 2 < 30:
    num_samples_per_dim += 1

# Generate evenly spaced values along each dimension
x_values = np.linspace(x_min, x_max, num_samples_per_dim)
y_values = np.linspace(y_min, y_max, num_samples_per_dim)

# Create the 2D grid
x_grid, y_grid = np.meshgrid(x_values, y_values)

# Flatten the grid to create 2D sample points
samples = np.column_stack([x_grid.ravel(), y_grid.ravel()])

# Select only the first 30 samples
samples = samples[:30]

# Plot the samples to visualize the uniform spacing
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.title('Uniformly Spaced Samples in 2D Space')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()
