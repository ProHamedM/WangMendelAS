import numpy as np
import matplotlib.pyplot as plt

# Number of points for x1 and x2
n_points = 41

# Generate equally spaced values for x1 and x2 in the range [-5, 5]
x1 = np.linspace(-5, 5, n_points)
x2 = np.linspace(-5, 5, n_points)

# Create a grid of x1 and x2 values
X1, X2 = np.meshgrid(x1, x2)

# Compute the output function F(x1, x2) = x1^2 + x2^2
F = X1**2 + X2**2

# Visualize the function as a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, F, cmap='viridis')
ax.set_title("F(x1, x2) = x1^2 + x2^2")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('F(x1, x2)')
plt.show()
