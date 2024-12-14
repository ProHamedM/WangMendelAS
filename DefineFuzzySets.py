import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Define ranges for inputs and output
x_range = np.linspace(-5, 5, 100)  # Input range for x1 and x2
f_range = np.linspace(0, 50, 100)  # Output range for F

# Number of triangular fuzzy sets
n_sets = 7

# Generate triangular membership functions for input (x_range)
triangular_sets = []
centers = np.linspace(-5, 5, n_sets)  # Centers of the triangles

for center in centers:
    # Define triangular fuzzy set with a small overlap between neighboring sets
    triangular_set = fuzz.trimf(x_range, [center - 1.5, center, center + 1.5])
    triangular_sets.append(triangular_set)

# Plot the triangular fuzzy sets for input
plt.figure(figsize=(8, 5))
for i, triangle in enumerate(triangular_sets):
    plt.plot(x_range, triangle, label=f"Set {i+1}")
plt.title("Triangular Fuzzy Sets for Input x1 and x2")
plt.xlabel("Input Range")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()
plt.show()
