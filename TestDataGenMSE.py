import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Number of points for x1 and x2
n_points = 41
x1 = np.linspace(-5, 5, n_points)  # Input range for x1
x2 = np.linspace(-5, 5, n_points)  # Input range for x2
X1, X2 = np.meshgrid(x1, x2)  # Create grid
F = X1 ** 2 + X2 ** 2  # Output F(x1, x2)

# Define triangular fuzzy sets
n_sets = 7
triangular_sets = []
centers = np.linspace(-5, 5, n_sets)
x_range = np.linspace(-5, 5, 100)

for center in centers:
    triangular_sets.append(fuzz.trimf(x_range, [center - 1.5, center, center + 1.5]))


# Function to find maximum membership
def max_membership(value, fuzzy_sets, membership_range):
    memberships = [fuzz.interp_membership(membership_range, fs, value) for fs in fuzzy_sets]
    return np.argmax(memberships), memberships


# Centre Average Defuzzifier
def center_average_defuzzifier(weights, local_centers):
    return np.sum(weights * local_centers) / np.sum(weights)


# Mean Squared Error (MSE)
def mse(y_true, y_predicted):
    return np.sum((y_true - y_predicted) ** 2) / (2 * len(y_true))


# Generate test points (168 random points)
test_size = 168
np.random.seed(42)
test_x1 = np.random.uniform(-5, 5, test_size)
test_x2 = np.random.uniform(-5, 5, test_size)
test_y = test_x1 ** 2 + test_x2 ** 2

# Perform validation over 100 iterations
mse_values = []
centers_output = np.linspace(0, 50, n_sets)  # Output fuzzy set centers

for iteration in range(100):
    y_pred = []
    for i in range(test_size):
        # Find memberships for x1 and x2
        idx1, mem1 = max_membership(test_x1[i], triangular_sets, x_range)
        idx2, mem2 = max_membership(test_x2[i], triangular_sets, x_range)

        # Combine memberships using MIN operator (implication = conjunction)
        weight = min(mem1[idx1], mem2[idx2])

        # Defuzzify to predict output
        crisp_output = center_average_defuzzifier(np.array([weight]), np.array([centers_output[idx1]]))
        y_pred.append(crisp_output)

    # Calculate and store MSE for this iteration
    error = mse(test_y, y_pred)
    mse_values.append(error)

# Results
print("Average MSE over 100 iterations:", np.mean(mse_values))
print("Variance of MSE:", np.var(mse_values))

# Plot MSE over iterations
plt.plot(range(1, 101), mse_values, marker='o', linestyle='--')
plt.title("MSE Over 100 Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.grid()
plt.show()
