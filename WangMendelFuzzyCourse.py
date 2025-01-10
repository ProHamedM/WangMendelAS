import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Modify the input and output ranges
xone_min, xone_max = -5, 5
xtwo_min, xtwo_max = -5, 5
f_min, f_max = 0, 50

# Produce the training data grid
xone = np.linspace(xone_min, xone_max, 41)  # Generate 41 values for x1
xtwo = np.linspace(xtwo_min, xtwo_max, 41)  # Generate 41 values for x2
x_one, x_two = np.meshgrid(xone, xtwo)
target_function = x_one ** 2 + x_two ** 2  # Target function


def triangular_membership_function(x, num_sets, x_min, x_max):
    """Generate triangular membership functions."""
    step = (x_max - x_min) / (num_sets - 1)
    centers = np.linspace(x_min, x_max, num_sets)
    membership_functions_list = np.maximum(0, 1 - np.abs(x[:, None] - centers) / step)
    return centers, membership_functions_list


def generate_rules(xone_centers, xtwo_centers, function_centers):
    """Generate Wang-Mendel fuzzy rules."""
    x_1, x_2 = np.meshgrid(xone_centers, xtwo_centers)
    rule_outputs = x_1**2 + x_2**2
    rule_outputs = function_centers[np.argmin(np.abs(function_centers[:, None, None] - rule_outputs), axis=0)]
    return np.array(list(zip(x_1.flatten(), x_2.flatten(), rule_outputs.flatten())))


def center_average_defuzzifier(x1_val, x2_val, rules, num_sets):
    """Defuzzification: center-average defuzzifier."""
    xone_centers = rules[:, 0]
    xtwo_centers = rules[:, 1]
    function_centers = rules[:, 2]

    step = (xone_max - xone_min) / (num_sets - 1)
    membership_degree_xone = np.maximum(0, 1 - np.abs(x1_val - xone_centers) / step)
    membership_degree_xtwo = np.maximum(0, 1 - np.abs(x2_val - xtwo_centers) / step)

    rule_strengths = np.minimum(membership_degree_xone, membership_degree_xtwo)
    numerator = np.sum(rule_strengths * function_centers)
    denominator = np.sum(rule_strengths)
    return numerator / denominator if denominator != 0 else 0


def calculate_mse(y_actual, y_predicted):
    """Calculate MSE."""
    return np.mean((np.array(y_actual) - np.array(y_predicted)) ** 2)


def simulated_annealing():
    """Optimize using simulated annealing."""
    print("Starting simulated annealing optimization...")

    # Initial configuration
    best_rules = None
    current_sets = random.randint(2, 100)  # Start with a random number of sets
    best_sets = current_sets
    current_temp = 100  # Initial temperature
    cooling_rate = 0.95
    min_temp = 1e-3
    best_mse = float("inf")
    history = []

    while current_temp > min_temp:
        # Generate fuzzy sets and rules
        xone_centers, _ = triangular_membership_function(xone, current_sets, xone_min, xone_max)
        xtwo_centers, _ = triangular_membership_function(xtwo, current_sets, xtwo_min, xtwo_max)
        function_centers, _ = triangular_membership_function(target_function.flatten(), current_sets, f_min, f_max)
        rules = generate_rules(xone_centers, xtwo_centers, function_centers)

        # Calculate train error
        train_function_predicted = np.array([
            center_average_defuzzifier(x1_val, x2_val, rules, current_sets)
            for x1_val, x2_val in zip(x_one.flatten(), x_two.flatten())
        ])
        mse = calculate_mse(target_function.flatten(), train_function_predicted)

        # Update history
        history.append({"Number of Sets": current_sets, "MSE": mse})

        # Check if this is the best configuration
        if mse < best_mse:
            best_mse = mse
            best_sets = current_sets
            best_rules = rules

        # Simulated Annealing Probability
        next_sets = current_sets + random.choice([-1, 1])  # Neighboring solution
        next_sets = max(2, min(next_sets, 100))  # Keep within bounds

        delta_mse = mse - best_mse
        if delta_mse < 0 or np.exp(-delta_mse / current_temp) > random.random():
            current_sets = next_sets

        # Cool down
        current_temp *= cooling_rate

        # log
        print(f"Temp: {current_temp:.4f}, Number of Sets: {current_sets}, MSE: {mse:.4f}")

    return best_rules, best_sets, best_mse, history


# Run Simulated Annealing
optimal_rules, optimal_num_sets, optimal_mse, iteration_results = simulated_annealing()

# Generate Iterations Table
iterations_table = pd.DataFrame(iteration_results)
iterations_table.to_excel("Iterations_and_MSEs.xlsx", index=False)  # Save to Excel
print("Saved iterations and MSEs table to Iterations_and_MSEs.xlsx")

# Generate Fuzzy Rules Table
fuzzy_rules_table = pd.DataFrame({
    "x1 Center": optimal_rules[:, 0],
    "x2 Center": optimal_rules[:, 1],
    "Output (Fuzzy Set)": optimal_rules[:, 2]
})
fuzzy_rules_table.to_excel("Fuzzy_Rules.xlsx", index=False)  # Save to Excel
print("Saved fuzzy rules table to Fuzzy_Rules.xlsx")

# Evaluate optimal fuzzy system
train_f_predicted = np.array([
    center_average_defuzzifier(x1_val, x2_val, optimal_rules, optimal_num_sets)
    for x1_val, x2_val in zip(x_one.flatten(), x_two.flatten())
])
train_mse = calculate_mse(target_function.flatten(), train_f_predicted)
print(f"\nFinal Optimal Fuzzy Sets: {optimal_num_sets}")
print(f"Final Train MSE: {train_mse:.4f}")

# Test data
np.random.seed(42)
test_x1 = np.random.uniform(xone_min, xone_max, 168)
test_x2 = np.random.uniform(xtwo_min, xtwo_max, 168)
test_f_actual = test_x1**2 + test_x2**2
test_f_predicted = np.array([
    center_average_defuzzifier(x1, x2, optimal_rules, optimal_num_sets)
    for x1, x2 in zip(test_x1, test_x2)
])

test_mse = calculate_mse(test_f_actual, test_f_predicted)
test_mae = np.mean(np.abs(np.array(test_f_actual) - np.array(test_f_predicted)))
print(f"Final Test MSE: {test_mse:.4f}")
print(f"Final Mean Absolute Error (Test): {test_mae:.4f}")

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
Z_predicted = train_f_predicted.reshape(x_one.shape)
ax.plot_surface(x_one, x_two, Z_predicted, cmap="viridis", alpha=0.8)
ax.set_title("Optimized 3D Fuzzy Output Surface")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Fuzzy Output")
plt.show()
