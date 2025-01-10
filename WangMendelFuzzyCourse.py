import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pandas

# Modify the input and output ranges
xone_min, xone_max = -5, 5
xtwo_min, xtwo_max = -5, 5
f_min, f_max = 0, 50

# Produce the training data grid
xone = numpy.linspace(xone_min, xone_max, 41)  # Generate 41 values for x1
xtwo = numpy.linspace(xtwo_min, xtwo_max, 41)  # Generate 41 values for x2
x_one, x_two = numpy.meshgrid(xone, xtwo)
target_function = x_one ** 2 + x_two ** 2  # Target function

# Define the fuzzy system with triangular membership functions
def triangular_membership_function(x, num_sets, x_min, x_max):
    # Generate triangular membership functions.
    step = (x_max - x_min) / (num_sets - 1)
    centers = numpy.linspace(x_min, x_max, num_sets)
    membership_functions_list = numpy.maximum(0, 1 - numpy.abs(x[:, None] - centers) / step)
    return centers, membership_functions_list

# Rule production: Wang-Mendel fuzzy rule
def generate_rules(xone_centers, xtwo_centers, function_centers):
    # Generate Wang-Mendel fuzzy rules using broadcasting.
    x_one, x_two = numpy.meshgrid(xone_centers, xtwo_centers)
    rule_outputs = x_one**2 + x_two**2
    rule_outputs = function_centers[numpy.argmin(numpy.abs(function_centers[:, None, None] - rule_outputs), axis=0)]
    return numpy.array(list(zip(x_one.flatten(), x_two.flatten(), rule_outputs.flatten())))

# Defuzzification: center-average defuzzifier
def center_average_defuzzifier(x1_val, x2_val, rules, num_sets):
    xone_centers = rules[:, 0]
    xtwo_centers = rules[:, 1]
    function_centers = rules[:, 2]

    # Compute membership degrees for x1 and x2
    step = (xone_max - xone_min) / (num_sets - 1)
    membership_degree_xone = numpy.maximum(0, 1 - numpy.abs(x1_val - xone_centers) / step)
    membership_degree_xtwo = numpy.maximum(0, 1 - numpy.abs(x2_val - xtwo_centers) / step)

    # Compute rule activation and defuzzified output
    rule_strengths = numpy.minimum(membership_degree_xone, membership_degree_xtwo)
    numerator = numpy.sum(rule_strengths * function_centers)
    denominator = numpy.sum(rule_strengths)
    return numerator / denominator if denominator != 0 else 0

# Calculate MSE Function (Minimum Mean Squared Error)
def calculate_mse(y_actual, y_predicted):
    return numpy.mean((numpy.array(y_actual) - numpy.array(y_predicted)) ** 2)

# Optimize to reduce error
def optimize_fuzzy_system():
    min_mse = float('inf')
    optimal_number_sets = 0
    optimal_fuzzy_rules = None

    # Search for the best number of fuzzy sets
    for num_sets in range(2, 101):  # Searching between 2 and 100 fuzzy sets
        xone_centers, _ = triangular_membership_function(xone, num_sets, xone_min, xone_max)
        xtwo_centers, _ = triangular_membership_function(xtwo, num_sets, xtwo_min, xtwo_max)
        function_centers, _ = triangular_membership_function(target_function.flatten(), num_sets, f_min, f_max)

        rules = generate_rules(xone_centers, xtwo_centers, function_centers)

        # Calculate train error
        train_function_predicted = numpy.array([
            center_average_defuzzifier(x1_val, x2_val, rules, num_sets)
            for x1_val, x2_val in zip(x_one.flatten(), x_two.flatten())
        ])
        train_mse_function = calculate_mse(target_function.flatten(), train_function_predicted)

        # Track progress during the 100 iterations
        print(f"Iteration {num_sets}: Train MSE = {train_mse_function:.4f}")

        # Check if this is the best configuration
        if train_mse_function < min_mse:
            min_mse = train_mse_function
            optimal_number_sets = num_sets
            optimal_fuzzy_rules = rules

    return optimal_fuzzy_rules, optimal_number_sets

# Run optimization
optimal_rules, optimal_num_sets = optimize_fuzzy_system()

# Evaluate optimal fuzzy system
train_f_predicted = numpy.array([
    center_average_defuzzifier(x1_val, x2_val, optimal_rules, optimal_num_sets)
    for x1_val, x2_val in zip(x_one.flatten(), x_two.flatten())
])
train_mse = calculate_mse(target_function.flatten(), train_f_predicted)
print(f"\nFinal Optimal Fuzzy Sets: {optimal_num_sets}")
print(f"Final Train MSE: {train_mse:.4f}")

# Test data
numpy.random.seed(42)
test_x1 = numpy.random.uniform(xone_min, xone_max, 168)
test_x2 = numpy.random.uniform(xtwo_min, xtwo_max, 168)
test_f_actual = test_x1**2 + test_x2**2
test_f_predicted = numpy.array([
    center_average_defuzzifier(x1, x2, optimal_rules, optimal_num_sets)
    for x1, x2 in zip(test_x1, test_x2)
])

test_mse = calculate_mse(test_f_actual, test_f_predicted)
test_mae = numpy.mean(numpy.abs(numpy.array(test_f_actual) - numpy.array(test_f_predicted)))
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

# Create a Table of x1, x2, and Generated Fuzzy Sets
x_one_centers, _ = triangular_membership_function(xone, optimal_num_sets, xone_min, xone_max)
x_two_centers, _ = triangular_membership_function(xtwo, optimal_num_sets, xtwo_min, xtwo_max)
f_centers, _ = triangular_membership_function(target_function.flatten(), optimal_num_sets, f_min, f_max)

table_data = []
for x_one_center in x_one_centers:
    for x_two_center in x_two_centers:
        # Calculate the corresponding fuzzy output for this x1, x2 combination
        function_output = f_centers[numpy.argmin(numpy.abs(f_centers - (x_one_center ** 2 + x_two_center ** 2)))]
        table_data.append({"x1": x_one_center, "x2": x_two_center, "Output (Fuzzy Set)": function_output})

# Convert the data to a DataFrame
dataframe = pandas.DataFrame(table_data)
print("Generated Fuzzy Sets Table:")
print(dataframe)
