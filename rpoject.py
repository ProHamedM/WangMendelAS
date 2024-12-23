#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


# Define the input and output ranges
x1_min, x1_max = -5, 5
x2_min, x2_max = -5, 5
f_min, f_max = 0, 50


# In[3]:


# Generate the training data grid
x1 = np.linspace(x1_min, x1_max, 41)  # 41 values for x1
x2 = np.linspace(x2_min, x2_max, 41)  # 41 values for x2
X1, X2 = np.meshgrid(x1, x2)
F = X1**2 + X2**2  # Target function


# In[4]:


# Define the fuzzy system with triangular membership functions
def triangular_mf(x, num_sets, xmin, xmax):
    """Generate triangular membership functions."""
    step = (xmax - xmin) / (num_sets - 1)
    centers = np.linspace(xmin, xmax, num_sets)
    mfs = []
    for c in centers:
        mf = np.maximum(0, 1 - np.abs(x - c) / step)
        mfs.append(mf)
    return centers, mfs


# In[5]:


# Rule generation: Wang-Menel fuzzy rule (minimum operation)
def generate_rules(x1_centers, x2_centers, f_centers):
    """Generate Wang-Mendel fuzzy rules."""
    rules = []
    for i, x1_c in enumerate(x1_centers):
        for j, x2_c in enumerate(x2_centers):
            rule_out = f_centers[np.argmin(np.abs(f_centers - (x1_c**2 + x2_c**2)))]
            rules.append((x1_c, x2_c, rule_out))
    return rules


# In[6]:


# Defuzzification: center-average defuzzifier
def center_average_defuzzifier(x1_val, x2_val, rules, num_sets):
    numerator, denominator = 0, 0
    for x1_c, x2_c, f_c in rules:
        mu_x1 = np.maximum(0, 1 - np.abs(x1_val - x1_c) / (x1_max - x1_min) * (num_sets - 1))
        mu_x2 = np.maximum(0, 1 - np.abs(x2_val - x2_c) / (x2_max - x2_min) * (num_sets - 1))
        mu_rule = min(mu_x1, mu_x2)  # Minimum operation
        numerator += mu_rule * f_c
        denominator += mu_rule
    return numerator / denominator if denominator != 0 else 0


# In[7]:


# MSE Calculation Function
def calculate_mse(y_actual, y_predicted):
    N = len(y_actual)
    mse = (1 / (2 * N)) * np.sum((np.array(y_actual) - np.array(y_predicted))**2)
    return mse


# In[8]:


# Optimization to reduce error
def optimize_fuzzy_system():
    min_mse = float('inf')
    optimal_num_sets = 0
    optimal_rules = None
    mse_threshold = 0.01  # Target error threshold

    # Search for the best number of fuzzy sets
    for num_sets in range(5, 21):  # Searching between 5 and 20 fuzzy sets
        x1_centers, _ = triangular_mf(x1, num_sets, x1_min, x1_max)
        x2_centers, _ = triangular_mf(x2, num_sets, x2_min, x2_max)
        f_centers, _ = triangular_mf(F, num_sets, f_min, f_max)

        rules = generate_rules(x1_centers, x2_centers, f_centers)

        # Calculate train error
        train_f_predicted = [center_average_defuzzifier(x1_val, x2_val, rules, num_sets) 
                             for x1_val, x2_val in zip(X1.flatten(), X2.flatten())]
        train_mse = calculate_mse(F.flatten(), train_f_predicted)

        print(f"Num Sets: {num_sets}, Train MSE: {train_mse:.4f}")

        # Check if this is the best configuration
        if train_mse < min_mse:
            min_mse = train_mse
            optimal_num_sets = num_sets
            optimal_rules = rules

        # Stop early if the error is below the threshold
        if train_mse < mse_threshold:
            break

    return optimal_rules, optimal_num_sets


# In[9]:


# Run optimization
optimal_rules, optimal_num_sets = optimize_fuzzy_system()


# In[10]:


# Final evaluation with optimal fuzzy system
train_f_predicted = [center_average_defuzzifier(x1_val, x2_val, optimal_rules, optimal_num_sets)
                     for x1_val, x2_val in zip(X1.flatten(), X2.flatten())]
train_mse = calculate_mse(F.flatten(), train_f_predicted)
print(f"\nFinal Optimal Fuzzy Sets: {optimal_num_sets}")
print(f"Final Train MSE: {train_mse:.4f}")


# In[11]:


# Test data
np.random.seed(42)
test_x1 = np.random.uniform(x1_min, x1_max, 168)
test_x2 = np.random.uniform(x2_min, x2_max, 168)
test_f_actual = test_x1**2 + test_x2**2
test_f_predicted = [center_average_defuzzifier(x1, x2, optimal_rules, optimal_num_sets) 
                    for x1, x2 in zip(test_x1, test_x2)]

test_mse = calculate_mse(test_f_actual, test_f_predicted)
test_mae = np.mean(np.abs(np.array(test_f_actual) - np.array(test_f_predicted)))
print(f"Final Test MSE: {test_mse:.4f}")
print(f"Final Mean Absolute Error (Test): {test_mae:.4f}")


# In[12]:


# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
Z_predicted = np.array(train_f_predicted).reshape(X1.shape)
ax.plot_surface(X1, X2, Z_predicted, cmap="viridis", alpha=0.8)
ax.set_title("Optimized 3D Fuzzy Output Surface")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Fuzzy Output")
plt.show()


# In[23]:


# Create a Table of x1, x2, and Generated Fuzzy Sets
x1_centers, _ = triangular_mf(x1, optimal_num_sets, x1_min, x1_max)
x2_centers, _ = triangular_mf(x2, optimal_num_sets, x2_min, x2_max)
f_centers, _ = triangular_mf(F, optimal_num_sets, f_min, f_max)

table_data = []
for x1_c in x1_centers:
    for x2_c in x2_centers:
        # Calculate the corresponding fuzzy output for this x1, x2 combination
        f_output = f_centers[np.argmin(np.abs(f_centers - (x1_c**2 + x2_c**2)))]
        table_data.append({"x1": x1_c, "x2": x2_c, "Output (Fuzzy Set)": f_output})


# In[24]:


# Convert the data to a DataFrame
df = pd.DataFrame(table_data)
print("Generated Fuzzy Sets Table:")
print(df)  

