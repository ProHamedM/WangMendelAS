import matplotlib.pyplot as plt
import numpy as np

from WangMendelAS.Base.FuzzySetGenerator import FuzzySetGenerator

# Create an instance of FuzzySetGenerator
fuzzy_gen = FuzzySetGenerator(data_range=np.linspace(-5, 5, 100), n_sets=7)

# Generate fuzzy sets for inputs and outputs
input_sets = fuzzy_gen.get_sets()
output_sets = fuzzy_gen.get_sets()  # For simplicity, using the same for output in this test

# Visualize the input fuzzy sets
plt.figure(figsize=(10, 6))
for i, mf in enumerate(input_sets):
    plt.plot(np.linspace(-5, 5, 100), mf, label=f"Input Set {i+1}")
plt.title("Triangular Fuzzy Sets for Input")
plt.xlabel("Input Range")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()
plt.show()

# Visualize the output fuzzy sets
plt.figure(figsize=(10, 6))
for i, mf in enumerate(output_sets):
    plt.plot(np.linspace(0, 50, 100), mf, label=f"Output Set {i+1}")
plt.title("Triangular Fuzzy Sets for Output")
plt.xlabel("Output Range")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()
plt.show()
