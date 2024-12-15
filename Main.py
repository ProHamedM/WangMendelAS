import numpy as np

from FuzzySetGenerator import FuzzySetGenerator
from RuleGenerator import RuleGenerator
from FuzzyInferenceSystem import FuzzyInferenceSystem
from Validation import Validation
from Optimization import Optimization


# Step 1: Generate training data
def generate_training_data(n_points=41):
    x1 = np.linspace(-5, 5, n_points)
    x2 = np.linspace(-5, 5, n_points)
    X1, X2 = np.meshgrid(x1, x2)
    F = X1 ** 2 + X2 ** 2  # Replace with the desired formula for training data
    return X1.flatten(), X2.flatten(), F.flatten()


# Main execution function
def main():
    # Training data generation
    print("Generating training data...")
    x1_train, x2_train, f_train = generate_training_data()

    # Step 2: Generate fuzzy sets
    print("Generating fuzzy sets...")
    fuzzy_gen = FuzzySetGenerator(n_sets=7, x_range=(-5, 5), f_range=(0, 50))
    input_sets = fuzzy_gen.get_sets()
    output_sets = fuzzy_gen.get_sets()

    # Step 3: Generate rules using Wang-Mendel
    print("Generating Wang-Mendel rules...")
    rule_gen = RuleGenerator(input_sets, output_sets)
    rules = rule_gen.generate_rules(x1_train, x2_train, f_train)

    # Step 4: Initialize Fuzzy Inference System (FIS)
    print("Initializing Fuzzy Inference System...")
    fis = FuzzyInferenceSystem(rules, input_sets, output_sets)

    # Step 5: Validation
    print("Validating the system...")
    validation = Validation(fis, generate_training_data)
    mse_list = validation.validate(n_repeats=100)
    print(f"Mean MSE over 100 runs: {np.mean(mse_list):.4f}")

    # Step 6: Optimization using Simulated Annealing
    print("Starting optimization with Simulated Annealing...")
    optimizer = Optimization(fis, generate_training_data, n_iterations=50)
    optimized_fis, best_mse = optimizer.optimize()
    print(f"Optimized Mean MSE: {best_mse:.4f}")

    # Display results
    print("Project completed.")
    print(f"Original MSE: {np.mean(mse_list):.4f}")
    print(f"Optimized MSE: {best_mse:.4f}")


# Run the main function
if __name__ == "__main__":
    main()
