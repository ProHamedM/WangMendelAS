import numpy as np

# Handles test data generation, MSE calculation, and repetitions
class Validation:
    def __init__(self, n_repeats=100):
        self.n_repeats = n_repeats  # Number of repetitions for MSE calculation

    # Calculate Mean Squared Error (MSE).
    @staticmethod
    def calculate_mse(true_values, predicted_values):
        return np.mean((true_values - predicted_values)**2) / 2

    # Validate the fuzzy system with test data
    def validate(self, fuzzy_system, test_data_size=100):

        mse_list = []
        for _ in range(self.n_repeats):
            # Generate random test inputs
            test_x1 = np.random.uniform(-5, 5, test_data_size)
            test_x2 = np.random.uniform(-5, 5, test_data_size)
            true_outputs = test_x1 ** 2 + test_x2 ** 2

            # Predict outputs
            predicted_outputs = fuzzy_system.predict(test_x1, test_x2)

            # Filter out invalid predictions
            valid_mask = ~np.isnan(predicted_outputs)
            if not valid_mask.any():
                print("Warning: All predictions are invalid (NaN).")
                continue  # Skip this repetition

            filtered_true_outputs = true_outputs[valid_mask]
            filtered_predicted_outputs = predicted_outputs[valid_mask]

            # Compute MSE
            mse = self.calculate_mse(filtered_true_outputs, filtered_predicted_outputs)
            mse_list.append(mse)

            # Final MSE calculation
        if mse_list:
            return np.mean(mse_list)
        else:
            print("Error: No valid MSE values calculated.")
            return float('nan')