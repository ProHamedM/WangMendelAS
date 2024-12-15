import numpy as np

class Validation:
    """
    Handles test data generation, MSE calculation, and repetitions.
    """
    def __init__(self, n_repeats=100):
        self.n_repeats = n_repeats  # Number of repetitions for MSE calculation

    def calculate_mse(self, true_values, predicted_values):
        """
        Calculate Mean Squared Error (MSE).
        """
        return np.mean((true_values - predicted_values)**2) / 2

    def validate(self, fuzzy_system, test_data):
        """
        Validate the fuzzy system with test data.
        """
        mse_list = []
        for _ in range(self.n_repeats):
            # Generate random test inputs
            test_x1 = np.random.uniform(-5, 5, len(test_data))
            test_x2 = np.random.uniform(-5, 5, len(test_data))
            true_outputs = test_x1**2 + test_x2**2

            # Predict outputs
            predicted_outputs = fuzzy_system.predict(test_x1, test_x2)

            # Compute MSE
            mse = self.calculate_mse(true_outputs, predicted_outputs)
            mse_list.append(mse)

        return np.mean(mse_list)