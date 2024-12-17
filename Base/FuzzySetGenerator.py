import numpy as np
import skfuzzy as fuzz

#Handles the generation of triangular membership functions.
class FuzzySetGenerator:

    def __init__(self, data_range, n_sets):
        self.data_range = np.array(data_range, dtype=np.float64)  # Ensure data_range is a float array
        self.n_sets = n_sets  # Number of fuzzy sets
        self.fuzzy_sets = self._generate_triangular_sets()  # Generate fuzzy sets

    # Generate triangular membership functions.
    def _generate_triangular_sets(self):

        # Ensure centers are evenly spaced within the data range
        centers = np.linspace(min(self.data_range), max(self.data_range), self.n_sets)

        # Create triangular membership functions
        return [fuzz.trimf(self.data_range, [c - 1.5, c, c + 1.5]) for c in centers]

    # Return the generated fuzzy sets.
    def get_sets(self):
        return self.fuzzy_sets
