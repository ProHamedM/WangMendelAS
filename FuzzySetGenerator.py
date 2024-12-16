import numpy as np
import skfuzzy as fuzz

#Handles the generation of triangular membership functions.
class FuzzySetGenerator:

    def __init__(self, data_range, n_sets):
        self.data_range = data_range  # Range of values for the variable
        self.n_sets = n_sets  # Number of fuzzy sets per variable
        self.fuzzy_sets = self._generate_triangular_sets()  # Generate fuzzy sets

    # Generate triangular membership functions.
    def _generate_triangular_sets(self):
        centers = np.linspace(self.data_range[0], self.data_range[-1], self.n_sets)
        return [fuzz.trimf(self.data_range, [c - 1.5, c, c + 1.5]) for c in centers]

    # Return the generated fuzzy sets.
    def get_sets(self):
        return self.fuzzy_sets
