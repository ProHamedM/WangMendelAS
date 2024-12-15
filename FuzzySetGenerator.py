import numpy as np
import skfuzzy as fuzz

class FuzzySetGenerator:
    """
    Handles the generation of triangular membership functions.
    """
    def __init__(self, data_range, n_sets):
        self.data_range = data_range  # Range of values for the variable
        self.n_sets = n_sets  # Number of fuzzy sets per variable
        self.fuzzy_sets = self._generate_triangular_sets()  # Generate fuzzy sets

    def _generate_triangular_sets(self):
        """
        Generate triangular membership functions.
        """
        centers = np.linspace(self.data_range[0], self.data_range[-1], self.n_sets)
        return [fuzz.trimf(self.data_range, [c - 1.5, c, c + 1.5]) for c in centers]

    def get_sets(self):
        """
        Return the generated fuzzy sets.
        """
        return self.fuzzy_sets
