import numpy as np
import skfuzzy as fuzz

#Handles the generation of triangular membership functions.
class FuzzySetGenerator:
    def __init__(self, n_sets, x_range=None, f_range=None):
        self.n_sets = n_sets
        self.x_range = x_range
        self.f_range = f_range

    def get_sets(self):
        if self.x_range:
            return self._generate_sets(self.x_range)
        elif self.f_range:
            return self._generate_sets(self.f_range)
        else:
            raise ValueError("Either x_range or f_range must be provided.")

    def _generate_sets(self, value_range):
        # Generate fuzzy sets (for example, triangular memberships)
        start, end = value_range
        step = (end - start) / (self.n_sets - 1)
        sets = []

        for i in range(self.n_sets):
            center = start + i * step
            sets.append(self._generate_triangular_set(center, step))

        return sets

    @staticmethod
    def _generate_triangular_set(center, step):
        left = center - step
        right = center + step
        return [left, center, right]

