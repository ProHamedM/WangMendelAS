import numpy as np
import skfuzzy as fuzz

# Manage fuzzy inference and prediction
class FuzzyInferenceSystem:
    def __init__(self, rules, x1_sets, x2_sets, f_sets, x1_range, x2_range, f_range):
        self.rules = rules
        self.x1_sets = x1_sets
        self.x2_sets = x2_sets
        self.f_sets = f_sets
        self.x1_range = x1_range
        self.x2_range = x2_range
        self.f_range = f_range

    # Predict outputs for given inputs using fuzzy inference
    def predict(self, x1, x2):
        predictions = []

        # If single inputs are passed, wrap them into arrays
        if isinstance(x1, (int, float, np.float64)):
            x1 = np.array([x1])
        if isinstance(x2, (int, float, np.float64)):
            x2 = np.array([x2])

        for i in range(len(x1)):
            # Fuzzify inputs
            x1_memberships = [fuzz.interp_membership(self.x1_range, s, x1[i]) for s in self.x1_sets]
            x2_memberships = [fuzz.interp_membership(self.x2_range, s, x2[i]) for s in self.x2_sets]

            weighted_sum = 0
            total_weight = 0

            for rule in self.rules:
                x1_set, x2_set, f_set = rule

                # Compute the rule's weight
                weight = min(x1_memberships[x1_set], x2_memberships[x2_set])

                # Use the center of the output set for defuzzification
                center_f = np.mean(self.f_range[f_set * len(self.f_range) // len(self.f_sets):
                                                (f_set + 1) * len(self.f_range) // len(self.f_sets)])

                weighted_sum += weight * center_f
                total_weight += weight

            # Handle division by zero safely
            if total_weight > 0:
                predictions.append(weighted_sum / total_weight)
            else:
                predictions.append(0)  # Append 0 when no rules fire

        return np.array(predictions)