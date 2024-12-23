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

        for i in range(len(x1)):
            # Fuzzify inputs
            x1_memberships = [fuzz.interp_membership(self.x1_range, s, x1[i]) for s in self.x1_sets]
            x2_memberships = [fuzz.interp_membership(self.x2_range, s, x2[i]) for s in self.x2_sets]

            # Check for invalid memberships
            if all(m == 0 for m in x1_memberships) or all(m == 0 for m in x2_memberships):
                print(f"Warning: No valid membership values for X1={x1[i]}, X2={x2[i]}")
                predictions.append(0)  # Default output
                continue

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

            # Avoid division by zero
            prediction = weighted_sum / total_weight if total_weight > 0 else 0
            predictions.append(prediction)

        return np.array(predictions)