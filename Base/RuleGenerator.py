import numpy as np
import skfuzzy as fuzz

class RuleGenerator:
    """
    Manages Wang-Mendel rule extraction.
    """
    def __init__(self, x1_sets, x2_sets, f_sets):
        self.x1_sets = x1_sets
        self.x2_sets = x2_sets
        self.f_sets = f_sets
        self.rules = []  # List of rules

    def _fuzzify(self, value, sets, data_range):
        """
        Fuzzify a crisp value into fuzzy memberships.
        """
        memberships = [fuzz.interp_membership(data_range, s, value) for s in sets]
        return memberships

    def generate_rules(self, X1, X2, F, x1_range, x2_range, f_range):
        """
        Generate fuzzy rules using Wang-Mendel method.
        """
        for i in range(len(X1)):
            # Fuzzify inputs and output
            x1_memberships = self._fuzzify(X1[i], self.x1_sets, x1_range)
            x2_memberships = self._fuzzify(X2[i], self.x2_sets, x2_range)
            f_memberships = self._fuzzify(F[i], self.f_sets, f_range)

            # Get the indices of the highest membership degree
            x1_set = np.argmax(x1_memberships)
            x2_set = np.argmax(x2_memberships)
            f_set = np.argmax(f_memberships)

            # Create a rule
            rule = (x1_set, x2_set, f_set)
            self.rules.append(rule)

        # Remove duplicate rules
        self.rules = list(set(self.rules))
        print(f"Generated {len(self.rules)} fuzzy rules.")

    def get_rules(self):
        """
        Return the generated rules.
        """
        return self.rules