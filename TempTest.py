import numpy as np
from WangMendelAS.Base.RuleGenerator import RuleGenerator
from WangMendelAS.Base.FuzzySetGenerator import FuzzySetGenerator

# Define input and output ranges
x_range = np.linspace(-5, 5, 100)
f_range = np.linspace(0, 50, 100)

# Generate fuzzy sets for inputs and outputs
n_sets = 7
fuzzy_gen_x1 = FuzzySetGenerator(x_range, n_sets)
fuzzy_gen_x2 = FuzzySetGenerator(x_range, n_sets)
fuzzy_gen_f = FuzzySetGenerator(f_range, n_sets)

x1_sets = fuzzy_gen_x1.get_sets()
x2_sets = fuzzy_gen_x2.get_sets()
f_sets = fuzzy_gen_f.get_sets()

# Generate training data
n_train = 1681
X1 = np.random.uniform(-5, 5, n_train)
X2 = np.random.uniform(-5, 5, n_train)
F = X1**2 + X2**2  # Output: F(x1, x2) = x1^2 + x2^2

# Initialize RuleGenerator
rule_gen = RuleGenerator(x1_sets, x2_sets, f_sets)

# Generate rules (explicitly pass variables with keyword arguments)
rule_gen.generate_rules(
    x1=X1,
    x2=X2,
    f=F,
    x1_range=x_range,
    x2_range=x_range,
    f_range=f_range
)

# Retrieve and display rules
rules = rule_gen.get_rules()
print(f"Generated {len(rules)} rules:")
for rule in rules[:10]:  # Display the first 10 rules
    print(f"IF x1 is Set {rule[0]} AND x2 is Set {rule[1]} THEN output is Set {rule[2]}")
