import numpy as np
from WangMendelAS.Base.FuzzySetGenerator import FuzzySetGenerator
from WangMendelAS.Base.RuleGenerator import RuleGenerator
from WangMendelAS.Base.FuzzyInferenceSystem import FuzzyInferenceSystem
from WangMendelAS.Base.Validation import Validation

# Step 1: Generate Fuzzy Sets
x1_range = (-5, 5)
x2_range = (-5, 5)
f_range = (0, 50)
n_sets = 7

x1_fuzzy_gen = FuzzySetGenerator(x1_range, n_sets)
x2_fuzzy_gen = FuzzySetGenerator(x2_range, n_sets)
f_fuzzy_gen = FuzzySetGenerator(f_range, n_sets)

x1_sets = x1_fuzzy_gen.get_sets()
x2_sets = x2_fuzzy_gen.get_sets()
f_sets = f_fuzzy_gen.get_sets()

# Step 2: Generate Rules Using Training Data
train_x1 = np.random.uniform(-5, 5, 100)
train_x2 = np.random.uniform(-5, 5, 100)
train_f = train_x1**2 + train_x2**2  # True function: f(x1, x2) = x1^2 + x2^2

rule_generator = RuleGenerator(x1_sets, x2_sets, f_sets)
rule_generator.generate_rules(train_x1, train_x2, train_f, x1_range, x2_range, f_range)
rules = rule_generator.get_rules()

# Step 3: Create Fuzzy Inference System
fuzzy_system = FuzzyInferenceSystem(rules, x1_sets, x2_sets, f_sets, x1_range, x2_range, f_range)

# Step 4: Validate the System
test_size = 100  # Number of test data points
validator = Validation(n_repeats=100)
mean_mse = validator.validate(fuzzy_system, test_data=np.zeros(test_size))

print(f"Final Mean Squared Error (MSE) after validation: {mean_mse:.4f}")

