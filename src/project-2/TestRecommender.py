import numpy as np
import pandas

# NOTE: changed to reflect reward used in exercise text
def default_reward_function(action, outcome):
    return (-0.1)*action + outcome

# NOTE: utility is divided by number of steps, to enable 
# comparison with the estimates using historical data
def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    # policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
    return u/T

features = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2
sex = features[:, 0]
smoker = features[:, 1]
gene_expr = features[:, 2:128]

import data_generation
import ImprovedRecommender
import HistoricalRecommender
import AdaptiveRecommender

hist_fac = HistoricalRecommender.HistoricalRecommender
imp_fac = ImprovedRecommender.ImprovedRecommender
ad_fac = AdaptiveRecommender.AdaptiveRecommender

print("---- Estimating utility for historical recommender on historical data ----")
policy_hist = hist_fac(len(actions), len(outcome))
policy_hist.fit_treatment_outcome(features, actions, outcome)
# arguments don't matter since historical data is used anyway
hist_hist_E_U = policy_hist.estimate_utility(0,0,0)
print("Estimated utility: %.4f" % hist_hist_E_U)

print("---- Estimating utility for improved recommender on historical data ----")
policy_imp = imp_fac(len(actions), len(outcome))
policy_imp.fit_treatment_outcome(features, actions, outcome)
imp_hist_E_U = policy_imp.estimate_utility(features, actions, outcome)
print("Estimated utility: %.4f" % imp_hist_E_U)

# NOTE: change this to determine recommender used for the rest
# policy_factory = hist_fac
policy_factory = ad_fac

import time

## First test with the same number of treatments
start_time = time.time()
print()
print("==== Start of original test bench, using %s ====" % policy_factory.__name__)
print("---- Testing with only two treatments ----")

print("Setting up simulator")
generator = data_generation.DataGenerator(matrices="./generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
## Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
## Run an online test with a small number of actions
print("Running an online test")
n_tests = 10000
# n_tests = 100
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward: %.4f" % result)
print("Final analysis of results")
policy.final_analysis()
end_time = time.time()
print("time taken to test: %.3f seconds" % (end_time - start_time))

start_time = time.time()
## First test with the same number of treatments
print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
print("Setting up simulator")
generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
## Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
## Run an online test with a small number of actions
print("Running an online test")
# n_tests = 1000
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward: %.4f" % result)
print("Final analysis of results")
policy.final_analysis()
end_time = time.time()
print("time taken to test: %.3f seconds" % (end_time - start_time))



