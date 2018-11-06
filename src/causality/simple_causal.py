## -*- Mode: python -*-
## A simple causal model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n_samples = 1000

# Each policy specifies a different distribution
# y = normal
# a = normla(policy + y)
def non_cause_model(policy, n_samples):
    y_t = np.random.normal(size = n_samples)
    a_t = np.random.normal(policy + y_t)
    return a_t, y_t

# Each policy specifies a different distribution
# a = normal(policy)
# y = normal(a)
def direct_cause_model(policy, n_samples):
    a_t = np.random.normal(policy, 1, size = n_samples)
    y_t = np.random.normal(a_t)
    return a_t, y_t

# Each policy specifies a different distribution
# a = normal(policy)
# y = normal(a)
def sufficient_covariate_model(policy, n_samples):
    x_t = np.random.normal(size = n_samples)
    a_t = np.random.normal(x_t + policy, 1, size = n_samples)
    y_t = np.random.normal(x_t + a_t)
    return a_t, y_t, x_t

# Each policy specifies a different distribution
# a = normal(policy)
# y = normal(a)
def instrumental_variable_model(policy, n_samples):
    x_t = np.random.normal(size = n_samples)
    z_t = np.random.normal(x_t)
    a_t = np.random.normal(z_t + policy, 1, size = n_samples)
    y_t = np.random.normal(x_t + a_t)
    return a_t, y_t, x_t

policy_0 = -1
policy_1 = 1

D_non_cause_0 = non_cause_model(policy_0, n_samples)
D_non_cause_1 = non_cause_model(policy_1, n_samples)

sns.distplot(D_non_cause_0[1])
sns.distplot(D_non_cause_1[1])
plt.title("Non-Cause")
plt.savefig("non-cause.pdf")
plt.show()

D_direct_cause_0 = direct_cause_model(policy_0, n_samples)
D_direct_cause_1 = direct_cause_model(policy_1, n_samples)

sns.distplot(D_direct_cause_0[1])
sns.distplot(D_direct_cause_1[1])
plt.title("Direct cause")
plt.savefig("direct-cause.pdf")
plt.show()


D_non_cause_a = np.concatenate([D_non_cause_0[0], D_non_cause_1[0]])
D_non_cause_y = np.concatenate([D_non_cause_0[1], D_non_cause_1[1]])
sns.jointplot(D_non_cause_a, D_non_cause_y, kind="reg")
plt.title("When the decision is not a cause")
plt.savefig("a-y-non-cause.pdf")
plt.show()
D_direct_cause_a = np.concatenate([D_direct_cause_0[0], D_direct_cause_1[0]])
D_direct_cause_y = np.concatenate([D_direct_cause_0[1], D_direct_cause_1[1]])
sns.jointplot(D_direct_cause_a, D_direct_cause_y, kind="reg")
plt.title("When the decision is a direct cause")
plt.savefig("a-y-direct-cause.pdf")
plt.show()

D_sufficient_covariate_0 = sufficient_covariate_model(policy_0, n_samples)
D_sufficient_covariate_1 = sufficient_covariate_model(policy_1, n_samples)

sns.distplot(D_sufficient_covariate_0[1])
sns.distplot(D_sufficient_covariate_1[1])
plt.title("Sufficient covariate")
plt.savefig("sufficient.pdf")
plt.show()


D_instrumental_variable_0 = instrumental_variable_model(policy_0, n_samples)
D_instrumental_variable_1 = instrumental_variable_model(policy_1, n_samples)

sns.distplot(D_instrumental_variable_0[1])
sns.distplot(D_instrumental_variable_1[1])
plt.title("Instrumental variable")
plt.savefig("instrumental.pdf")
plt.show()
