# -*- Mode: python -*-
# A simple reference recommender
#
#
# This is a medical scenario with historical data.
#
# General functions
#
# - set_reward
#
# There is a set of functions for dealing with historical data:
#
# - fit_data
# - fit_treatment_outcome
# - estimate_utiltiy
#
# There is a set of functions for online decision making
#
# - predict_proba
# - recommend
# - observe

from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class ImprovedRecommender:

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    ## By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        return outcome

    # Set the reward function r(a, y)
    def set_reward(self, reward):
        self.reward = reward

    ##################################
    # Fit a model from patient data.
    #
    # This will generally speaking be an
    # unsupervised model. Anything from a Gaussian mixture model to a
    # neural network is a valid choice.  However, you can give special
    # meaning to different parts of the data, and use a supervised
    # model instead.
    def fit_data(self, data):
        observations = data[:, :128]
        symptoms = data[:,128] + data[:,129]*2
        selector = SelectKBest(chi2, k=10).fit(observations, symptoms)
        feature_list = selector.get_support(indices=True)   #selected features

        new_data = np.concatenate((observations[:,feature_list],
                                   data[:,128].reshape(data.shape[0], 1),
                                   data[:,129].reshape(data.shape[0], 1)),
                                   axis=1)
        self.mask = np.append(feature_list, (128, 129))
        return new_data

    ## Fit a model from patient data, actions and their effects
    ## Here we assume that the outcome is a direct function of data and actions
    ## This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcome):
        #selected_data = self.fit_data(data)

        n_samples = 5
        X = np.concatenate((data, actions), axis=1)
        outcome = outcome.flat

        # Bootstrapping for k
        """K = 15
        k_accuracy = np.zeros(K)
        for k in range(K):
            for i in range(n_samples):
                train_set, test_set = train_test_split(X, test_size=0.2)
                train_sample_index = np.random.choice(len(train_set), len(train_set))
                test_sample_index = np.random.choice(len(test_set), len(test_set))
                k_model = KNeighborsClassifier(n_neighbors=k+1).fit(X[train_sample_index], outcome[train_sample_index])
                k_accuracy[k] += accuracy_score(outcome[test_sample_index], k_model.predict(X[test_sample_index]))
            k_accuracy[k] /= n_samples
        k = np.argmax(k_accuracy[1:]) + 1"""
        k = 19
        self.model = KNeighborsClassifier(n_neighbors=k).fit(X, outcome)


    ## Estimate the utility of a specific policy from historical data (data, actions, outcome),
    ## where utility is the expected reward of the policy.
    ##
    ## If policy is not given, simply use the average reward of the observed actions and outcomes.
    ##
    ## If a policy is given, then you can either use importance
    ## sampling, or use the model you have fitted from historical data
    ## to get an estimate of the utility.
    ##
    ## The policy should be a recommender that implements get_action_probability()
    def estimate_utility(self, data, actions, outcome, policy=None):
        if policy is None:
            return self.predict_proba(data, actions)[outcome]*self.reward(actions, outcome)
        else:
            return None
        


    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        X = np.append(data, treatment).reshape(1,-1)
        return self.model.predict_proba(X)[0]

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        return [self.predict_proba(user_data, a) for a in range(self.n_actions)]


    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        s = np.zeros((self.n_actions, self.n_outcomes))
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                s[i, j] = self.estimate_utility(user_data, i, j)
        action = np.unravel_index(s.argmax(), s.shape)[0]
        return action

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        return None

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None
