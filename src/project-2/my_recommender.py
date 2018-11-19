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

# NOTES: I haven't had time to thoroughly test this, nor to migrate
# it to a notebook for more thorough comments. This will be done
# by the next deadline.
# The parts solved in this delivery:
# Exercise 1:
#   - K-means implemented in fit(), but not tested. Rest not done.
# Exercise 2:
#   - estimate_utility implemented, but not tested. Not sure
#     this is how it's intended, assumptions are commented below.
#     Error bounds lack.
#   - Improved policies not done.
# TODO put my final touches on this

import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier

class MyRecommender:

    #################################
    # Initialise
    # Set default no. of actions and outcomes,
    # since these can differ between historical data
    # TODO expand
    # and in other policy
    #
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    ## By default, the reward is just equal to the outcome, as the actions play no role
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
        # {0,1} X {0,1} -> {0, 1, 2, 3}
        symptoms = data[:, 128] + data[:, 129]*2
        # TODO params?
        selector = SelectKBest(chi2, k=10).fit(observations, symptoms)
        feature_list = selector.get_support(indices=True) # sel features

        new_data = np.concatenate((observations[:, feature_list],
                                   data[:, 128].reshape(data.shape[0], 1),
                                   data[:, 129].reshape(data.shape[0], 1)),
                                   axis=1)
        self.mask = np.append(feature_list, (128, 129))
        return new_data

    ## Fit a model from patient data, actions and their effects
    ## Here we assume that the outcome is a direct function of data and actions
    ## - sets model to self.model
    def fit_treatment_outcome(self, data, actions, outcome):

        n_samples = 5
        X = np.concatenate((data, actions), axis=1)
        outcome = outcome.flat

        # TODO run this by endex Monday
        # Bootstrapping for K
        K = 15
        k_accuracy = np.zeros(K)
        for k in range(K):
            for i in range(n_samples):
                train_set, test_set = train_test_split(X, test_size = 0.2)
                # pick len(train_set) indexes in (0 , len(train_set)-1)
                train_sample_index = np.random.choice(len(train_set),
                                                      len(train_set))
                test_sample_index = np.random.choice(len(test_set),
                                                     len(test_set))
                # use picked indexes to pick data points with
                # replacement for bootstrap
                k_model = KNeighborsClassifier(n_neighbors=k+1).fit(X[train_sample_index], outcome[train_sample_index])
                k_accuracy[k]+= accuracy_score( outcome[test_sample_index],
                        k_model.predict(X[test_sample_index]) )
            k_accuracy[k] /= n_samples
        k = np.argmax(k_accuracy[1:]) + 1
        # Bootstrap end

        # hard set k to avoid running bootstrap all the time
        # k = 19

        self.model = KNeighborsClassifier(n_neighbors = k).fit(X, outcome)

    ## Estimate the utility of a specific policy from historical data.
    ## If the policy is None, return use the historical policy
    def estimate_utility(self, data, actions, outcome, policy=None):
        if policy is None:
            proba = self.predict_proba(data, actions)[outcome]
            rewa = self.reward(actions, outcome)
            result = proba * rewa
            return result
        else:
            # return policy.estimate_utility(data, actions, outcome)
            return None


    # Return a distribution of effects for a given person's data and a specific treatment
    def predict_proba(self, data, treatment):
        X = np.append(data, treatment).reshape(1,-1)
        return self.model.predict_proba(X)[0]

    # TODO who dis TODO CONT
    def get_action_probabilities(self, user_data):
        return [self.predict_proba(user_data, a) for a in range(self.n_actions)]


    # Return recommendations for a specific user data
    def recommend(self, user):
        print("Recommending")
        # note: this default assigment handles case
        # (user,0,0,0) > # (user,1,0,0)
        self.treatment = 0
        if self.estimate_utility(user,0,0,0) <= self.estimate_utility(user,1,0,0):
            self.treatment = 1
        return self.treatment

    # Observe the effect of an action
    def observe(self, user, action, outcome):
        # TODO implement
        return None
