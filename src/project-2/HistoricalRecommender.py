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

import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier

class HistoricalRecommender:

    #################################
    # Initialise
    # Set default no. of actions and outcomes,
    # since these can differ between historical data
    # and in other policy
    #
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    ## This is the reward function given as (2.2) in the exercise text
    def _default_reward(self, action, outcome):
        # the (-0.1) factor means giving treatment must outperform placebo by 10%.
        return (-0.1)*action + outcome

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
    #
    # NOTE: this function isn't in use.
    def fit_data(self, data):
        observations = data[:, :128]
        # {0,1} X {0,1} -> {0, 1, 2, 3}
        symptoms = data[:, 128] + data[:, 129]*2
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
    ## We use a K-Nearest Nea,ighbors classifier to fit P(a|x)
    def fit_treatment_outcome(self, data, actions, outcomes):

        # save as historical data
        self.hist_data = data
        self.hist_actions = actions
        self.hist_outcomes = outcomes

        actions = actions.flat

        K = 15
        k_accuracy = np.zeros(K)
        # Change this flag to use the bootstrap.
        use_bootstrap = False
        # This is a bootstrap to find k.
        # See notes on issues with this in my report.
        if use_bootstrap:
            n_samples = 5
            print("Bootstrap for k begin, K = %d" % K)
            for k in range(K):
                print("testing k = %d" % k)
                for i in range(n_samples):
                    train_set, test_set = \
                        train_test_split(data, test_size = 0.2)
                    # pick len(train_set) indexes
                    train_sample_index = \
                        np.random.choice(len(train_set),
                                         len(train_set))
                    test_sample_index = \
                        np.random.choice(len(test_set),
                                         len(test_set))
                    # use picked indexes to pick data points
                    # with replacement for bootstrap
                    k_model = KNeighborsClassifier(n_neighbors=k+1).fit(data[train_sample_index], actions[train_sample_index])
                    k_accuracy[k] += accuracy_score( actions[test_sample_index], k_model.predict(data[test_sample_index]) )
                k_accuracy[k] /= n_samples
            k = np.argmax(k_accuracy[1:]) + 1

            print("Bootstrap END, k = %d" % k)
        else: # don't use bootstrap for k
            # hard set k to avoid running bootstrap all the time
            k = 2
        print("k neighbors: %d" % k)

        self.model = KNeighborsClassifier(n_neighbors = k).fit(data, actions)

    ## Estimate the utility of a specific policy from historical data.
    def estimate_utility(self, data, actions, outcomes, policy=None):
        if policy == None: # use historical data
            T = len(self.hist_data)
            estimate = 0
            for i in range(T):
                r_t = self.reward(self.hist_actions[i],
                                  self.hist_outcomes[i])
                estimate += r_t
            estimate /= T
            return estimate
        else:
            return policy.estimate_utility(data, actions, outcomes)



    # Return a distribution of effects for a given person's data and a specific treatment
    # NOTE: only uses data, as this is the only input to the model
    def predict_proba(self, data, treatment):
        X = data.reshape(1,-1)
        return self.model.predict_proba(X)[0]

    def get_action_probabilities(self, user_data):
        return [self.predict_proba(user_data, a) for a in range(self.n_actions)]


    # Return recommendations for a specific user data
    def recommend(self, user_data):
        # Historical is independent of treatment, so just use 0 as
        # dummy treatment value
        probs = self.predict_proba(user_data, 0)
        if probs[0] > probs[1]:
            action = 0
        else:
            action = 1
        return action

    # Observe the effect of an action
    def observe(self, user, action, outcome):
        return None

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None
