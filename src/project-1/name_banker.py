import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas
import random


class NameBanker:

    def __init__(self):
        self.labels_seen = False
        # created by running once looking at every new piece of data x
        self.senses = {'marital status_A92': 1, 'employment_A75': 1, 'age': 56, 'persons': 1,
            'credit history_A33': 1, 'job_A174': 1, 'employment_A72': 1, 'other installments_A142': 1,
            'foreign_A202': 1, 'checking account balance_A12': 1, 'marital status_A94': 1, 'savings_A65': 1,
            'other debtors_A102': 1, 'housing_A152': 1, 'savings_A63': 1, 'checking account balance_A14': 1,
            'other installments_A143': 1, 'phone_A192': 1, 'employment_A74': 1, 'purpose_A48': 1,
            'property_A123': 1, 'purpose_A42': 1, 'installment': 3, 'purpose_A43': 1, 'savings_A62': 1,
            'job_A173': 1, 'marital status_A93': 1, 'residence time': 3, 'property_A122': 1,
            'employment_A73': 1, 'property_A124': 1, 'purpose_A45': 1, 'housing_A153': 1,
            'credit history_A32': 1, 'job_A172': 1, 'purpose_A44': 1, 'credit history_A31': 1,
            'purpose_A410': 1, 'credits': 3, 'other debtors_A103': 1, 'purpose_A49': 1,
            'checking account balance_A13': 1, 'credit history_A34': 1, 'purpose_A41': 1,
            'savings_A64': 1, 'amount': 18174, 'purpose_A46': 1, 'duration': 68}
        self.i = 0

    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        self.data = [X, y]
        self.model = KNeighborsClassifier(n_neighbors=50)
        self.model.fit(X, y)
        # create dict of [min, max] values for every type of data



    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate
        return

    # Predict the probability of failure for a specific person with data x
    def predict_proba(self, x):
        # data needs to be packed in a list, as the function expects a double array
        prob = self.model.predict_proba([x])
        # unpack, and we only need the first probability p, as the other one is (1-p)
        self.proba = prob[0][0]
        return self.proba

    def get_proba(self):
        return self.proba

    # The expected utility of granting the loan or not. 
    # Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you
    # is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    def expected_utility(self, x, action):
        # See part 2 for comments on safeguard_rate and return_margin
        duration = x[0]
        amount = x[1]
        # we expect the average interest rate to never drop below today's
        # rate, # this could be increased to safeguard against a declining
        # interest rate.
        safeguard_rate = 0
        rate = self.rate - safeguard_rate
        return_win = amount*(1+rate)**duration
        return_loss = -amount
        print("i: %d" % self.i)
        self.i += 1
        # noisify data
        x = self.add_noise(x)
        success_prob = self.predict_proba(x)
        expected_return = (success_prob*return_win +
                           (1-success_prob)*return_loss)

        # Assume purely that if we get expect anything more than
        # the original amount back, we grant the loan. In practice,
        # you'd likely have a margin so you're making at least say 5%
        # on every loan.
        return_margin = 0
        if (expected_return - amount*return_margin) > amount:
            action = 1
        else:
            action = 0
        return action

    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    def get_best_action(self, x):
        # dummy value, action will be set by expected_utility()
        action=0
        action = self.expected_utility(x, action)
        return action
    
    def add_noise(self, x):
        epsilon = 0.1
        print(self.i)
        # flip coin, if hit we change every attribute
        change = random.random()
        if change > 0.5:
            comp = [[label, x[label], np.random.laplace(self.senses[label]/epsilon, size=1)] for label in x.index]
            for label, data, noise in comp:
                x[label] = data+noise
        return x
"""
        self.data_ranges = {'checking account balance_A12': [0, 1], 'age': [19, 75],
            'credits': [1, 4], 'other installments_A143': [0, 1],
            'housing_A152': [0, 1], 'employment_A74': [0, 1], 'purpose_A46': [0, 1],
            'installment': [1, 4], 'property_A122': [0, 1],
            'marital status_A92': [0, 1], 'purpose_A43': [0, 1],
            'other debtors_A102': [0, 1], 'other debtors_A103': [0, 1],
            'checking account balance_A14': [0, 1], 'job_A173': [0, 1],
            'job_A174': [0, 1], 'duration': [4, 72], 'phone_A192': [0, 1],
            'marital status_A94': [0, 1], 'purpose_A42': [0, 1], 'savings_A64': [0, 1],
            'purpose_A48': [0, 1], 'employment_A72': [0, 1], 'purpose_A410': [0, 1],
            'job_A172': [0, 1], 'marital status_A93': [0, 1], 'persons': [1, 2],
            'property_A123': [0, 1], 'other installments_A142': [0, 1],
            'credit history_A33': [0, 1], 'purpose_A41': [0, 1],
            'credit history_A31': [0, 1], 'housing_A153': [0, 1],
            'credit history_A32': [0, 1], 'savings_A65': [0, 1],
            'credit history_A34': [0, 1], 'employment_A73': [0, 1],
            'purpose_A44': [0, 1], 'amount': [250, 18424], 'employment_A75': [0, 1],
            'savings_A62': [0, 1], 'purpose_A45': [0, 1], 'residence time': [1, 4],
            'purpose_A49': [0, 1], 'savings_A63': [0, 1], 'property_A124': [0, 1],
            'checking account balance_A13': [0, 1], 'foreign_A202': [0, 1]}
        self.i = 0
"""
