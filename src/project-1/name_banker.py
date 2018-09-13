import numpy as np

class NameBanker:
    
    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        self.data = [X, y]

    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate
        return

    # Predict the probability of failure for a specific person with data x
    def predict_proba(self, x):
        return 0

    # dummy value, assume 80% of loans are repaid
    def get_proba(self):
        self.proba = 0.1
        return self.proba

    # THe expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    def expected_utility(self, x, action):
        duration = x[0]
        amount = x[1]
        paid_off = True
        rate = self.rate
        return_win = amount*(1+rate)**duration
        return_loss = -amount
        success_prob = self.get_proba()

        expected_return = (success_prob*return_win +
                           (1-success_prob)*return_loss)

        # Assume purely that if we get expect anything more than
        # the original amount back, we grant the loan. In practice,
        # you'd likely have a margin so you're making at least say 5%
        # on every loan
        return_margin = 0
        if (expected_return + return_margin) > amount:
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
