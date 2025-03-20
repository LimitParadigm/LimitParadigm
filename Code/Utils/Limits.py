import numpy as np
import pandapower as pp
import copy
import Utils.GA_limits as GA_limits

class Limits:
    def __init__(self, input_path, net):
        np.random.seed(19)
        self.input_path = input_path

        self.net = copy.deepcopy(net)
        self.limits = np.zeros((len(self.net.load),2))

    def objective_function(self, limits):
        # Eq. X in paper
        return np.sum(np.abs(self.limits[:, 0]) + self.limits[:, 1]) 
    def calculate_limits(self, net):
        ga_instance = GA_limits.GA_limits(net)
