import numpy as np
import pandapower as pp
import copy
import Utils.GA_limits as GA_limits

class Limits:
    def __init__(self, network):
        np.random.seed(19)

        self.network = network
        self.net = copy.deepcopy(network.net)
        self.limits = np.zeros(self.network.limits_shape)

    def safety_verification(self, limits, deflatten = False, debug=False):
        """
        Checks if the network is in a safe state after applying DOEs.
        Returns 1 if voltage and current levels are within safe limits, otherwise 0.
        """
        limits = limits if deflatten == False else self.reshape_function(limits)
        safe = False
        for i in [0,1]:
            self.net.load.p_mw = limits[:, i] / 1000
            try:
                pp.runpp(self.net)
                voltages_ok = self.net.res_bus.vm_pu.between(0.95, 1.05).all()
                currents_ok = self.net.res_line.loading_percent.max() < 100

                if voltages_ok and currents_ok:
                    safe = True
                else:
                    if(debug):
                        print(f"Issue detected: \n \
                            - Voltage: Min: {self.net.res_bus.vm_pu.min()}. Max: {self.net.res_bus.vm_pu.max()}, \n \
                            - Current: Min: {self.net.res_line.loading_percent.min()}. Max: {self.net.res_line.loading_percent.max()}")
                    return False
            except pp.powerflow.LoadflowNotConverged:
                print("Power flow calculation failed. Network is unstable!")
                return False  # Network is unstable if power flow does not converge.
        return safe
    def objective_function(self, limits, deflatten=False):
        # Eq. X in paper
        limits = limits if deflatten==False else self.reshape_function(limits)
        return np.sum(np.abs(limits[:, 0]) + limits[:, 1]) 
    def calculate_limits(self):
        ga_instance = GA_limits.GA_limits(self.network, self)

        ga_instance = ga_instance.runGA()
        return ga_instance

    def reshape_function(self, limits):
        return np.array(limits).reshape(self.network.limits_shape, order='F')
    
    def check_energy_inside_limits(self, energy_usage):
        for i, energy in enumerate(energy_usage):
            if self.limits[i,0] > energy or self.limits[i,1] < energy:
                return False
        return True

    def store_limits(self, limits, path):
        np.save(path, limits)
    def load_limits(self, path):
        self.limits = np.load(path)
