import math
import numpy as np
import os
import pandapower as pp
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import pf_res_plotly
import pandapower.topology as top
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import itertools
from tqdm import tqdm #loading bar
import pickle 

class Network:
    def __init__(self, input_path):
        # Conventions:
        # - Use kWh for timeseries (easier to understand even it pandapower asks for MWh at the end)
        np.random.seed(19)
        self.input_path = input_path

        self.avalilable_phases = ['A']
        self.len_timeseries = 4*24*365 #to change depending on the availbale data
        self.dict_phasecode_to_number = {'Monophasé (sans neutre)':1, 'Monophasé':1, 'Triphasé':2, 'Tétraphasé':3}
        self.pv_scaling_factor = 712 #2850/4=712 -> energy produced by a system of 1 kWhp in an year. 
        self.pv_installation_sizes = np.array([4, 16])#Ref: https://www.yesenergysolutions.co.uk/advice/how-much-energy-solar-panels-produce-home
        self.pv_installation_annual = self.pv_installation_sizes * self.pv_scaling_factor 
        self.ev_scaling_factor = 350
        self.ev_installation_sizes = np.array([3.5, 18]) #Ref: TODO
        self.ev_installation_annual = self.ev_installation_sizes * self.ev_scaling_factor

        self.consumption_file = pd.read_excel(os.path.join(self.input_path, "RESA", "anonymized_consumption_file.xlsx"))
        self.list_load_timeseries = pd.read_csv(os.path.join(input_path, 'RESA', 'anonymized_Load_SM_timeseries.csv'), sep=',', index_col=0).reset_index()
        self.load_timeseries_default = pd.read_csv(os.path.join(self.input_path, 'Timeseries', '1-LV-rural2--1-sw', 'LoadProfile.csv'), sep=';').reset_index()
        self.list_pv_timeseries = pd.read_csv(os.path.join(input_path, 'RESA', 'anonymized_PV_SM_timeseries.csv'), sep=',', index_col=0).reset_index()
        self.pv_timeseries_default = pd.read_csv(os.path.join(self.input_path, 'Timeseries', '1-LV-rural2--1-sw', 'RESProfile.csv'), sep=';').reset_index()
        self.evhp_timeseries = pd.read_excel(os.path.join(self.input_path, 'Timeseries', 'HPEV timeseries.xlsx')).reset_index()[:self.len_timeseries]
        self.ev_timeseries_high = pd.read_csv(os.path.join(self.input_path, 'Timeseries', 'EV_High.csv'),  index_col=0)[:self.len_timeseries]['High']

        self.net, self.choosable_buses = self.import_network(self.input_path)
        self.number_customers = len(self.net.load)
        self.feeders = [270,61]
        # self.assign_feeders(self.feeders) #Depends on the network (The first bus(es) after the substation low-voltage bus )
        self.temp_P = None
        self.feeder_colors = np.random.choice(list(mcolors.CSS4_COLORS.keys()), len(self.feeders))
        
        max_consumption_power = 6 #kW
        self.limits_shape = (self.number_customers,2)
        self.contractual_power = np.ones(self.limits_shape) * [0,max_consumption_power]
        self.contractual_limits = self.contractual_power


        self.timesteps = list(range(self.len_timeseries))
        self.number_timesteps = len(self.timesteps)
        self.load_timeseries(plot_timeseries=False)
        self.assign_ts_to_phase()
        self.feeder_eans = {
                f: self.net.asymmetric_load.loc[self.net.asymmetric_load['feeder'] == f, 'ean'].tolist()
                for f in range(len(self.feeders)) }
        self.considered_feeder = None
        self.feeder_index_eans = {
                f: self.net.asymmetric_load.loc[self.net.asymmetric_load['feeder'] == f, 'ean'].index.tolist()
                for f in range(len(self.feeders)) }
        self.considered_index_eans = None


    def import_network(self, input_path):
        net = pp.from_pickle(os.path.join(input_path, 'anonymized_net.p'))
        choosable_buses = list(net.load.bus)

        pp.create_loads(net, list(net.asymmetric_load['bus']), p_mw=0)
        columns_to_copy = ['ean', 'feeder', 'ann_cons', 'probPV', 'probEV', 'hh_surf', 'hh_size']
        for c in columns_to_copy:
            net.load[c] = list(net.asymmetric_load[c])
        net.load['ph_load'] = 'A'
        net.load['ph_pv'] = ''
        net.load['ph_ev'] = ''

        # For 3-phase PF (not needed for now)
        columns_to_del = ['s_sc_max_mva', 'rx_max', 'x0x_max', 'r0x0_max']
        for c in columns_to_del:
            del net.ext_grid[c]
        columns_to_del = ['vector_group', 'vk0_percent', 'vkr0_percent', 'mag0_percent', 'mag0_rx', 'si0_hv_partial']
        for c in columns_to_del:
            del net.trafo[c]
        columns_to_del = ['r0_ohm_per_km', 'c0_nf_per_km', 'x0_ohm_per_km']
        for c in columns_to_del:
            del net.line[c]

        # Remove old tables
        net.asymmetric_load.drop(net.asymmetric_load.index, inplace=True)
        net.asymmetric_sgen.drop(net.asymmetric_sgen.index, inplace=True)

        # self.NetInfo(net)
        # simple_plotly(net, bus_color=net.bus['color'])
        return net, choosable_buses
    def NetInfo(self, net):
        #Print various statistics about the network.
        print(f'There are {len(net.ext_grid)} substation(s)')
        print(f'There are {len(net.trafo)} transformer(s). Rated power: {net.trafo.sn_mva.values} kVAR')
        print(f'There are {len(net.bus)} nodes')
        print(f"There are {len(net.line)} lines")
        print(f"There are {np.sum(net.bus['info']=='customer')} customers")

    def get_phasecode_from_number(self, number_phases):
        value = {i for i in self.dict_phasecode_to_number if self.dict_phasecode_to_number[i]==number_phases}
        return value

    def get_phase_splitting_values(self, number_phases):
        if number_phases==1:
            return [1]
        else:
            values = np.random.normal(1/number_phases, 0.05, number_phases)
            values = values / values.sum()
            return values

    def shift_timeseries(self, shifting_value):
        if(shifting_value!=0):
            period = np.random.randint(-shifting_value, shifting_value)
        else:
            period = 0
        return period
    def load_timeseries(self, plot_timeseries=True):
        assigned_load_timeseries = pd.DataFrame() #Better a Pandas DF: rows-timesteps, columns-EANs
        
        def assign_find_closest_load_ts(customers_wo_ts, assigned_load_timeseries, list_load_timeseries, avalilable_ts_for_ean):
            timeseries_default1 = self.load_timeseries_default['H0-A_pload'][:self.len_timeseries]
            timeseries_default2 = self.load_timeseries_default['H0-B_pload'][:self.len_timeseries]
            timeseries_default = [timeseries_default1, timeseries_default2]
            for i,c in enumerate(customers_wo_ts):
                ean = c.ean
                annual_consumption = c['ann_cons']
                phases = c['ph_load']
                phasecode = self.get_phasecode_from_number(len(phases))

                closest_ean = None
                closest_diff = np.inf
                # Find the closest matching timeseries based on consumption and phasecode
                for _,other_c in self.consumption_file[(self.consumption_file['RACCORD_CLIENT']==phasecode) & (self.consumption_file['EAN'].isin(avalilable_ts_for_ean)) ].iterrows():
                    other_ean = str(other_c['EAN'])
                    ts_consumption = other_c['Total Actif calcule']
                    ts_consumption = ts_consumption if ts_consumption is not None else other_c['P_CONTRACTUELLE']

                    # Calculate difference in consumption and check phase compatibility
                    consumption_diff = abs(annual_consumption - ts_consumption)
                    if consumption_diff < closest_diff:
                        closest_ean = other_ean
                        closest_diff = consumption_diff

                if(closest_ean):
                    ts = list_load_timeseries[f"{closest_ean}_A"]
                else:
                    ts = timeseries_default[i%len(timeseries_default)] #Use a standard one
                tmp_load_timeseries = self.normalize_time_series(ts, annual_consumption)
                assigned_load_timeseries[f'{ean}_A'] = tmp_load_timeseries

            return assigned_load_timeseries
        
        avalilable_ts_for_ean = [i[:-2] for i in self.list_load_timeseries.columns.values[1:]]
        avalilable_ts_for_ean = set(avalilable_ts_for_ean)

        customers_wo_ts = []
        for _,c in self.net.load.iterrows():
            ean = c['ean']
            if(ean in avalilable_ts_for_ean):
                phases = c['ph_load']
                annual_consumption = c['ann_cons']
                ts = self.list_load_timeseries[f"{ean}_A"]
                tmp_load_timeseries = self.normalize_time_series(ts, annual_consumption)[:self.len_timeseries]
                assigned_load_timeseries[f"{ean}_A"] = tmp_load_timeseries
            else:
                customers_wo_ts.append(c)
        assigned_load_timeseries =  assign_find_closest_load_ts(customers_wo_ts, assigned_load_timeseries, self.list_load_timeseries, avalilable_ts_for_ean)


        def assign_find_closest_pv_ts(customers_wo_ts, assigned_timeseries):
            timeseries_default1 = self.pv_timeseries_default['PV1'][:self.len_timeseries]
            timeseries_default2 = self.pv_timeseries_default['PV3'][:self.len_timeseries]
            timeseries_default3 = self.pv_timeseries_default['PV7'][:self.len_timeseries]
            timeseries_default = [timeseries_default1, timeseries_default2, timeseries_default3]

            pv_pen_rate = 0.85 #[0,1]
            current_pv_pen_rate =  len(customers_wo_ts) / len(self.net.load)
            delta_pv_pen_rate = pv_pen_rate # pv_pen_rate - current_pv_pen_rate #TODO: fix to a different amount
            choosable_buses_missing_pv = [i.bus for i in customers_wo_ts]
            self.net = self.PVinstallation(self.net, choosable_buses_missing_pv, delta_pv_pen_rate)
            for j,c in enumerate(customers_wo_ts):
                ean = c['ean']
                c = self.net.load[self.net.load['ean'] == ean]
                phases = c['ph_pv'].values[0]
                if(phases is not None):
                    annual_prod = c['ann_pv_prod'].values[0]

                    period = self.shift_timeseries(4*2)
                    factor = np.random.random(1)*0.2+0.8
                    tmp_load_timeseries = self.scale_time_series(timeseries_default[j%len(timeseries_default)], annual_prod, self.pv_scaling_factor)
                    assigned_timeseries[f'{ean}_A'] = -tmp_load_timeseries.shift(periods=period, fill_value=0) * factor
            return assigned_timeseries

        assigned_pv_timeseries = pd.DataFrame()
        customers_wo_ts = []
        avalilable_ts_for_ean = set(self.list_pv_timeseries.columns.values[1:])
        for j,c in self.net.load.iterrows():
            ean = c['ean']
            if(ean in avalilable_ts_for_ean):
                annual_prod = np.sum(self.list_pv_timeseries[ean])
                phases_load = c['ph_load']

                # Decide the number of phases based on the annual consumption
                number_phase = 1 if annual_prod <= self.pv_installation_annual[0] else 3 #It can generate conflicts with the load phase
                number_phase = min(number_phase, len(phases_load)) #so avoid that the load is 1-phase but the PV 3-phase
                
                phases = phases_load if number_phase < len(self.avalilable_phases) else  self.avalilable_phases[:number_phase]
                
                self.net.load.at[j, 'ph_pv'] = ''.join(phases)
                self.net.load.at[j, 'ann_pv_prod'] = int(annual_prod)
                
                #Update limits
                self.contractual_limits[j,0] -= self.pv_installation_sizes[0]
                
                tmp_pv_timeseries = self.normalize_time_series(self.list_pv_timeseries[ean], annual_prod)[:self.len_timeseries]
                assigned_pv_timeseries[f"{ean}_A"] = -tmp_pv_timeseries
            else:
                customers_wo_ts.append(c)

        assigned_pv_timeseries =  assign_find_closest_pv_ts(customers_wo_ts, assigned_pv_timeseries)

        def assign_timeseries_EVHP(column, pen_rate):
            if(column=='ev'):
                ts_to_consider = "5000 kWh"

            assigned_timeseries = pd.DataFrame()
            for i ,c in self.net.load.iterrows():
                install_random = np.random.rand()
                if(install_random>pen_rate):
                    continue
                ean = c['ean']
                phases = 'A'
                annual_consumption = self.ev_installation_annual[0]
                self.net.load.at[i, f'ann_{column}_cons'] = annual_consumption
                self.net.load.at[i, f'ph_{column}'] = phases
                #Update limits
                self.contractual_limits[i,1] += self.ev_installation_sizes[0]

                period = self.shift_timeseries(4*5)
                factor = np.random.random(1)*0.2+0.8
                tmp_load_timeseries = self.normalize_time_series(self.evhp_timeseries[ts_to_consider], annual_consumption)
                assigned_timeseries[f"{ean}_A"] = tmp_load_timeseries.shift(periods=period, fill_value=0) * factor
            return assigned_timeseries
        
        ev_pen_rate = 0.7
        assigned_ev_timeseries = assign_timeseries_EVHP('ev', ev_pen_rate)

        self.assigned_load_timeseries = assigned_load_timeseries
        self.assigned_pv_timeseries = assigned_pv_timeseries
        self.assigned_ev_timeseries = assigned_ev_timeseries
        
        total_customers = len(self.net.load)
        pv_customers = (self.net.load['ph_pv'] != '').sum()
        ev_customers = (self.net.load['ph_ev'] != '').sum()
        
        print("Customer Distribution:")
        print(f"Total Customers: {total_customers}")
        print(f"PV Installations: {pv_customers} ({pv_customers/total_customers:.1%})")
        print(f"EV Charging Points: {ev_customers} ({ev_customers/total_customers:.1%})")


    def assign_ts_to_phase(self):
        temp_columns_P = [f"{ean}_{p}" for ean in self.net.load['ean'] for p in self.avalilable_phases]
        tmp_values_P = np.zeros( [self.len_timeseries, len(temp_columns_P)] )
        P = pd.DataFrame(tmp_values_P, columns=temp_columns_P)
        for i,c in self.net.load.iterrows():
            phases = c['ph_load']
            ean = c['ean']
            column = f'{ean}_{phases[0]}'
            P[column] += self.assigned_load_timeseries[column]
            if(c['ph_ev'] != ''):
                P[column] += self.assigned_ev_timeseries[column]
            if(c['ph_pv'] != ''):
                P[column] += self.assigned_pv_timeseries[column]
        self.P = P

    def chose_buses(self, choosable_buses, penetration_rate, probabilities):
        # Function to choose buses for PV scenarios
        elements_to_select = round(len(choosable_buses)*penetration_rate)
        ##Without probabilities
        # ids = np.random.choice(choosable_buses, elements_to_select, replace=False)
        ##With probabilities
        p = probabilities / np.sum(probabilities) #Make the sum of p equal to 1 (Required by numpy)
        ids = np.random.choice(choosable_buses, elements_to_select, replace=False, p=p)
        return ids

    def PVinstallation(self, net, choosable_buses, penetration):
        prob = net.load[net.load.bus.isin(choosable_buses)]['probPV']
        ids = self.chose_buses(choosable_buses, penetration, prob)

        for i in ids:
            phases, annual_prod = 'A', self.pv_installation_annual[0]
            ind = net.load[net.load['bus']==i].index[0]
            net.load.at[ind, 'ann_pv_prod'] = annual_prod
            net.load.at[ind, 'ph_pv'] = ''.join(phases)
            #Update limits
            self.contractual_limits[ind,0] -= self.pv_installation_sizes[0]
        return net
    def EVinstallation(self, net, choosable_buses, penetration):
        ids = self.Chose_buses(self, choosable_buses, penetration, net.load['probEV'])

        for i in ids:
            #Add EV to customer's bus. No need to create a new element, just add 'E' to 'tech' columns
            net.load.loc[net.load['bus'] == i, 'tech'] += 'E'
        return net

    def normalize_time_series(self, timeseries, total_consumption):
        #Use this function in order to set the sum over time of a given timeseries equal to total_consumption
        t = timeseries / np.sum(timeseries) * total_consumption
        return pd.Series(t)
    def scale_time_series(self, timeseries, scaling_factor, tech_installation_scaling_factor):
        #Use this function in order to set the max a given timeseries equal to scaling_factor (Used to PV peak for example)
        t = timeseries * scaling_factor / tech_installation_scaling_factor
        return pd.Series(t)

    def get_meaningful_days_timesteps(self, meaningful_days):
        timeseries_steps = []
        multiplier = 4 * 24 #Number timesteps in a day considering 15 minutes resolutions (1h = 4 * 15minutes)
        for day in meaningful_days:
            if(day<1 or day>365):
                raise("Error in timestep parsing. Required a day in [1,365], received {day}")
            timeseries_steps.extend(list(range((day-1) * multiplier, day * multiplier)))
        
        return sorted(timeseries_steps)

    def plot_P(self,number=-1):
        for _,c in self.net.load[:number].iterrows():
            ean = c['ean']
            for p in self.avalilable_phases:
                plt.plot(self.P.iloc[self.timesteps][f'{ean}_{p}'].values)
            
            print(f"Client ean: {ean}. \n Load: #phases: {c['ph_load']}, consumption: {c['ann_cons']}. \n PV: #phases: {c['ph_pv']}, consumption: {c['ann_pv_prod']}. \n EV: #phases: {c['ph_ev']}, consumption: {c['ann_ev_cons']}.")
            plt.show()

    def get_feeder(self, net, bus, prev_buses = [], prev_lines = []):
        #Find this info plotting the network: _ = simple_plotly(net, aspectratio=(10,8))
        trafo_bus_id = self.net.trafo['lv_bus'].values[0] #may depend on the network
        trafo_line_ids = [] #Depends on the network
        prev_buses.append(bus)

        elems = pp.toolbox.get_connected_elements_dict(net, bus)
        for b in elems['bus']:
            if(b!=trafo_bus_id and b not in prev_buses):
                self.get_feeder(net, b, prev_buses, prev_lines)
        for l in elems['line']:
            if(l not in trafo_line_ids and l not in prev_lines):
                prev_lines.append(l)
    def assign_feeders(self, starting_buses):
        for i,b in enumerate(starting_buses):
            buses = []
            lines = []
            self.get_feeder(self.net, b, buses, lines)

            self.net.bus.loc[self.net.bus.index.isin(buses), 'feeder'] = i
            self.net.line.loc[self.net.line.index.isin(lines), 'feeder'] = i
            self.net.load.loc[self.net.load['bus'].isin(buses), 'feeder'] = i
        self.customers_index_per_feeder = {i:self.net.load.loc[self.net.load['feeder'] == i].index.values for i in range(len(self.feeders))}

    def load_time_series_at_timestep(self, P, current_net, time_step):
        power_factor = 0.98
        for i,l in current_net.asymmetric_load.iterrows():
            for p in self.avalilable_phases:
                load = P[f"{l['ean']}_{p}"].iloc[time_step] / 1000 #to convert in MW
                current_net.asymmetric_load.loc[i, f'p_{p.lower()}_mw'] = load
                current_net.asymmetric_load.loc[i, f'q_{p.lower()}_mvar'] = load * power_factor

    def run_simulations(self, B, output_path):
        lbar = tqdm(total=len(self.timesteps))
        ti = time.time()
        issues_to_consider = ['voltage', 'load_line', 'loss_line', 'load_trafo', 'loss_trafo']

        times=[]
        debug_time_executions = {'generate': [], 'ts': [], 'pf': [], 'res': []}
        P = self.change_P(B)

        results = []
        for f in range(len(self.feeders)):
            res = []
            for timestep in range(len(self.timesteps)):
                r = {}
                for issues in issues_to_consider:
                    r[issues] = [[],[],[]]
                res.append(r)
            results.append(res)
        
        net = pp.pandapowerNet(self.net.copy())

        for t, timestep in enumerate(self.timesteps):
            tii = time.time()
            self.load_time_series_at_timestep(P, net, timestep)
            debug_time_executions['ts'].append(time.time()-tii)
            
            #Run PF
            tii = time.time()
            pp.runpp_3ph(net) #asymmetric (multi phases)
            debug_time_executions['pf'].append(time.time()-tii)
            
            tii = time.time()
            #Voltage. Drop not useful elements
            for f in range(len(self.feeders)):
                buses = net.bus['feeder']==f
                lines = net.line['feeder']==f
                for i,p in enumerate(self.avalilable_phases):
                    voltages =      net.res_bus_3ph    [f'vm_{p.lower()}_pu'][buses].values
                    loading_lines = net.res_line_3ph   [f'loading_{p.lower()}_percent'][lines].values
                    loss_lines =    net.res_line_3ph   [f'p_{p.lower()}_l_mw'][lines].values
                    loading_trafo = net.res_trafo_3ph  [f'loading_{p.lower()}_percent'].values
                    loss_trafo =    net.res_trafo_3ph  [f'p_{p.lower()}_l_mw'].values

                    results[f][t][issues_to_consider[0]][i].append(voltages)
                    results[f][t][issues_to_consider[1]][i].append(loading_lines)
                    results[f][t][issues_to_consider[2]][i].append(loss_lines)
                    results[f][t][issues_to_consider[3]][i].append(loading_trafo)
                    results[f][t][issues_to_consider[4]][i].append(loss_trafo)
            debug_time_executions['res'].append(time.time()-tii)
            lbar.update(1)

        timestep = round((time.time()-ti)*100)/100
        print(f'Elapsed time: {timestep} s ({(timestep/60):.1f} m). Average: {(timestep/len(self.timesteps)):.5f} s/pf')

        lbar.close()
        return debug_time_executions, results