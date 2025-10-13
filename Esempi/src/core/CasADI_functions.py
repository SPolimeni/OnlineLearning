import casadi as ca
import math
import numpy as np
import matplotlib.pyplot as plt

class CasADI_sets:
    def __init__(self,Opts,ConfigParams):

                    
        if 'EL' in Opts.keys():
            Tsim = Opts['EL']['TimeDiscretization']['Tsim']
            dt   = Opts['EL']['TimeDiscretization']['dt']

            self.T      = list((range(int(Tsim/dt))))
            self.dt     = dt*3600
        else:
            self.T      = list((range(int(Opts['Tsim']/Opts['dt']))))
            self.dt     = Opts['dt']*3600

        self.tSteps = len(self.T)
        
        self.LocalNetwork      = 'EL'
        self.ExternalNetworks   = ConfigParams['ExternalNetworks']
        self.NetworkConfig      = ConfigParams['NetworkConfig']
        self.LinkedLocalNet     = [key for key in self.ExternalNetworks.keys() if self.ExternalNetworks[key]['Type'] == 'Local']
        self.InternalVectors    = ConfigParams['NetworkConfig'][self.LocalNetwork]['InternalVector'].split(',')
        self.NetworksExchange   = ConfigParams['NetworksExchange'][self.LocalNetwork]

        self.HierarchicalControl = Opts['HierarchicalControl']
        self.tStepsHier          = int(Opts['H_MPC']['TimeDiscretization']['dt'] / Opts['EL']['TimeDiscretization']['dt']) if Opts['HierarchicalControl'] else self.tSteps


class CasADI_Generators: 
    def __init__(self,ocp,name,GenParams,OptiSets):
        self.name       = name
        self.ocp        = ocp
        self.OptiSets   = OptiSets
        self.Params     = GenParams

        if self.Params['Output1'] != 'NONE':
            self.Out1 = ocp.variable(self.OptiSets.tSteps)
            self.Out1_sol = None

        if self.Params['Output2'] != 'NONE':
            self.Out2 = ocp.variable(self.OptiSets.tSteps)
            self.Out2_sol = None

        self.GenIn          = ocp.variable(self.OptiSets.tSteps)
        self.GenSlackPos    = ocp.variable(self.OptiSets.tSteps) # Slack variable for the generator
        self.GenSlackNeg    = ocp.variable(self.OptiSets.tSteps) # Slack variable for the generator

        self.GenIn_sol   = None

        self.SingleGeneratorConstraints()

    def PowerLimit(self):
        
        if self.Params['Output1'] != 'NONE':
            self.ocp.subject_to(self.Out1 <= self.Params['MaxOut1'])
            self.ocp.subject_to(self.Out1 >= self.Params['MinOut1'])

        if self.Params['Output2'] != 'NONE':
            self.ocp.subject_to(self.Out2 <= self.Params['MaxOut2'])
            self.ocp.subject_to(self.Out2 >= self.Params['MinOut2'])

    def Consumption(self):

        mOut1 = self.Params['mOut1']
        qOut1 = self.Params['qOut1']

        self.ocp.subject_to(self.Out1 == mOut1 * self.GenIn + qOut1)

    def SingleGeneratorConstraints(self):

        self.PowerLimit()
        self.Consumption()
    

class CasADI_Storages:
    def __init__(self,ocp,name,StorParams,OptiSets):  

        self.name       = name
        self.OptiSets   = OptiSets
        self.Params     = StorParams
        self.ocp        = ocp
    
        self.P_ch       = ocp.variable(OptiSets.tSteps)
        self.P_dch      = ocp.variable(OptiSets.tSteps)
        self.P_stor     = ocp.variable(OptiSets.tSteps)  # Power storage variable
        self.dP_abs     = ocp.variable(OptiSets.tSteps)  # Absolute value of the power storage variable
        self.En         = ocp.variable(OptiSets.tSteps+1) 
        self.En0        = ocp.parameter()  # Initial energy in the storage, set by the user
        self.P0         = ocp.parameter()  # Initial power in the storage
        self.dPslack    = ocp.variable(OptiSets.tSteps)  # Slack variable for the storage

        self.SingleStorageConstraints()

    def PowerLimits(self):
        Pmax = self.Params['CrateMax']*self.Params['Capacity']
        Cmin = self.Params['SOCmin']*self.Params['Capacity']
        Cmax = self.Params['SOCmax']*self.Params['Capacity']
        Ramp = self.Params['RampRate']

        self.ocp.subject_to(self.P_ch <= Pmax)
        self.ocp.subject_to(self.P_dch <= Pmax)
        self.ocp.subject_to(self.P_ch >= 0)
        self.ocp.subject_to(self.P_dch >= 0)
        self.ocp.subject_to(self.En >= Cmin)
        self.ocp.subject_to(self.En <= Cmax)

        self.ocp.subject_to(self.P_stor == self.P_dch - self.P_ch)  

        # Ramp constraints
        self.ocp.subject_to(self.dP_abs[0] >= self.P_stor[0] - self.P0)
        self.ocp.subject_to(self.dP_abs[0] >= -(self.P_stor[0] - self.P0))
        self.ocp.subject_to(self.dP_abs[1:] >= self.P_stor[1:] - self.P_stor[:-1])
        self.ocp.subject_to(self.dP_abs[1:] >= -(self.P_stor[1:] - self.P_stor[:-1]))
        self.ocp.subject_to(self.dP_abs <= Ramp + self.dPslack)
        self.ocp.subject_to(self.dPslack >= 1e-8)
        self.ocp.subject_to(self.dPslack <= 5)
        self.ocp.subject_to(self.dP_abs >= 1e-8)
   
        
    def CapacityEvolultion(self):
        Eff_ch  = self.Params['Eff_ch']
        Eff_dch = self.Params['Eff_dch']

        for t in range(1,self.OptiSets.tSteps+1):
            self.ocp.subject_to(self.En[t] == self.En[t-1] + self.OptiSets.dt/3600 * (Eff_ch * self.P_ch[t-1] - self.P_dch[t-1]/Eff_dch))
        self.ocp.subject_to(ca.vec(self.P_ch)*ca.vec(self.P_dch) == 0) 

        self.ocp.subject_to(self.En[0] == self.En0)  # Initial energy in the storage

    def SingleStorageConstraints(self):

        self.PowerLimits()
        self.CapacityEvolultion()

class CasADI_Renewables: 
    def __init__(self,ocp,name,RESParams,OptiSets):
        self.name       = name
        self.ocp        = ocp
        self.OptiSets   = OptiSets
        self.Params     = RESParams

        self.Power     = ocp.parameter(OptiSets.tSteps)  # Power output of the renewable energy source

class CasADI_ElectricalLoads:
    def __init__(self,ocp,name,OptiSets):
        self.ocp            = ocp
        self.name           = name
        self.OptiSets       = OptiSets
        self.Power          = ocp.parameter(OptiSets.tSteps)
        self.Power_sol      = None  # Solution variable for the power output of the electrical load

class CasADI_Network:
    def __init__(self,ocp,Generators,Storages,Renewables,ElectricalLoads,OptiSets):
        self.ocp                = ocp
        self.Generators         = Generators
        self.Renewables         = Renewables
        self.Storages           = Storages
        self.ElectricalLoads    = ElectricalLoads
        self.OptiSets           = OptiSets

        self.Obj = ocp.variable()

        NetName = self.OptiSets.LocalNetwork 
        self.P_track        = {}
        self.P_exch         = {}
        self.P_exch_sol     = {}  # Exchange power solution variable
        self.E_int          = ocp.variable(self.OptiSets.tSteps+1)    # Internal energy variable
        self.E_track        = ocp.parameter(self.OptiSets.tSteps)   # Energy tracking parameter
        for net in self.OptiSets.ExternalNetworks.keys():
            if self.OptiSets.ExternalNetworks[net]['Vector'] == self.OptiSets.NetworkConfig[NetName]['ExchangeVector']:
                self.P_exch[net]        = ocp.variable(self.OptiSets.tSteps)
                self.P_track[net]       = ocp.parameter(self.OptiSets.tSteps)
                self.P_exch_sol[net]    = None  # Solution variable for the exchange power

        self.CostParams         = {}
        self.CostParams['Buy']  = {}
        self.CostParams['Sell'] = {}
        for n in self.OptiSets.ExternalNetworks.keys():
            self.CostParams['Buy'][n]  = ocp.parameter(self.OptiSets.tSteps)
            self.CostParams['Sell'][n] = ocp.parameter(self.OptiSets.tSteps)

        self.BalanceConstraints()
        self.ObjectiveFunction()

    def BalanceConstraints(self):
        
        Production = np.zeros(self.OptiSets.tSteps)
        for g in self.Generators:
            if self.Generators[g].Params['Output1'] == 'EE_AC':
                Production += self.Generators[g].Out1
            elif self.Generators[g].Params['Output2'] == 'EE_AC':
                Production += self.Generators[g].Out2
        for r in self.Renewables:
            if self.Renewables[r].Params['Output'] == 'EE_AC':
                Production += self.Renewables[r].Power

        for s in self.Storages:
            Production += self.Storages[s].P_stor 

        Exchange = np.zeros(self.OptiSets.tSteps)
        NetName = self.OptiSets.LocalNetwork 
        for net in self.OptiSets.ExternalNetworks.keys():
            if self.OptiSets.ExternalNetworks[net]['Vector'] == self.OptiSets.NetworkConfig[NetName]['ExchangeVector']:
                Exchange += self.P_exch[net] 

        TotalLoad = np.zeros(self.OptiSets.tSteps)
        for load in self.ElectricalLoads.keys():
            TotalLoad += self.ElectricalLoads[load].Power

        self.ocp.subject_to(Production + Exchange == TotalLoad)

    def ObjectiveFunction(self):
        Costs = 0
        for g in self.Generators.keys():
            InputVector = self.Generators[g].Params['Input'] 
            for n in self.OptiSets.ExternalNetworks.keys():
                if InputVector in self.OptiSets.ExternalNetworks[n]['Vector']: 
                    Costs += ca.sum1(ca.vec(self.CostParams['Buy'][n]) * ca.vec(self.Generators[g].GenIn)) #* self.OptiSets.dt*3600 #*self.OptiSets.rho_cp*self.Generators[g].m #*self.OptiSets.dt*3600

        InternalEnergy = 0 
        for s in self.Storages.keys():
            InternalEnergy += self.Storages[s].En
        self.ocp.subject_to(self.E_int == InternalEnergy)  # Set the internal energy variable

        for s in self.Storages.keys():
            Costs += ca.sum1(ca.vec(self.Storages[s].dPslack)*1e2)
            Costs += ca.sum1(1e-2*ca.vec(self.Storages[s].dP_abs))  # Add a small cost for the storage power
        
        if self.OptiSets.HierarchicalControl:
            TrackCosts = ca.sum1((self.E_int[1:] - self.E_track)**2)  
            for net in self.P_track.keys():
                TrackCosts += ca.sum1((self.P_exch[net]-self.P_track[net])**2) # TODO: add weigthts for each network
        else:
            TrackCosts = 1e-3*ca.sum1((self.E_int[0] - self.E_int[-1])**2)  # Add a tracking cost for the internal energy

            self.P_exchPos = {}
            self.P_exchNeg = {}
            for net in self.OptiSets.ExternalNetworks.keys():
                if net == 'ElectricGrid': # Quick Fix
                    self.P_exchPos[net] = self.ocp.variable(self.OptiSets.tSteps)
                    self.P_exchNeg[net] = self.ocp.variable(self.OptiSets.tSteps)
                    self.ocp.subject_to(self.P_exch[net] == self.P_exchPos[net] - self.P_exchNeg[net])

                    self.ocp.subject_to(self.P_exchPos[net] >= 1e-8)
                    self.ocp.subject_to(self.P_exchNeg[net] >= 1e-8)
                    Costs += ca.sum1(ca.vec(self.CostParams['Buy'][net]) * ca.vec(self.P_exchPos[net]))  # Add the cost for buying energy from the network
                    Costs -= ca.sum1(ca.vec(self.CostParams['Sell'][net]) * ca.vec(self.P_exchNeg[net]))  # Add the cost for selling energy to the network

        self.ocp.subject_to(self.Obj == Costs*self.OptiSets.dt/3600 + TrackCosts) 
        self.ocp.minimize(self.Obj)

    def SetInitialConditions(self):
        for s in self.Storages:
            self.ocp.subject_to(self.Storages[s].En[0] == self.Storages[s].Params['SOCinit']*self.Storages[s].Params['Capacity'])
        
        for g in self.Generators:
            if self.Generators[g].Params['Output1'] != 'NONE':
                self.ocp.subject_to(self.Generators[g].Out1[0] == self.Generators[g].Params['MinOut1'])
            if self.Generators[g].Params['Output2'] != 'NONE':
                self.ocp.subject_to(self.Generators[g].Out2[0] == self.Generators[g].Params['MinOut2'])

    def GetSolutionValues(self,ocp,Solved=True):
        
        GetVals = ocp.value if Solved else ocp.debug.value
        for g in self.Generators.keys():
            if self.Generators[g].Params['Output1'] != 'NONE':
                self.Generators[g].Out1_sol = GetVals(self.Generators[g].Out1)
            if self.Generators[g].Params['Output2'] != 'NONE':
                self.Generators[g].Out2_sol = GetVals(self.Generators[g].Out2)
            self.Generators[g].GenIn_sol = GetVals(self.Generators[g].GenIn)

        for s in self.Storages.keys():
            self.Storages[s].P_ch_sol   = GetVals(self.Storages[s].P_ch)
            self.Storages[s].P_dch_sol  = GetVals(self.Storages[s].P_dch)
            self.Storages[s].P_stor_sol = GetVals(self.Storages[s].P_stor)
            self.Storages[s].En_sol     = GetVals(self.Storages[s].En)

        for l in self.ElectricalLoads.keys():
            self.ElectricalLoads[l].Power_sol = GetVals(self.ElectricalLoads[l].Power)

        NetName = self.OptiSets.LocalNetwork 
        for net in self.OptiSets.ExternalNetworks.keys():
            if self.OptiSets.ExternalNetworks[net]['Vector'] == self.OptiSets.NetworkConfig[NetName]['ExchangeVector']:
                self.P_exch_sol[net] = GetVals(self.P_exch[net])


    def ConvertSolutionToDict(self):
        solution_dict = {}
        for g in self.Generators.keys():
            solution_dict[g] = {
                'Out1'  : self.Generators[g].Out1_sol,
                # 'Out2'  : self.Generators[g].Out2_sol,
                'GenIn' : self.Generators[g].GenIn_sol
            }

        for s in self.Storages.keys():
            solution_dict[s] = {
                'P_ch'  : self.Storages[s].P_ch_sol,
                'P_dch' : self.Storages[s].P_dch_sol,
                'P_stor': self.Storages[s].P_stor_sol,
                'En'    : self.Storages[s].En_sol
            }

        for l in self.ElectricalLoads.keys():
            solution_dict[l] = {
                'Power' : self.ElectricalLoads[l].Power_sol
            }
        
        NetName = self.OptiSets.LocalNetwork 
        for net in self.OptiSets.ExternalNetworks.keys():
            if self.OptiSets.ExternalNetworks[net]['Vector'] == self.OptiSets.NetworkConfig[NetName]['ExchangeVector']:
                solution_dict[net] = {
                    'P_exch' : self.P_exch_sol[net]
                }

        self.solution_dict = self.convert_arrays_to_lists(solution_dict)

    def convert_arrays_to_lists(self, data=None):

        if data is None:
            data = self.solution_dict

        if isinstance(data, dict):
            # If the data is a dictionary, recursively process its values
            return {key: self.convert_arrays_to_lists(value) for key, value in data.items()}
        elif isinstance(data, list):
            # If the data is a list, recursively process its elements
            return [self.convert_arrays_to_lists(item) for item in data]
        elif hasattr(data, "tolist"):  
            # Check if the object has a `tolist` method (e.g., NumPy or CasADi arrays)
            return data.tolist()
        else:
            # If it's neither a dict, list, nor array, return it as is
            return data
        