import os
import sys

# Add the main directory to sys.path
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(main_dir)

import json
from datetime import datetime
import casadi as ca
import numpy as np
import time

from src.L_MPC.EL.support.ConfiguratorsFn               import ElectricalFacilityConfigurator
from src.H_MPC.Centralized.support.ConfiguratorsFn      import ProfileConfigurator
from src.L_MPC.EL.support.ParamUpdate                   import SetOCPParameters
from src.L_MPC.EL.core.CreateOCP                        import CreateCasADIModel, SetSolverOptions
from utils.Communication.Modbus.Modbus_mgmt             import Modbus_connect

class EL_MPC:
    def __init__(self, Opts):

        self.Opts   = Opts
        self.t_step = 0
        self.Create_EL_mpc()
        if not self.Opts['Debug']:
            self.SetUpModbusConnection()

        self.t_step_hl = 0 #QUICK FIX; REMOVE
        if self.Opts['HierarchicalControl']:
            self.nStepsHier = int(self.Opts['H_MPC']['TimeDiscretization']['dt'] / self.Opts['EL']['TimeDiscretization']['dt'])
            self.t_step_hl  = 0
            self.nSteps_tot = int(self.Opts['H_MPC']['TimeDiscretization']['Tsim'] / self.Opts['EL']['TimeDiscretization']['dt'])
        else: 
            self.nSteps_tot = 1e6  # Placeholder, to be set from the main script

        self.Reference = None
        self.current_day_str = datetime.now().strftime('%Y_%m_%d')
        self.RunOpt = True
        

    def Create_EL_mpc(self):
        self.ConfigParams       = ElectricalFacilityConfigurator(self.Opts)
        self.ProfilesParams     = ProfileConfigurator(self.Opts, self.ConfigParams, Networkname = 'EL')
        self.ocp, self.EL_Net   = CreateCasADIModel(self.Opts, self.ConfigParams)

    def SetUpModbusConnection(self):
        # Placeholder for Modbus connection setup
        self.ModbusClient = Modbus_connect()
        self.ModbusClient.connect()

    def SetPointsWriter(self):
        # Verifica la connessione a Modbus, se non conmnesso si ricconette
        if not self.ModbusClient.connect:
            self.SetUpModbusConnection()

        # Example mapping of Modbus addresses (not used currently)
        ModbusMap = {'Async': 137, 'ESS4': 151, 'MainLoad': [105, 107, 109]} # TODO: Scrivere un file di configurazione per i modbus

        # # self.EL_Net.ConvertSolutionToDict() # TODO: Implement modubus writer
        # # Esempio: indirizzi in ordine corrispondente ai componenti
        # gen_addresses       = [138]
        # storage_addresses   = [153]
        # load_addresses      = [107, 109, 111]

        # if self.t_step >= self.nSteps_tot:

        #     regs    = [151,137] # MainLoad
        #     value   = 0 
        #     for reg in regs:
        #         if reg == 137:
        #             value = 0 
        #         self.ModbusClient.write_float_values(reg, value)

        #     time.sleep(5)

        #     regs    = [139,129]
        #     value   = 0 # Start
        #     for reg in regs:
        #         self.ModbusClient.write_float_values(reg, value)

        # else:

        for gen in self.EL_Net.Generators.keys():
            if gen in self.EL_Net.solution_dict:
                gen_key = self.EL_Net.solution_dict[gen]
                if 'Out1' in gen_key:
                    if self.RunOpt:
                        value = gen_key['Out1'][0]
                    else:
                        value = 0
                    reg   = ModbusMap.get(gen, None)

                    self.ModbusClient.write_float_values(reg, value)

                    # Quick fix for Async generator
                    if gen_key == 'Async':
                        self.ModbusClient.write_float_values(139, 1)  

        for storage in self.EL_Net.Storages.keys():
            if storage in self.EL_Net.solution_dict:
                storage_key = self.EL_Net.solution_dict[storage]
                if 'P_ch' in storage_key and 'P_dch' in storage_key:
                    if self.RunOpt:
                        value = -storage_key['P_ch'][0] + storage_key['P_dch'][0]
                    else:
                        value = 0
                    Pmax = self.EL_Net.Storages[storage].Params['CrateMax']*self.EL_Net.Storages[storage].Params['Capacity']
                    value = max(min(value, 15), -15)
                    reg   = ModbusMap.get(storage, None)

                    self.ModbusClient.write_float_values(reg, value)

        for load in self.EL_Net.ElectricalLoads.keys():
            if load in self.EL_Net.solution_dict:
                load_key = self.EL_Net.solution_dict[load]
                if 'Power' in load_key:
                    if self.RunOpt:
                        value = load_key['Power'][0] / 3
                    else:
                        value = 0
                    regs   = ModbusMap.get(load, None)
                    for reg in regs:
                        self.ModbusClient.write_float_values(reg, value)


    def StatusReader(self):
        if self.Opts['Debug']:
            if self.t_step == 0:
                self.MeasuredValues = {'En0': 40}
            else:
                self.MeasuredValues = {'En0': self.EL_Net.solution_dict['ESS4']['En'][1]} # Example for debug mode, should be replaced with actual values
        else:
        # Verifica la connessione a Modbus, se non conmnesso si ricconette
            if not self.ModbusClient.connect:     
                self.SetUpModbusConnection()

            address = 2826  # Address of the first register to read
            value = self.ModbusClient.read_float_values(address) # Sistemare la funzione per leggere i registri in modo dinamico (non hardcoded) in base ai tag dei sistema
            self.MeasuredValues = {'En0': value[0]*self.EL_Net.Storages['ESS4'].Params['Capacity']/100} # Initial energy in the storage TODO: Fix with modubs (adjust also SetOCPParameters)

    def SaveSolution(self):
        solution_dict   = self.EL_Net.solution_dict  

        current_day_str    = self.current_day_str 
        current_time_str   = datetime.now().strftime('%H_%M')
        if self.Opts['Debug']:
            solution_dict_path = os.path.join(os.getcwd(), 'DebugResults', current_day_str, 'L_MPC', 'EL', 'SolutionData')
            solution_dict_file = os.path.join(solution_dict_path, f'Solution_t{self.t_step}_tHl{self.t_step_hl}_{current_time_str}.json')
        else:
            solution_dict_path = os.path.join(os.getcwd(), 'Results', current_day_str, 'L_MPC', 'EL', 'SolutionData') 
            solution_dict_file = os.path.join(solution_dict_path, f'Solution_t{self.t_step}_tHl{self.t_step_hl}_{current_time_str}.json')

        os.makedirs(solution_dict_path, exist_ok=True) 

        with open(solution_dict_file, 'w') as f:
            json.dump(solution_dict, f, indent=4)

                    
        MeasDict_path = os.path.join(os.getcwd(), 'OnFieldReadings', 'EL')
        MeasDict_file = os.path.join(MeasDict_path,f'MeasDict_t{self.t_step}_tHl{self.t_step_hl}.json')
        os.makedirs(MeasDict_path, exist_ok=True) 
        with open(MeasDict_file, 'w') as f:
            json.dump(self.MeasuredValues, f, indent=4)

        if self.Opts['Debug']:
            ref_dict = self.Reference
            ref_dict_path = solution_dict_path
            ref_dict_file = os.path.join(ref_dict_path, f'Reference_t{self.t_step}_tHl{self.t_step_hl}_{current_time_str}.json')
            with open(ref_dict_file, 'w') as f:
                json.dump(ref_dict, f, indent=4)

    def Solve(self):
        
        if self.Reference is None and self.Opts['HierarchicalControl']:
            raise ValueError("Reference values must be set before solving the OCP.")
        
        self.StatusReader() 

        SetOCPParameters(self.ocp, self.EL_Net, self.ConfigParams, self.ProfilesParams, self.MeasuredValues, self.Reference, self.t_step)
        SetSolverOptions(self.ocp)

        try:
            sol = self.ocp.solve()
            self.EL_Net.GetSolutionValues(self.ocp)
        except RuntimeError as e:
            self.EL_Net.GetSolutionValues(self.ocp, Solved = False)

        self.EL_Net.ConvertSolutionToDict()
        if not self.Opts['Debug']:
            self.SetPointsWriter()  
        self.SaveSolution()

        self.t_step += 1

if __name__ == '__main__':


    ConfigDataPath  = os.path.join(os.getcwd(),'utils','L_MPC','ConfigData')
    ProfileDataPath = os.path.join(os.getcwd(),'utils','L_MPC','ProfileData')
        # Setting default options, that can be changed by the user from the script XXXX (to be defined)
    DefaultOpts = {
            # Selection of the configuration to be used
            'LocalNetInConfig' : ['TH','EL','EH'],
            
            'H_MPC': {'TimeDiscretization' : {'dt': 0.25, 'Tsim': 24}, # hours
                    },

            'EL': { 'FullConfig'            : ['Async','Li-Ion','ESS4'],
                    'SystemsInConfig'       : ['Async','ESS4'],
                    'TimeDiscretization'    : {'dt': 1/60, 'Tsim': 0.5}, # hours
                    'ScaleFactor_EE_AC'      : 1, # Scale factor for electrical demand
                    },

            'TH': { 'FullConfig'            : ['S100','S200','S300','S400','S500','S700_HL1','S700_HL2','S700_HL3','S700_HL4','S700_bypass'],
                    'SystemsInConfig'       : ['S100','S500','S700_HL1','S700_HL2','S700_HL3','S700_HL4'],#,'S700_bypass'],
                    'TESSinConfig'          : ['D201','D202'],
                    'TimeDiscretization'    : {'dt': 0.25, 'Tsim': 1}, # hours
                    'ScaleFactor_H_HT'      : 1, # Scale factor for thermal demand
                    },

            # Parameters for the EMS configuration
            'json_FileName_Config'  : 'DER_TH_EL_EH.json',
            'json_FileName_Pipes'   : 'PipesList.json',
            'ConfigDataPath'        : ConfigDataPath, 

            # Demand, RES and prices profile for the EMS 
            'json_FileName_Profile' : 'DERTF_Profile_Components.json',
            'ProfileDataPath'       : ProfileDataPath,

            # Settings 
            'PlotNetwork'           : False,    
            'HierarchicalControl'   : False,     # If True, the hierarchical control is used
            'Debug'                : False,     # If True, the debug mode is used (no Modbus connection, no writing of setpoints)   

            'ScaleFactor_H_HT'      : 4.1,      # Scale factor for thermal demand
            'ScaleFactor_EE_AC'     : 3,        # Scale factor for electrical demand
            }

    el_mpc = EL_MPC(DefaultOpts)

    # el_mpc.SetPointsWriter()
    stop = 1

    el_mpc.Solve()
