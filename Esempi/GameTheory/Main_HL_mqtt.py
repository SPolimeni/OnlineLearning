import os
import sys

from HL_MPC import HL_MPC

from utils.ConfiguratorsFn      import DistributedFacilityConfigurator, ProfileConfigurator
from MQTT.ClientDefinitions import General_MQTTclient

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import time
import numpy as np
import json

ConfigDataPath  = os.path.join(os.getcwd(),'utils','L_MPC','ConfigData')
ProfileDataPath = os.path.join(os.getcwd(),'utils','L_MPC','ProfileData')

DefaultOpts = {
        # Selection of the configuration to be used
        'LocalNetInConfig' : ['TH','EL','EH'],

        'H_MPC': {'TimeDiscretization' : {'dt': 0.5, 'Tsim': 6}, # hours
                },

        'EL': { 'FullConfig'            : ['Async','Li-Ion','ESS4'],
                'SystemsInConfig'       : ['Async','ESS4'],
                'TimeDiscretization'    : {'dt': 1/60, 'Tsim': 0.5}, # hours
                'ScaleFactor_EE_AC'      : 1, # Scale factor for electrical demand
                },

        'TH': { 'FullConfig'            : ['S100','S200','S300','S400','S500','S700_HL1','S700_HL2','S700_HL3','S700_HL4','S700_bypass'],
                'SystemsInConfig'       : ['S100','S200','S500','S700_HL1','S700_HL2','S700_HL3','S700_HL4'],#,'S700_bypass'],
                'TESSinConfig'          : ['D201','D202'],
                'TimeDiscretization'    : {'dt': 0.25, 'Tsim': 2}, # hours
                'ScaleFactor_H_HT'      : 1, # Scale factor for thermal demand
                },

        'EH': { 'FullConfig'            : ['Li-Ion','S400','LoadPV','ElectronicLoad'],
                'SystemsInConfig'       : ['Li-Ion','S400','LoadPV','ElectronicLoad'],
                'TimeDiscretization'    : {'dt': 0.25, 'Tsim': 1}, # hours
                },   


        # Parameters for the EMS configuration
        'json_FileName_Config'  : 'DER_TH_EL_EH_withEB.json',
        'json_FileName_Pipes'   : 'PipesList.json',
        'ConfigDataPath'        : ConfigDataPath, 

        # Demand, RES and prices profile for the EMS 
        'json_FileName_Profile' : 'DERTF_Profile_Components_EHconEB_reduced.json',
        'ProfileDataPath'       : ProfileDataPath,

        # Settings 
        'PlotNetwork'           : False,    
        'HierarchicalControl'   : True,     # If True, the hierarchical control is used
        'Debug'                 : True,     # If True, the debug mode is used (no Modbus connection, no writing of setpoints)   
        }


# P_l_th = [35, 32, 34, 36, 37, 38, 45, 42, 37, 36, 34, 41, 35, 34, 32, 35, 37, 36, 35, 33, 31, 34, 35, 36]*2
# P_l_th = [p*3 for p in P_l_th] # triplicate the load power
# P_l_el = [33.045, 40.949, 40.779, 32.093, 23.084, 24.656, 25.331, 24.658, 27.580, 48.246, 49.615, 48.875, 56.390, 58.101, 47.988, 65.570, 57.762, 62.841, 58.457, 62.143, 56.387, 61.950, 48.322, 37.714]*2
# P_l_eh_th = [50, 52, 65, 62, 57, 56, 54, 61, 55, 54, 52, 51, 57, 55, 53, 51, 49, 52, 50, 48, 50, 48, 45, 47]*2
# P_l_eh_el = [6.553, 7.403, 6.385, 4.451, 4.482, 6.651, 5.724, 7.651, 8.966, 10.188, 10.335, 10.255, 9.063, 8.247, 7.160, 10.051, 9.211, 12.757, 10.286, 11.682, 11.063, 10.661, 8.196, 7.055]*2

# C_grid = [0.106, 0.0971, 0.0956, 0.0937, 0.0949, 0.0956, 0.092, 0.0909, 0.05, 0.013, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.011, 0.087, 0.1185, 0.1141, 0.1185, 0.0998, 0.85, 0.85]*2


def InitializeStaticParameters(hl_mpc, ProfilesParams):

    P_l_th      = ProfilesParams['Demands']['S700_HL1'] + ProfilesParams['Demands']['S700_HL2'] + ProfilesParams['Demands']['S700_HL3'] 
    P_l_el      = ProfilesParams['Demands']['MainLoad']
    P_l_eh_th   = ProfilesParams['Demands']['S700_HL4']
    P_l_eh_el   = ProfilesParams['Demands']['ElectronicLoad']+ ProfilesParams['Demands']['LoadPV']

    C_grid      = ProfilesParams['VarPrices']['ElectricGrid']
    C_gas   = 0.34
    C_el    = 0.16

    hl_mpc.set_disturbances(P_l_th, P_l_el, P_l_eh_th, P_l_eh_el)
    hl_mpc.set_known_terms(C_gas, C_el, C_grid)

def HL_measurement_update(hl_mpc, hl_mqtt_client):
    """
    Update the HL measurement values based on the TH and EL MPC solutions.
    """
    # hl_mqtt_client.subscribe_to_state_topics()

    el_state_data = hl_mqtt_client.get_state_from_topic("EL/MPC/Status")
    th_state_data = hl_mqtt_client.get_state_from_topic("TH/MPC/Status")
    eh_state_data = hl_mqtt_client.get_state_from_topic("EH/MPC/Status")

    if not el_state_data is None and not th_state_data is None and not eh_state_data is None:

        hl_mpc.MeasuredValues = {
            'EL': {'En0': el_state_data['En0']},
            'TH': {'En0': th_state_data['En0']},
            'EH': {'En0': eh_state_data['En0']}
        }

        # Save MeasuredValues to the Logs folder
        logs_dir = os.path.join(os.getcwd(), 'Logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"MeasuredValues_{timestamp}.json")
        with open(log_file, 'w') as f:
            json.dump(hl_mpc.MeasuredValues, f, indent=4)
    
    else:
        hl_mpc.MeasuredValues = None

    

def hl_solve():
    HL_measurement_update(hl_mpc, hl_mqtt_client)

    if not hl_mpc.MeasuredValues is None:
        
        E_0_th      = hl_mpc.MeasuredValues['TH']['En0']
        E_0_el      = hl_mpc.MeasuredValues['EL']['En0']
        E_0_eh_el   = hl_mpc.MeasuredValues['EH']['En0']

        hl_mpc.set_initial_state(E_0_th, E_0_el, E_0_eh_el)
        hl_mpc.run()

        hl_mpc.dt       = hl_mpc.dt + 1
        hl_mpc.t_step   = hl_mpc.dt
        solution_dict   = CreateHL_solution_dict(hl_mpc)

        SaveSolution(hl_mpc,solution_dict,DefaultOpts)

        hl_mqtt_client.publish_solution(solution_dict)
        time.sleep(0.5)  # Increased delay to ensure message delivery

        hl_mqtt_client.publish_time_step(hl_mpc.t_step)
        time.sleep(0.5)  # Increased delay to ensure message delivery

        hl_mqtt_client.publish_start_signal()
        time.sleep(0.5)  # Increased delay to ensure message delivery

def CreateHL_solution_dict(hl_mpc):
    x_th = hl_mpc.foRB.prox_dct['th'].hl_player_generator.unpack_solution(hl_mpc.foRB.x_history[-1]['th'], hl_mpc.foRB.prox_dct['th'].index)
    x_el = hl_mpc.foRB.prox_dct['el'].hl_player_generator.unpack_solution(hl_mpc.foRB.x_history[-1]['el'], hl_mpc.foRB.prox_dct['el'].index)
    x_eh = hl_mpc.foRB.prox_dct['eh'].hl_player_generator.unpack_solution(hl_mpc.foRB.x_history[-1]['eh'], hl_mpc.foRB.prox_dct['eh'].index)
    # Update the initial state
    E_0_th = x_th['E']
    E_0_el = x_el['E']
    E_0_eh_el = x_eh['E_el']

    solution_dict = {}
    solution_dict['TH'] = {}
    solution_dict['EL'] = {}
    solution_dict['EH'] = {}

    ## Thermal system
    solution_dict['TH']['H_HT'] = {}
    solution_dict['TH']['H_HT']['E_int'] = E_0_th.tolist()
    
    solution_dict['TH']['EL'] = {}
    solution_dict['TH']['EH'] = {}
    solution_dict['TH']['EL']['P_exch'] = x_th['P_th_el'].tolist()
    solution_dict['TH']['EH']['P_exch'] = x_th['P_th_eh'].tolist()

    ## Electrical system
    solution_dict['EL']['EE_AC'] = {}
    solution_dict['EL']['EE_AC']['E_int'] = E_0_el.tolist()

    solution_dict['EL']['TH']           = {}
    solution_dict['EL']['EH']           = {}
    solution_dict['EL']['ElectricGrid'] = {}
    solution_dict['EL']['TH']['P_exch']             = (-1*x_el['P_th_el']).tolist()
    solution_dict['EL']['EH']['P_exch']             = (x_el['P_el_eh']).tolist()
    solution_dict['EL']['ElectricGrid']['P_exch']   = (x_el['P_grid']).tolist()


    ## Energy Hub system
    solution_dict['EH']['EE_AC'] = {}
    solution_dict['EH']['EE_AC']['E_int'] = E_0_eh_el.tolist()

    solution_dict['EH']['TH'] = {}
    solution_dict['EH']['EL'] = {}
    solution_dict['EH']['TH']['P_exch'] = (-1*x_eh['P_th_eh']).tolist()
    solution_dict['EH']['EL']['P_exch'] = (-1*x_eh['P_el_eh']).tolist()

    return solution_dict
    
def SaveSolution(hl_mpc,solution_dict,Opts):
    current_day_str = datetime.now().strftime("%Y%m%d")
    current_time_str = datetime.now().strftime("%H%M%S")

    solution_dict_path = os.path.join(os.getcwd(),'Results','GT_Solutions',current_day_str)
    solution_dict_file = os.path.join(solution_dict_path,f'Solution_t{max([0,hl_mpc.t_step-1])}_{current_time_str}.json')
    
    os.makedirs(solution_dict_path,exist_ok=True)

    with open(solution_dict_file, 'w') as f:
        json.dump(solution_dict, f, indent=4)

    

if __name__ == '__main__':

    N = 12
    tau = 3600
    MAX_ITER = 6000

    ConfigParams       = DistributedFacilityConfigurator(DefaultOpts)
    ProfilesParams     = ProfileConfigurator(DefaultOpts,ConfigParams)

    foRB_opts = {
            'MAX_ITER': 1500,
            'lambda_0': 35000,
            'alpha': 0.1,
            'theta': 0.1,
            'beta': 1,
            'tol': 1e-1,
            'verbose': True
        } 

    # Create the HL-MPC object
    hl_mpc                  = HL_MPC.HL_MPC(N, tau, foRB_opts)
    hl_mpc.dt               = 0
    hl_mpc.ProfileParams    = ProfilesParams

    InitializeStaticParameters(hl_mpc, ProfilesParams)
    
    hl_mqtt_client = General_MQTTclient('HL', Opts = DefaultOpts)
    
    dt_hl = DefaultOpts['H_MPC']['TimeDiscretization']['dt'] * 3600
    gr_sol_time     = 30
    gr_state_time   = 5

    # hl_mpc.MeasuredValues = {
    #     'EL': {'En0': 30},
    #     'TH': {'En0': 830},
    #     'EH': {'En0': 15}
    #     }

    # hl_solve()

    hl_mqtt_client.connect()

    hl_mqtt_client.clear_retained_messages()
    time.sleep(0.2)

    hl_mqtt_client.subscribe_to_state_topics()
    hl_mqtt_client.start_loop()

    hl_mpc.MeasuredValues = None
    hl_mqtt_client.publish_stop_signal()
    time.sleep(0.2)

    while hl_mpc.MeasuredValues is None:
        HL_measurement_update(hl_mpc, hl_mqtt_client)
        time.sleep(10)
        print("Waiting for HL measurement values to be updated...")

    hl_solve()

    # TODO: Adjust the threadpool in ubuntu to allow more threads
    scheduler = BackgroundScheduler()

    # Schedule HL_MPC Solve (references updated after solve)
    scheduler.add_job(HL_measurement_update, 'interval', seconds = 60, args=[hl_mpc, hl_mqtt_client], id = 'hl_state_update', next_run_time = datetime.now() + timedelta(seconds = 180),misfire_grace_time = gr_state_time, coalesce = True)
    scheduler.add_job(hl_solve, 'interval', seconds = dt_hl, id = 'hl_solve', next_run_time = datetime.now() + timedelta(seconds = dt_hl),misfire_grace_time = gr_sol_time, coalesce = True)

    scheduler.start()
    
    try:
        while True:

            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        hl_mqtt_client.publish_stop_signal()
        time.sleep(0.2)
        hl_mqtt_client.disconnect()
        print("Scheduler stopped.")

    try:
        while True:
            
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("Scheduler stopped.")
    


