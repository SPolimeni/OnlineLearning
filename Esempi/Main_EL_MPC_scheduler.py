import os
import sys
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(main_dir)

import casadi as ca

from src.L_MPC.EL.main_EL_MPC           import EL_MPC
from apscheduler.schedulers.background  import BackgroundScheduler
from datetime import datetime, timedelta
import time
import numpy as np



ConfigDataPath  = os.path.join(main_dir, 'utils', 'L_MPC', 'ConfigData')
ProfileDataPath = os.path.join(main_dir, 'utils', 'L_MPC', 'ProfileData')


DefaultOpts = {
    # Selection of the configuration to be used
    'LocalNetInConfig' : ['EL'],

    'H_MPC': {'TimeDiscretization' : {'dt': 0.25, 'Tsim': 6}, # hours
            },

    'EL': { 'FullConfig'            : ['Async','Li-Ion','ESS4'],
            'SystemsInConfig'       : ['Async','ESS4'],
            'TimeDiscretization'    : {'dt': 1/60, 'Tsim': 6}, # hours
            'ScaleFactor_EE_AC'      : 1, # Scale factor for electrical demand
            },

    'TH': { 'FullConfig'            : ['S100','S200','S300','S400','S500','S700_HL1','S700_HL2','S700_HL3','S700_HL4','S700_bypass'],
            'SystemsInConfig'       : ['S100','S200','S500','S700_HL1','S700_HL2','S700_HL3','S700_HL4'],#,'S700_bypass'],
            'TESSinConfig'          : ['D201','D202'],
            'TimeDiscretization'    : {'dt': 0.25, 'Tsim': 2}, # hours
            'ScaleFactor_H_HT'      : 0.9, # Scale factor for thermal demand
            },

    # Parameters for the EMS configuration
    'json_FileName_Config'  : 'DER_TH_EL_EH_reducedGB.json',
    'json_FileName_Pipes'   : 'PipesList.json',
    'ConfigDataPath'        : ConfigDataPath, 

    # Demand, RES and prices profile for the EMS 
    'json_FileName_Profile' : 'DERTF_Profile_Components_Prezzo1Maggio.json',
    'ProfileDataPath'       : ProfileDataPath,

    # Settings 
    'PlotNetwork'           : False,    
    'HierarchicalControl'   : False,    # If True, the hierarchical control is used
    'Debug'                 : False,    # If True, the debug mode is enabled

    'ScaleFactor_H_HT'      : 1,        # Scale factor for thermal demand
    'ScaleFactor_EE_AC'     : 1,        # Scale factor for electrical demand

    
    }

def el_status():
    el_mpc.StatusReader()

def el_solve():
    el_mpc.Solve()


if __name__ == '__main__':

    Opts = DefaultOpts

    el_mpc = EL_MPC(Opts)
    el_mpc.StatusReader()

    Reference = {
        'P_exch' : {'EL': np.tile([-50],8)}, # Reference power exchange
        'E_int'  : np.tile([1624.2055548008004],8) # Reference internal energy
    }

    el_mpc.Reference = Reference

    dt_el = DefaultOpts['EL']['TimeDiscretization']['dt'] * 3600
    n_runs_max = int(DefaultOpts['H_MPC']['TimeDiscretization']['Tsim'] / DefaultOpts['EL']['TimeDiscretization']['dt'])

    dt_el_status    = 60
    gr_sol_time     = 5
    gr_state_time   = 5


    el_solve()

    scheduler = BackgroundScheduler()
    # Schedule StatusReader for TH and EL
    scheduler.add_job(el_status, 'interval', seconds = dt_el_status, id = 'el_status', misfire_grace_time = gr_state_time, coalesce = True)
    scheduler.add_job(el_solve, 'interval', seconds = dt_el, id = 'el_solve', next_run_time = datetime.now() + timedelta(seconds = dt_el), 
                        misfire_grace_time = gr_sol_time, coalesce = True)


    scheduler.start()

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("Scheduler stopped.")