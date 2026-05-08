import numpy as np
import csv
import argparse
import time
import traceback
import copy
from datetime import datetime, timedelta
from apscheduler.schedulers.background  import BackgroundScheduler

from class_MPC import MPC
from utils.DERTF_StatesDefinition       import main_status_readings
from utils.OPCUA.DERTF_SetPointsWriter  import OPCUA_SetPoint
from utils.DataManagement               import SaveAggregatedData
import os
import platform

if platform.system() == "Windows":
    from pyfmi import load_fmu


def AuxiliaryParameters(Params):
    controller.theta_history        = np.zeros([controller.num_steps, Params["n_x"], Params["n_out"]])  #MODIFIED
    controller.sigma                = np.zeros([Params["n_out"], controller.num_steps])  #MODIFIED
    controller.exploration          = np.zeros((1, controller.num_steps))
    controller.u_out                = np.zeros([Params["n_inp"],controller.num_steps])
    controller.y_out                = np.zeros([Params["n_out"],controller.num_steps])
    controller.y_rnn_out            = np.zeros([controller.num_steps,Params["n_out"]])
    controller.y_pred_out           = np.zeros([Params["n_out"],controller.num_steps])
    controller.Pb_pred_out          = np.zeros([2,controller.num_steps])
    controller.s_out                = np.zeros([Params["n_out"],controller.num_steps]) if controller.Param["Model"] == 'BNNExp' else np.zeros([Params["n_slack"],controller.num_steps])
    controller.final_cost           = np.zeros([controller.num_steps])
    controller.computation_time     = np.zeros([controller.num_steps])
    controller.time_points          = np.arange(0, controller.total_time + 1, controller.time_step)[:controller.num_steps]

def log_exception(msg):
    """Log exception message to exceptions log file."""
    LogsFolder = os.path.join(os.getcwd(), 'LogsFolder')
    os.makedirs(LogsFolder, exist_ok=True)
    log_file = os.path.join(LogsFolder, 'Exceptions.log')
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()}: {msg}\n")

def ReadStatus():
    if controller.Opts['Debug']:
        TdeliveryMeas = [71.06958, 70.96986, 71.104965, 71.12856]
        TreturnMeas   = [60.12344, 60.224037, 60.187595, 60.183292]
        FlowMeas      = [2.7702346, 2.7731984, 2.7713318, 2.7692516]
        
        if controller.t_step < controller.Param["T_C0"]:
            idx = min(controller.t_step, len(TdeliveryMeas) - 1)
            y_dict = {
                'T_delivery': TdeliveryMeas[idx],
                'T_return'  : TreturnMeas[idx],
                'flow'      : FlowMeas[idx],
            }
        else:
            y_dict = {
                'T_delivery': controller.y_out[0, controller.t_step],
                'T_return'  : controller.y_out[1, controller.t_step],
                'flow'      : controller.y_out[2, controller.t_step],
            }
        controller.y_dict = y_dict
    else:

        try:
            y_dict = main_status_readings()
            controller.y_dict = y_dict
        except Exception as e:
            print(f"Error reading status: {e}")
            print("Keeping previous state values.")
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                last_trace = tb[-1]
                msg = f"Error occurred in ReadStatus: {e}\nFile: {last_trace.filename}\nLine: {last_trace.lineno}"
            else:
                msg = f"Error occurred in ReadStatus: {e}"
            log_exception(msg)
            print(msg)

def SetPointsToDERTF(Solved=True):

    try: 

        k = controller.t_step-1
        GetValue = controller.opti.value if Solved else controller.opti.debug.value

        controller.SetPointsWriter.DataMap['Power_HL1']['Value'] = -1e-3 * GetValue(controller.u_out[0, k]) # W to kW
        controller.SetPointsWriter.DataMap['Power_HL2']['Value'] = -1e-3 * GetValue(controller.u_out[1, k]) # W to kW
        controller.SetPointsWriter.DataMap['Power_HL3']['Value'] = -1e-3 * GetValue(controller.u_out[2, k]) # W to kW
        controller.SetPointsWriter.DataMap['Power_HL4']['Value'] = -1e-3 * GetValue(controller.u_out[3, k]) # W to kW
        controller.SetPointsWriter.DataMap['T_out_GB']['Value']  = GetValue(controller.u_out[4, k])
        controller.SetPointsWriter.DataMap['T_out_EB']['Value']  = GetValue(controller.u_out[5, k])

        if abs(controller.SetPointsWriter.DataMap['Power_HL1']['Value']) < 10:
            controller.SetPointsWriter.DataMap['Power_HL1']['Value'] = 32
        if abs(controller.SetPointsWriter.DataMap['Power_HL2']['Value']) < 10:
            controller.SetPointsWriter.DataMap['Power_HL2']['Value'] = 32
        if abs(controller.SetPointsWriter.DataMap['Power_HL3']['Value']) < 10:
            controller.SetPointsWriter.DataMap['Power_HL3']['Value'] = 32
        if abs(controller.SetPointsWriter.DataMap['Power_HL4']['Value']) < 10:
            controller.SetPointsWriter.DataMap['Power_HL4']['Value'] = 32
        if controller.SetPointsWriter.DataMap['T_out_GB']['Value'] < 45:
            controller.SetPointsWriter.DataMap['T_out_GB']['Value'] = 72
        if controller.SetPointsWriter.DataMap['T_out_EB']['Value'] < 45:
            controller.SetPointsWriter.DataMap['T_out_EB']['Value'] = 72

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_trace = tb[-1]
            msg = f"Error occurred in SetPointsToDERTF: {e}\nFile: {last_trace.filename}\nLine: {last_trace.lineno}"
        else:
            msg = f"Error occurred in SetPointsToDERTF: {e}"
        log_exception(msg)
        print(msg)

    node_data = []
    for key in controller.SetPointsWriter.DataMap.keys():
        node_data.append(controller.SetPointsWriter.DataMap[key])


    if controller.SetPointsWriter.client is None:
        controller.SetPointsWriter.connect()
        
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            controller.SetPointsWriter.write_node(node_data)
            break
        except Exception as e:

            if attempt == max_attempts - 1:
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    last_trace = tb[-1]
                    msg = f"Error occurred while writing to OPC UA server: {e}\nFile: {last_trace.filename}\nLine: {last_trace.lineno}"
                else:
                    msg = f"Error occurred while writing to OPC UA server: {e}"
                log_exception(msg)
                print(msg)

            if attempt < max_attempts - 1:
                try:
                    controller.SetPointsWriter.disconnect()
                except:
                    pass
                controller.SetPointsWriter.connect()

            else:
                time.sleep(1)


def SetPointsToFMU():
    k    = controller.t_step - 1  # t_step already incremented at this point
    Ts   = controller.time_step
    T_C0 = controller.Param["T_C0"]

    Time = np.arange(0, Ts * k + 1, Ts)
    input_array = np.column_stack([
        Time,
        controller.u_out[0, :k+1],
        controller.u_out[1, :k+1],
        controller.u_out[2, :k+1],
        controller.u_out[3, :k+1],
        controller.u_out[4, :k+1],
        controller.u_out[5, :k+1],
    ])

    controller.fmu.set(['Ts_L4', 'T_return', 'mf_supply'], controller.y_out[:, 0])
    res = controller.fmu.simulate(
        start_time=Ts * T_C0,
        final_time=Ts * (k + 1),
        options=controller.fmu_opts,
        input=(['PowerLoad1', 'PowerLoad2', 'PowerLoad3', 'PowerLoad4', 'Tref_GB', 'Tref_EB'], input_array),
    )
    controller.y_out[0, k + 1] = res['Ts_L4'][-1]
    controller.y_out[1, k + 1] = res['T_return'][-1]
    controller.y_out[2, k + 1] = res['mf_supply'][-1]
    controller.fmu.reset()


def MPC_solve():

    k = copy.deepcopy(controller.t_step)

    ReadStatus()

    controller.y_out[0,k] = controller.y_dict['T_delivery']
    controller.y_out[1,k] = controller.y_dict['T_return']
    controller.y_out[2,k] = controller.y_dict['flow']

    try:

        if controller.Param['Model']=='BNN' or controller.Param['Model']=='BNNExp':
            if k > controller.Param["T_C0"]-1:
                controller.theta_current = controller.bayesian_correction(k,controller.y_out,controller.u_out) #MODIFIED 
                controller.theta_history[k,:,:] = controller.theta_current #MODIFIED
    
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_trace = tb[-1]
            msg = f"Error occurred in Bayesian correction: {e}\nFile: {last_trace.filename}\nLine: {last_trace.lineno}"
        else:
            msg = f"Error occurred in Bayesian correction: {e}"
        log_exception(msg)
        print(msg)
    
    try:
        flag=1
        slack_tol=1e-3 # TUNING PARAMETER
        ### Compute the control law every 15 minutes ###

        controller.u_control, controller.slack, controller.y_nnarx, controller.computation_time[k], controller.sigma_current, controller.slack_explo= controller.mpc_controller(
            k, controller.Param["T_C0"], controller.y_out, controller.u_out, controller.y_rnn_out,flag)
        if np.any(controller.slack_explo <= slack_tol): # TUNING, TO CHECK
            print("exploration ON")

        else:
            flag = 0
            print(" exploration OFF")
            controller.u_control, controller.slack, controller.y_nnarx, controller.computation_time[k],controller.sigma_current, controller.slack_explo= controller.mpc_controller(
            k, controller.Param["T_C0"], controller.y_out, controller.u_out, controller.y_rnn_out,flag)
        
        controller.exploration[0,k] = flag   #MODIFIED
        sigma_arr = np.asarray(controller.sigma_current)
        if sigma_arr.ndim == 0:
            sigma_arr = np.full((1, controller.Param["n_out"]), sigma_arr)
        elif sigma_arr.ndim == 1:
            sigma_arr = sigma_arr.reshape(1, -1)
        controller.sigma[:,k] = sigma_arr[0,:].reshape(controller.Param["n_out"]) #MODIFIED

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_trace = tb[-1]
            msg = f"Error occurred in MPC controller: {e}\nFile: {last_trace.filename}\nLine: {last_trace.lineno}"
        else:
            msg = f"Error occurred in MPC controller: {e}"
        log_exception(msg)
        print(msg)
        # mpc_controller sets self.y_nnarx to a symbolic CasADi MX during
        # SetOptimizationVariables(); if it throws before returning, the
        # attribute remains symbolic. Restore to the last numeric fallback.
        controller.y_nnarx = controller.y_prec

    controller.u_out[:, k] = controller.u_control
    # print( u[4:6, k])
    # print(controller.y_prec)
    if k < controller.Param["T_C0"]-1:
        controller.s_out[:, k] = controller.slack[:, 0]
    else:
        controller.s_out[:, k] = controller.slack[:, 0] if controller.Param["Model"] == 'BNNExp' else controller.slack

    controller.y_rnn_out[k + 1, :]  = controller.y_nnarx[:, 0]
    controller.y_pred_out[:, k + 1] = controller.y_prec[:, 0]
    controller.Pb_pred_out[0, k]    = controller.Pb_prec[0, 0]
    controller.Pb_pred_out[1, k]    = controller.Pb_prec[1, 0]
    
    gamma_pred_gas  = 0.034     #TUNING
    COP             = 0.8       #TUNING
    eff_EB          = 0.8       #TUNING # Metterla variabile?
    r               = int(np.floor(k / 3))
    c_el            = controller.ensure2D(controller.c_el)
    gamma_pred_el   = c_el[r, 0] / 1000
    T_s             = 300

    controller.final_cost[k] = gamma_pred_gas * (1/1000*T_s/3600 * controller.Pb_pred_out[0,k]/COP)+gamma_pred_el * (1/1000*T_s/3600 * controller.Pb_pred_out[1,k]/eff_EB) 

    controller.t_step += 1

    
    try:
        ### Save Data ###
        SaveAggregatedData(controller, k)

        if np.mod(k,controller.Param["T_C0"]-1)==0 or k==0:
            if k > 0 and not Opts['Debug']:
                SetPointsToDERTF()

        if k >= controller.Param["T_C0"] and Opts['Debug']:
            SetPointsToFMU()

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_trace = tb[-1]
            msg = f"Error occurred while saving data or sending setpoints: {e}\nFile: {last_trace.filename}\nLine: {last_trace.lineno}"
        else:
            msg = f"Error occurred while saving data or sending setpoints: {e}"
        log_exception(msg)
        print(msg)




if __name__ == "__main__":

    Opts = {'Debug':False}
    
    Param={
        "Nb"                :3,
        "n_out"             :3,
        "n_x"               :10,    #MODIFIED       
        "n_inp"             :6,
        "tot_hours"         :5,
        "Ts"                :300,
        "N"                 :12,
        "q_eb"              :3/3.6,  
        "Tr_max"            :66,
        "Tr_min"            :45, 
        "Ts_max"            :80,
        "Ts_max_eb"         :72,
        "Ts_min_supply"     :70,     #HARD CODED 
        "Ts_max_eb"         :72,     #TEST PAR ALB2 preso dai file test Simone
        "Ts_min_eb"         :65,     #TEST PAR
        "Ts_max_gb"         :80,     #TEST PAR
        "Ts_min_gb"         :65,     #TEST PAR
        "m_max"             :5,
        "m_min"             :2,
        "Pb_max"            :147e3,
        "Pb_min"            :60e3,
        "Pb_max_eb"         :38e3,  # MODIFIED
        "Pb_min_eb"         :22e3,  # MODIFIED
        "beta0_1"           :35,    # MODIFIED
        "beta0_2"           :15,    # MODIFIED
        "beta0_3"           :15,    # MODIFIED
        "epsilon0"          :0.4,   # MODIFIED
        "L_prev0"           :100,    #MODIFIED
        "sigma_epsilon0_2"  :0.001, #MODIFIED
        "gamma"             :0.2,   #MODIFIED
        "T_ref"             :70,  #MODIFIED
        "c_gas"             :0.034, #MODIFIED
        "COP"               :0.8,   #MODIFIED
        "eff_EB"            :0.8,   #MODIFIED
        "gamma_exp0"        :5,     #MODIFIED
        "gamma_slack0"      :1e2,   #MODIFIED
        "alpha_slack0"      :100,   #MODIFIED
        "nnarx_mat"         :os.path.join('NNmodels','net_1020.mat'),  #TUNING -DA VERIFICARE!!!
        "c_el"              :np.genfromtxt('20250501_20250501_MGP_PrezziZonali_Nord.csv', delimiter=';', usecols=(2), skip_header=2),
        "Potenza"           :np.genfromtxt('DHN_ground_truth_shifted.csv', delimiter=',', usecols=(3,4,5,6), skip_header=28),#potenza all'istante k
        "P_Shift"           :4000, 
        "P_Shift_mpc1"      :0,     #MODIFIED  
        "P_Shift_mpc2"      :0,  #MODIFIED          
        "P_Shift_mpc3"      :0,  #MODIFIED          
        "P_Shift_mpc4"      :0,  #MODIFIED                    
        "Model"             :'BNNExp',
        "T_C0"              :4, #istante in cui inizio ad applicare la legge di controllo
        "n_slack"           :1,

    }

    parser = argparse.ArgumentParser(description="Run script in debug or normal mode")
    parser.add_argument(
        "--Model",
        type=str,
        default='BNNExp',
        help="Set the model type: 'NNARX', 'GP', or 'BNN' or 'BNNExp'",
    )
    parser.add_argument(
        "--Debug",
        action="store_true",
        default=False,
        help="Run the script in debug mode with simulated data with FMU",
    )
    
    Opts['Debug'] = parser.parse_args().Debug
    args = parser.parse_args()
    Param["Model"] = args.Model

    # Opts['Debug']   = True
    # Param['Model']  = 'BNNExp'

    # Initizialization of MPC
    controller              = MPC(Param)
    controller.Opts         = Opts
    controller.Param        = Param
    controller.total_time   = Param["tot_hours"] * 3600
    controller.time_step    = Param["Ts"]
    controller.num_steps    = controller.total_time // controller.time_step + 1
    controller.t_step       = 0
    controller.y_dict       = None
    controller.aux          = AuxiliaryParameters(Param)
    controller.InitialDate  = datetime.now().strftime("%Y%m%d_%H%M%S")

    if Param['Model'] == 'BNNExp':
        controller.y_rnn_out[0, :]  = controller.y_prec[:, 0]
        controller.y_pred_out[:, 0] = controller.y_prec[:, 0]
    
    if not Opts['Debug']:
        SetPointsWriter         = OPCUA_SetPoint()
        SetPointsWriter.DataMap = {
            'Power_HL1' : {'NodeID': 'ns=4;s=CO.FC_HL1_FT701L.r_PotenzaDesiderata',    'Value': None},
            'Power_HL2' : {'NodeID': 'ns=4;s=CO.FC_HL2_FT711L.r_PotenzaDesiderata',    'Value': None},
            'Power_HL3' : {'NodeID': 'ns=4;s=CO.FC_HL3_FT721L.r_PotenzaDesiderata',    'Value': None},
            'Power_HL4' : {'NodeID': 'ns=4;s=CO.FC_HL4_FT731L.r_PotenzaDesiderata',    'Value': None},
            'T_out_GB'  : {'NodeID': 'ns=4;s=CO.GB.r_Ctrl_setpointGB',                 'Value': None},
            'T_out_EB'  : {'NodeID': 'ns=4;s=CO.EB.r_TemperOutput',                    'Value': None},
        }
        controller.SetPointsWriter = SetPointsWriter

    if Opts['Debug']:
        controller.fmu = load_fmu('FMUs/DHN_MC_V3_FMU_EB_60.fmu')
        fmu_opts = controller.fmu.simulate_options()
        fmu_opts['CVode_options']['rtol'] = 1e-4
        fmu_opts['CVode_options']['atol'] = 1e-4
        controller.fmu_opts = fmu_opts

        # Initialize FMU state history to match MPC_BNNExp_TestParameters
        init_steps = min(5, controller.num_steps, controller.N)
        controller.y_out[:, :init_steps] = controller.y_prec[:, :init_steps]
        controller.u_out[:, :init_steps] = controller.u_prec[:, :init_steps]
        controller.y_pred_out[:, 0] = controller.y_prec[:, 0]
        controller.y_rnn_out[0, :] = controller.y_prec[:, 0]

    if Opts['Debug']:
        for k in range(0, controller.num_steps - 1):
            MPC_solve()

    else: 
        scheduler = BackgroundScheduler()

        scheduler.add_job(ReadStatus, 'interval', seconds= 10, next_run_time=datetime.now() + timedelta(seconds=2))
        scheduler.add_job(MPC_solve, 'interval', seconds= controller.time_step, next_run_time=datetime.now() + timedelta(seconds=5))
        scheduler.start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
