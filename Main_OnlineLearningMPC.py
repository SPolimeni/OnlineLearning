import numpy as np
import csv
import argparse
import time
import traceback
from datetime import datetime, timedelta
from apscheduler.schedulers.background  import BackgroundScheduler

from class_MPC import MPC
from utils.DERTF_StatesDefinition       import main_status_readings
from utils.OPCUA.DERTF_SetPointsWriter  import OPCUA_SetPoint
from utils.DataManagement               import SaveData
import os


def AuxiliaryParameters(Params):
    controller.u_out                = np.zeros([Params["n_inp"],controller.num_steps])
    controller.y_out                = np.zeros([Params["n_out"],controller.num_steps])
    controller.y_rnn_out            = np.zeros([controller.num_steps,Params["n_out"]])
    controller.y_pred_out           = np.zeros([Params["n_out"],controller.num_steps])
    controller.Pb_pred_out          = np.zeros([2,controller.num_steps])
    controller.s_out                = np.zeros([Params["n_slack"],controller.num_steps])
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
        y_dict = {
            'T_delivery': controller.y_nnarx_out[controller.t_step,0],
            'T_return'  : controller.y_nnarx_out[controller.t_step,1],
            'flow'      : controller.y_nnarx_out[controller.t_step,2],
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

        k = controller.t_step
        GetValue = controller.opti.value if Solved else controller.opti.value

        controller.SetPointsWriter.DataMap['Power_HL1']['Value'] = -1e-3 * GetValue(controller.u_out[0, k]) # W to kW
        controller.SetPointsWriter.DataMap['Power_HL2']['Value'] = -1e-3 * GetValue(controller.u_out[1, k]) # W to kW
        controller.SetPointsWriter.DataMap['Power_HL3']['Value'] = -1e-3 * GetValue(controller.u_out[2, k]) # W to kW
        controller.SetPointsWriter.DataMap['Power_HL4']['Value'] = -1e-3 * GetValue(controller.u_out[3, k]) # W to kW
        controller.SetPointsWriter.DataMap['T_out_GB']['Value']  = GetValue(controller.u_out[4, k])+2
        controller.SetPointsWriter.DataMap['T_out_EB']['Value']  = GetValue(controller.u_out[5, k])+2
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

    try:
        controller.SetPointsWriter.write_node(node_data)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_trace = tb[-1]
            msg = f"Error occurred while writing to OPC UA server: {e}\nFile: {last_trace.filename}\nLine: {last_trace.lineno}"
        else:
            msg = f"Error occurred while writing to OPC UA server: {e}"
        log_exception(msg)
        print(msg)

def MPC_solve():

    k = controller.t_step
    u = controller.u_out

    if Opts['Debug']:
        controller.y_out[0,k] = 70.86
        controller.y_out[1,k] = 61.3
        controller.y_out[2,k] = 3.36

    else:
        ReadStatus()

        controller.y_out[0,k] = controller.y_dict['T_delivery']
        controller.y_out[1,k] = controller.y_dict['T_return']
        controller.y_out[2,k] = controller.y_dict['flow']

    try:

        if controller.Param['Model']=='BNN':
            if k > controller.Param["T_C0"]-1:
                controller.bayesian_correction(k,controller.y_out,controller.u_out)

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

        ### Compute the control law every 15 minutes ###
        controller.u_control, controller.slack, controller.y_nnarx, controller.computation_time[k] = controller.mpc_controller(
            k, controller.Param["T_C0"], controller.y_out, controller.u_out, controller.y_rnn_out)

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_trace = tb[-1]
            msg = f"Error occurred in MPC controller: {e}\nFile: {last_trace.filename}\nLine: {last_trace.lineno}"
        else:
            msg = f"Error occurred in MPC controller: {e}"
        log_exception(msg)
        print(msg)

    controller.u_out[:, k] = controller.u_control
    # print( u[4:6, k])
    # print(controller.y_prec)
    if k < controller.Param["T_C0"]:
        controller.s_out[:, k] = controller.slack[:, 0]
    else:
        controller.s_out[:, k] = controller.slack

    controller.y_rnn_out[k + 1, :]  = controller.y_nnarx[:, 0]
    controller.y_pred_out[:, k + 1] = controller.y_prec[:, 0]
    controller.Pb_pred_out[0, k]    = controller.Pb_prec[0, 0]
    controller.Pb_pred_out[1, k]    = controller.Pb_prec[1, 0]
    
    gamma_pred_gas  = 0.034
    COP             = 0.8
    r               = int(np.floor(k / 3))
    c_el            = controller.ensure2D(controller.c_el)
    gamma_pred_el   = c_el[r, 0] / 1000
    T_s             = 300
    controller.final_cost[k] = gamma_pred_gas * (1/1000*T_s/3600 * controller.Pb_pred_out[0,k]/COP)+gamma_pred_el * (1/1000*T_s/3600 * controller.Pb_pred_out[1,k]/1)

    controller.t_step += 1

    
    try:
        ### Save Data ###
        SaveData(controller.y_out, controller.u_out, controller.s_out, controller.Pb_pred_out, controller.final_cost, 
        controller.computation_time[k], k, controller.Param["Model"], controller.InitialDate)

        if np.mod(k,controller.Param["T_C0"]-1)==0 or k==0:
            #send the control law
            controller.u_out[:, k] = controller.u_control

            # U: 
            # prime 4 le potenze in W e negative
            # T mandata Gas Boiler
            # T mandata Elctric Boiler
            if k > 0 and not Opts['Debug']:
                SetPointsToDERTF()

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
        "Nb"        :3,
        "n_out"     :3,
        "n_inp"     :6,
        "tot_hours" :5,
        "Ts"        :300,
        "N"         :12,
        "q_eb"      :3/3.6,
        "Tr_max"    :66,
        "Tr_min"    :45,
        "Ts_max"    :80,
        "Ts_max_eb" :75,
        "Ts_min"    :65,
        "m_max"     :10,
        "m_min"     :2,
        "Pb_max"    :147e3,
        "Pb_min"    :60e3,
        "Pb_max_eb" :50e3,
        "Pb_min_eb" :30e3,
        "nnarx_mat" :'NNARX_9-9_H3_bs20_Ts300_Ns300_20251020_154518\\net.mat',
        "c_el"      :np.genfromtxt('20250501_20250501_MGP_PrezziZonali_Nord.csv', delimiter=';', usecols=(2), skip_header=2),
        "Potenza"   :np.genfromtxt('DHN_ground_truth.csv', delimiter=',', usecols=(3,4,5,6), skip_header=28),#potenza all'istante k
        "Model"     :'NNARX',
        "T_C0"      :4, #istante in cui inizio ad applicare la legge di controllo
        "L_prev"    :np.eye(10),
        "Q_prev"    :np.zeros([10,3]),
        "n_slack"   :1,

    }

    parser = argparse.ArgumentParser(description="Run script in debug or normal mode")
    parser.add_argument(
        "--Model",
        type=str,
        default='NNARX',
        help="Set the model type: 'NNARX', 'GP', or 'BNN'",
    )
    parser.add_argument(
        "--Debug",
        action="store_true",
        default=False,
        help="Run the script in debug mode with simulated data",
    )
    
    Opts['Debug'] = parser.parse_args().Debug
    args = parser.parse_args()
    Param["Model"] = args.Model

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

    SetPointsWriter         = OPCUA_SetPoint()
    SetPointsWriter.DataMap = {
        'Power_HL1' : {'NodeID': 'ns=4;s=CO.FC_HL1_FT701L.r_PotenzaDesiderata',    'Value': None},
        'Power_HL2' : {'NodeID': 'ns=4;s=CO.FC_HL2_FT711L.r_PotenzaDesiderata',    'Value': None},
        'Power_HL3' : {'NodeID': 'ns=4;s=CO.FC_HL3_FT721L.r_PotenzaDesiderata',    'Value': None},
        'Power_HL4' : {'NodeID': 'ns=4;s=CO.FC_HL4_FT731L.r_PotenzaDesiderata',    'Value': None},
        'T_out_GB'  : {'NodeID': 'ns=4;s=CO.GB.r_Ctrl_setpointGB',                 'Value': None},
        'T_out_EB'  : {'NodeID': 'ns=4;s=CO.EB.r_TemperOutput',                    'Value': None},
    }
    if not Opts['Debug']:
        controller.SetPointsWriter = SetPointsWriter
    
    if Opts['Debug']:
        for k in range(0, controller.num_steps - 1):
            MPC_solve()

    else: 
        scheduler = BackgroundScheduler()

        scheduler.add_job(ReadStatus, 'interval', seconds= 10, next_run_time=datetime.now() + timedelta(seconds=2))
        scheduler.add_job(MPC_solve, 'interval', seconds=controller.time_step, next_run_time=datetime.now() + timedelta(seconds=5))
        scheduler.start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
