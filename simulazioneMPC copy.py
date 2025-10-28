
import numpy as np
import matplotlib.pyplot as plt
import csv

from class_MPC import MPC
from utils.DERTF_StatesDefinition import main_status_readings



def SaveData(y,u,s,Pb_pred,final_cost, computational_time):
    ### Save Output ###
    dati=y.transpose()
    with open('y_measured.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for riga in dati:
            writer.writerow(riga)
    ### Save Control Law ###
    dati=u.transpose()
    with open('controlLaw.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for riga in dati:
            writer.writerow(riga)

    ### Save Power boilers ###
    dati=Pb_pred.transpose()
    with open('Pb.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for riga in dati:
            writer.writerow(riga)
    
    ### Save Slack
    dati=s.transpose()
    with open('slack.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for riga in dati:
            writer.writerow(riga)
    ### Save Slack
    dati=final_cost.transpose()
    with open('final_cost.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for riga in dati:
            writer.writerow(riga)
    print(np.mean(computation_time))
    

if __name__ == "__main__":
    
    Param={
        "Nb":3,
        "n_out":3,
        "n_inp":6,
        "tot_hours":5,
        "Ts":300,
        "N":12,
        "q_eb":3/3.6,
        "Tr_max": 66,
        "Tr_min": 45,
        "Ts_max": 80,
        "Ts_max_eb": 75,
        "Ts_min":65,
        "m_max":10,
        "m_min":2,
        "Pb_max":147e3,
        "Pb_min":60e3,
        "Pb_max_eb":50e3,
        "Pb_min_eb":30e3,
        "nnarx_mat": 'NNARX_9-9_H3_bs20_Ts300_Ns300_20251020_154518\\net.mat',
        "c_el": np.genfromtxt('20250501_20250501_MGP_PrezziZonali_Nord.csv', delimiter=';', usecols=(2), skip_header=2),
        "Potenza":np.genfromtxt('DHN_ground_truth.csv', delimiter=',', usecols=(3,4,5,6), skip_header=28),#potenza all'istante k
        "Model":'NNARX',
        "T_C0":4, #istante in cui inizio ad applicare la legge di controllo
        "L_prev": np.eye(10),
        "Q_prev": np.zeros([10,3]),
        "n_slack":1,
        
    }
    controller = MPC(Param)
    total_time = Param["tot_hours"] * 3600
    time_step = Param["Ts"]
    num_steps = total_time // time_step + 1

    u=np.zeros([Param["n_inp"],num_steps])
    y=np.zeros([Param["n_out"],num_steps])
    y_rnn=np.zeros([num_steps,Param["n_out"]])
    y_pred=np.zeros([Param["n_out"],num_steps])
    Pb_pred=np.zeros([2,num_steps])
    s=np.zeros([Param["n_slack"],num_steps])
    final_cost = np.zeros([num_steps])
    computation_time = np.zeros([num_steps])
    time_points = np.arange(0, total_time + 1, time_step)[:num_steps]
   
    for k in range(0, num_steps - 1):
        print(k)

        y_dict = main_status_readings()
        y[0,k] = y_dict['T_delivery']
        y[1,k] = y_dict['T_return']
        y[2,k] = y_dict['flow']


        if Param['Model']=='BNN':
        
            if k>Param["T_C0"]-1:
            
                controller.bayesian_correction(k,y,u)
        ### Compute the control law every 15 minutes ###

        u_control, slack, y_nnarx, computation_time[k] = controller.mpc_controller(k, Param["T_C0"], y, u, y_rnn)

        u[:, k] = u_control
        # print( u[4:6, k])
        # print(controller.y_prec)
        if k < Param["T_C0"]:
            s[:, k ] = slack[:, 0]
        else:
            s[:, k ] = slack
        y_rnn[ k + 1,:] = y_nnarx[:, 0]
        y_pred[: , k + 1] = controller.y_prec[:, 0]
        Pb_pred[0, k ] = controller.Pb_prec[0, 0]
        Pb_pred[1, k ] = controller.Pb_prec[1, 0]
        
        ### Save true value every 5 minutes 
        # y[0, k + 1] =
        # y[1, k + 1] = 
        # y[2, k + 1] = 
        
        
        gamma_pred_gas = 0.034
        COP=0.8
        r = int(np.floor(k / 3))
        c_el = controller.ensure2D(controller.c_el)
        gamma_pred_el = c_el[r, 0] / 1000
        T_s = 300
        final_cost[k] = gamma_pred_gas * (1/1000*T_s/3600 * Pb_pred[0,k]/COP)+gamma_pred_el * (1/1000*T_s/3600 * Pb_pred[1,k]/1)

        print(final_cost[k])
        print(controller.J_prec)
        if np.mod(k,3)==0 and k==0:
            #send the control law
            u[:, k] = u_control

            # U: 
            # prime 4 le potenze in W e negative
            # T mandata Gas Boiler
            # T mandata Elctric Boiler


    ### Save Data ###
    SaveData(y,u,s,Pb_pred,final_cost)