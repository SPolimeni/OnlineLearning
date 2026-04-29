import os
import csv
import pickle
import numpy as np
from datetime import datetime


def SaveAggregatedData(controller, t_step):
    Model       = controller.Param["Model"]
    InitialDate = controller.InitialDate

    folder_name = f"Solutions/{Model}/{datetime.now().strftime('%d%m')}"
    os.makedirs(folder_name, exist_ok=True)

    timestamp = datetime.now().strftime('%H:%M')

    def _append_row(filename, values):
        filepath = os.path.join(folder_name, filename)
        with open(filepath, 'a', newline='') as f:
            csv.writer(f).writerow([timestamp, t_step] + values)

    def save_col(data, filename):
        """data shape: [n_var, num_steps] — prende la colonna t_step"""
        _append_row(filename, list(data[:, t_step].astype(np.float32)))

    def save_row(data, filename):
        """data shape: [num_steps, n_var] — prende la riga t_step (es. y_rnn_out)"""
        _append_row(filename, list(data[t_step, :].astype(np.float32)))

    def save_scalar(data, filename):
        """data shape: [num_steps] — prende il valore scalare a t_step"""
        _append_row(filename, [float(data[t_step])])

    # CSV: serie temporali (append incrementale, resiliente ai crash)
    save_col(controller.y_out,          f'y_measured_{InitialDate}.csv')
    save_col(controller.y_pred_out,     f'y_pred_{InitialDate}.csv')
    save_row(controller.y_rnn_out,      f'y_rnn_{InitialDate}.csv')
    save_col(controller.u_out,          f'controlLaw_{InitialDate}.csv')
    save_col(controller.Pb_pred_out,    f'Pb_{InitialDate}.csv')
    save_col(controller.s_out,          f'slack_{InitialDate}.csv')
    save_scalar(controller.final_cost,         f'final_cost_{InitialDate}.csv')
    save_scalar(controller.computation_time,   f'computational_time_{InitialDate}.csv')

    # CSV: parametri di configurazione (scritto solo al primo timestep)
    params_csv_path = os.path.join(folder_name, f'config_{InitialDate}.csv')
    if not os.path.exists(params_csv_path):
        with open(params_csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['key', 'value'])
            for key, val in controller.Param.items():
                w.writerow([key, val])

    # PKL: sovrascrive con lo stato corrente degli array fino a t_step
    pkl_path = os.path.join(folder_name, f'ThetaAndSigma_{InitialDate}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'theta': controller.theta_history, 'sigma': controller.sigma}, f)

    print(np.mean(controller.computation_time[:t_step + 1]))


def SaveData(y, u, s, Pb_pred, final_cost, computational_time, t_step, Model, InitialDate):
    # Create folder name with current date
    folder_name = f"Solutions/{Model}/{datetime.now().strftime('%d%m')}"
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    
    def save_with_timestamp(data, filename,k):
        if 'final_cost' in filename or 'computational_time' in filename:
            dati = data[k].reshape(1, -1)
        else:
            dati = data[:,k].reshape(1, -1)
        filepath = os.path.join(folder_name, filename)
        # Open in append mode so existing files get new lines instead of being overwritten
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            for riga in dati:
            # Add timestamp (hour:minute) to beginning of each row
                timestamp = datetime.now().strftime('%H:%M')
                if np.isscalar(riga):
                    row_with_time = [timestamp, t_step, float(riga)]
                else:
                    row_with_time = [timestamp, t_step] + list(riga.astype(np.float32))
                writer.writerow(row_with_time)
        
    ### Save all data with timestamps ###
    save_with_timestamp(y, f'y_measured_{InitialDate}.csv', t_step)
    save_with_timestamp(u, f'controlLaw_{InitialDate}.csv', t_step)
    save_with_timestamp(Pb_pred, f'Pb_{InitialDate}.csv', t_step)
    save_with_timestamp(s, f'slack_{InitialDate}.csv', t_step)
    save_with_timestamp(final_cost, f'final_cost_{InitialDate}.csv', t_step)
    save_with_timestamp(computational_time, f'computational_time_{InitialDate}.csv', t_step)

    print(np.mean(computational_time))
