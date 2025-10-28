import os
import csv
import numpy as np
from datetime import datetime

def SaveData(y, u, s, Pb_pred, final_cost, computational_time, t_step, Model, InitialDate):
    # Create folder name with current date
    folder_name = f"Solutions/{Model}/{datetime.now().strftime('%d%m')}"
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    
    def save_with_timestamp(data, filename,k):
        if 'final_cost' in filename:
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

    print(np.mean(computational_time))
