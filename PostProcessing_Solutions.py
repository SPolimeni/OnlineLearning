import ast
import csv
import pickle
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def parse_config_value(value):
    if value is None:
        return None

    value = value.strip()
    if value == '':
        return None

    if value.startswith('[') and value.endswith(']'):
        text = re.sub(r'[\r\n]+', ' ', value)
        text = re.sub(r'\s+', ' ', text).strip()

        # Convert space-delimited numeric lists into valid Python list syntax.
        text = re.sub(r'(?<=[0-9\]])\s+(?=[\[\-0-9\.])', ', ', text)
        text = re.sub(r'\s*,\s*', ',', text)

        if '...' in text:
            return text

        try:
            return np.array(ast.literal_eval(text), dtype=float)
        except Exception:
            try:
                cleaned = text.replace('"', '').replace("'", '')
                return np.array(ast.literal_eval(cleaned), dtype=float)
            except Exception:
                numeric = np.fromstring(re.sub(r'[\[\]]', ' ', text), sep=' ')
                if numeric.size > 0:
                    return numeric
                return text

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def parse_config_file(path):
    config = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if key == '':
                continue
            value = ','.join(row[1:]).strip()
            config[key] = parse_config_value(value)
    return config


def find_latest_file(folder, prefix, suffix):
    files = sorted(folder.glob(f'{prefix}_*{suffix}'), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f'No files with prefix {prefix} and suffix {suffix} found in {folder}')
    return files[-1]


def load_csv_matrix(path, skipcols=2, transpose=True):
    data = np.genfromtxt(path, delimiter=',', dtype=float)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    values = data[:, skipcols:]
    return values.T if transpose else values


def load_csv_vector(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', dtype=float, usecols=2))


def load_solution_data(solution_folder):
    solution_folder = Path(solution_folder)
    config_path = find_latest_file(solution_folder, 'config', '.csv')
    config = parse_config_file(config_path)

    nnarx_mat = config.get('nnarx_mat', 'NNmodels/net_1020.mat')
    nnarx_mat = Path(str(nnarx_mat).replace(chr(92), '/').strip('"'))
    mat_path = Path(__file__).resolve().parent / nnarx_mat
    output_scaler = loadmat(mat_path)['output_scaler'][0, 0]
    out_scale = np.asarray(output_scaler['scale'][0], dtype=float)
    out_bias = np.asarray(output_scaler['bias'][0], dtype=float)

    y = load_csv_matrix(find_latest_file(solution_folder, 'y_measured', '.csv'))
    y_pred = load_csv_matrix(find_latest_file(solution_folder, 'y_pred', '.csv'))
    y_pred0 = load_csv_matrix(find_latest_file(solution_folder, 'y_rnn', '.csv'))
    u = load_csv_matrix(find_latest_file(solution_folder, 'controlLaw', '.csv'))
    Pb_pred = load_csv_matrix(find_latest_file(solution_folder, 'Pb', '.csv'))
    exploration = load_csv_matrix(find_latest_file(solution_folder, 'exploration', '.csv'), transpose=True)

    final_cost = load_csv_vector(find_latest_file(solution_folder, 'final_cost', '.csv'))
    computation_time = load_csv_vector(find_latest_file(solution_folder, 'computational_time', '.csv'))

    theta_sigma_file = find_latest_file(solution_folder, 'ThetaAndSigma', '.pkl')
    with open(theta_sigma_file, 'rb') as f:
        theta_sigma = pickle.load(f)
    theta_BNN = theta_sigma['theta']
    sigma = theta_sigma['sigma']

    c_el = np.asarray(config['c_el'], dtype=float)
    if c_el.ndim == 1:
        c_el = c_el.reshape(-1, 1)
    elif c_el.ndim == 2 and c_el.shape[0] == 1:
        c_el = c_el.T

    n_out = int(config['n_out'])
    n_inp = int(config['n_inp'])
    Ts = int(config.get('Ts', 300))
    num_steps = y.shape[1]
    ore_tot = float(config.get('tot_hours', num_steps * Ts / 3600.0))
    time_points = np.arange(num_steps) * Ts

    beta = np.array([
        float(config['beta0_1']) / out_scale[0],
        float(config['beta0_2']) / out_scale[1],
        float(config['beta0_3']) / out_scale[2],
    ], dtype=float)
    epsilon = float(config['epsilon0']) / beta[0] * beta

    return {
        'y': y,
        'y_pred': y_pred,
        'y_pred0': y_pred0,
        'u': u,
        'Pb_pred': Pb_pred,
        'exploration': exploration,
        'final_cost': final_cost,
        'computation_time': computation_time,
        'theta_BNN': theta_BNN,
        'sigma': sigma,
        'c_el': c_el,
        'out_bias': out_bias,
        'out_scale': out_scale,
        'beta': beta,
        'epsilon': epsilon,
        'n_out': n_out,
        'n_inp': n_inp,
        'num_steps': num_steps,
        'time_points': time_points,
        'ore_tot': ore_tot,
        'Ts_max': float(config['Ts_max']),
        'Ts_min': float(config['Ts_min']),
        'Tr_max': float(config['Tr_max']),
        'Tr_min': float(config['Tr_min']),
        'q_max': float(config['m_max']),
        'q_min': float(config['m_min']),
        'Ts_max_gb': float(config['Ts_max_gb']),
        'Ts_min_gb': float(config['Ts_min_gb']),
        'Ts_max_eb': float(config['Ts_max_eb']),
        'Ts_min_eb': float(config['Ts_min_eb']),
        'Pb_max_gb': float(config['Pb_max']),
        'Pb_min_gb': float(config['Pb_min']),
        'Pb_max_eb': float(config['Pb_max_eb']),
        'Pb_min_eb': float(config['Pb_min_eb']),
    }


def main(solution_folder=None):
    if solution_folder is None:
        solution_folder = Path(__file__).resolve().parent / 'Solutions' / 'BNNExp' / '2904'
    data = load_solution_data(solution_folder)

    y = data['y']
    y_pred = data['y_pred']
    y_pred0 = data['y_pred0']
    u = data['u']
    Pb_pred = data['Pb_pred']
    exploration = data['exploration']
    final_cost = data['final_cost']
    computation_time = data['computation_time']
    theta_BNN = data['theta_BNN']
    sigma = data['sigma']
    c_el = data['c_el']
    out_bias = data['out_bias']
    out_scale = data['out_scale']
    beta = data['beta']
    epsilon = data['epsilon']
    n_out = data['n_out']
    num_steps = data['num_steps']
    time_points = data['time_points']
    ore_tot = data['ore_tot']
    Ts_max = data['Ts_max']
    Ts_min = data['Ts_min']
    Tr_max = data['Tr_max']
    Tr_min = data['Tr_min']
    q_max = data['q_max']
    q_min = data['q_min']
    Ts_max_gb = data['Ts_max_gb']
    Ts_min_gb = data['Ts_min_gb']
    Ts_max_eb = data['Ts_max_eb']
    Ts_min_eb = data['Ts_min_eb']
    Pb_max_gb = data['Pb_max_gb']
    Pb_min_gb = data['Pb_min_gb']
    Pb_max_eb = data['Pb_max_eb']
    Pb_min_eb = data['Pb_min_eb']

    ##################### UNCERTAINTY #####################
    plt.plot(1 / 3600 * time_points[4:num_steps - 1], beta[0] * sigma[0, 4:num_steps - 1], linewidth=4)
    plt.plot(
        1 / 3600 * time_points[4:num_steps - 1],
        epsilon[0] * np.ones(sigma[0, 4:num_steps - 1].shape[0]),
        color='red',
        linestyle='--',
        linewidth=4,
    )
    expl = exploration.flatten()
    current_state = expl[0]
    start_idx = 0
    t = time_points / 3600
    for i in range(1, num_steps):
        if expl[i] != current_state:
            plt.axvspan(
                t[start_idx],
                t[i],
                facecolor='lightblue' if current_state == 0 else 'lightcoral',
                alpha=0.3,
            )
            start_idx = i
            current_state = expl[i]
    plt.axvspan(
        t[start_idx],
        t[num_steps - 1],
        facecolor='lightblue' if current_state == 0 else 'lightcoral',
        alpha=0.3,
    )
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'$w$', fontsize=30)
    plt.title(r'beta[0]*sigma[0] ', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1 / 3600 * time_points[4:num_steps - 1], beta[1] * sigma[1, 4:num_steps - 1], linewidth=4)
    plt.plot(
        1 / 3600 * time_points[4:num_steps - 1],
        epsilon[1] * np.ones(sigma[0, 4:num_steps - 1].shape[0]),
        color='red',
        linestyle='--',
        linewidth=4,
    )
    expl = exploration.flatten()
    current_state = expl[0]
    start_idx = 0
    t = time_points / 3600
    for i in range(1, num_steps):
        if expl[i] != current_state:
            plt.axvspan(
                t[start_idx],
                t[i],
                facecolor='lightblue' if current_state == 0 else 'lightcoral',
                alpha=0.3,
            )
            start_idx = i
            current_state = expl[i]
    plt.axvspan(
        t[start_idx],
        t[num_steps - 1],
        facecolor='lightblue' if current_state == 0 else 'lightcoral',
        alpha=0.3,
    )
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'$w$', fontsize=30)
    plt.title(r'beta[1]*sigma[1] ', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1 / 3600 * time_points[4:num_steps - 1], beta[2] * sigma[2, 4:num_steps - 1], linewidth=4)
    plt.plot(
        1 / 3600 * time_points[4:num_steps - 1],
        epsilon[2] * np.ones(sigma[0, 4:num_steps - 1].shape[0]),
        color='red',
        linestyle='--',
        linewidth=4,
    )
    expl = exploration.flatten()
    current_state = expl[0]
    start_idx = 0
    t = time_points / 3600
    for i in range(1, num_steps):
        if expl[i] != current_state:
            plt.axvspan(
                t[start_idx],
                t[i],
                facecolor='lightblue' if current_state == 0 else 'lightcoral',
                alpha=0.3,
            )
            start_idx = i
            current_state = expl[i]
    plt.axvspan(
        t[start_idx],
        t[num_steps - 1],
        facecolor='lightblue' if current_state == 0 else 'lightcoral',
        alpha=0.3,
    )
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'$w$', fontsize=30)
    plt.title(r'beta[2]*sigma[2] ', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    ##################### PRICE #####################
    price = np.zeros(num_steps)
    for k in range(num_steps):
        r = int(np.floor(k * len(c_el) / num_steps))
        r = min(r, len(c_el) - 1)
        price[k] = c_el[r, 0]
    plt.figure()
    plt.plot(1 / 3600 * time_points, price / 1000, linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel('Electricity price', fontsize=30)
    plt.title('Electricity price profile', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    ###################### OUTPUTS #####################
    y_BLL_max = ((y_pred - out_bias.reshape(n_out, 1)) / out_scale.reshape(n_out, 1)) + beta.reshape(n_out, 1) * sigma
    y_BLL_max = (y_BLL_max * out_scale.reshape(n_out, 1)) + out_bias.reshape(n_out, 1)
    y_BLL_min = ((y_pred - out_bias.reshape(n_out, 1)) / out_scale.reshape(n_out, 1)) - beta.reshape(n_out, 1) * sigma
    y_BLL_min = (y_BLL_min * out_scale.reshape(n_out, 1)) + out_bias.reshape(n_out, 1)

    plt.plot(1 / 3600 * time_points, y[0, :], label='Simulator', linewidth=4)
    plt.plot(1 / 3600 * time_points, y_pred[0, :], label='Output BNN', linewidth=4)
    plt.plot(1 / 3600 * time_points, y_pred0[0, :], label='Output Original NNARX', linewidth=2)
    plt.fill_between(1 / 3600 * time_points, y_BLL_min[0, :], y_BLL_max[0, :], color='orange', alpha=0.3)
    plt.plot(1 / 3600 * time_points, Ts_min * np.ones(((y_pred[0, :]).shape[0], 1)), color='k', linewidth=4)
    plt.plot(1 / 3600 * time_points, Ts_max * np.ones(((y_pred[0, :]).shape[0], 1)), color='k', linewidth=4)
    plt.legend()
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Temperature [°C]', fontsize=30)
    plt.title('Load supply temperature', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1 / 3600 * time_points, y[1, :], label='Simulator', linewidth=4)
    plt.plot(1 / 3600 * time_points, y_pred[1, :], label='Output BNN', linewidth=4)
    plt.plot(1 / 3600 * time_points, y_pred0[1, :], label='Output Original NNARX', linewidth=2)
    plt.fill_between(1 / 3600 * time_points, y_BLL_min[1, :], y_BLL_max[1, :], color='orange', alpha=0.3)
    plt.plot(1 / 3600 * time_points, Tr_min * np.ones(((y_pred[0, :]).shape[0], 1)), color='k', linewidth=4)
    plt.plot(1 / 3600 * time_points, Tr_max * np.ones(((y_pred[0, :]).shape[0], 1)), color='k', linewidth=4)
    plt.legend()
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Temperature [°C]', fontsize=30)
    plt.title('Return temperature', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1 / 3600 * time_points, y[2, :], label='Simulator', linewidth=4)
    plt.plot(1 / 3600 * time_points, y_pred[2, :], label='Output BNN', linewidth=4)
    plt.plot(1 / 3600 * time_points, y_pred0[2, :], label='Output Original NNARX', linewidth=2)
    plt.fill_between(1 / 3600 * time_points, y_BLL_min[2, :], y_BLL_max[2, :], color='orange', alpha=0.3)
    plt.plot(1 / 3600 * time_points, q_min * np.ones(((y_pred[0, :]).shape[0], 1)), color='k', linewidth=4)
    plt.plot(1 / 3600 * time_points, q_max * np.ones(((y_pred[0, :]).shape[0], 1)), color='k', linewidth=4)
    plt.legend()
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Mass flow rate [kg/s]', fontsize=30)
    plt.title('Flow rate', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    ###################### INPUTS #####################
    plt.plot(1 / 3600 * time_points[:num_steps - 1], u[4, :num_steps - 1], linewidth=4)
    plt.plot(1 / 3600 * time_points[:num_steps - 1], Ts_max_gb * np.ones(Pb_pred[1, :num_steps - 1].shape), 'k', linewidth=4)
    plt.plot(1 / 3600 * time_points[:num_steps - 1], Ts_min_gb * np.ones(Pb_pred[1, :num_steps - 1].shape), 'k', linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Temperature [°C]', fontsize=30)
    plt.title('Gas boiler input temperature', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1 / 3600 * time_points[0:num_steps - 1], u[5, :num_steps - 1], linewidth=4)
    plt.plot(1 / 3600 * time_points[0:num_steps - 1], Ts_max_eb * np.ones(Pb_pred[1, :num_steps - 1].shape), 'k', linewidth=4)
    plt.plot(1 / 3600 * time_points[0:num_steps - 1], Ts_min_eb * np.ones(Pb_pred[1, :num_steps - 1].shape), 'k', linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Temperature [°C]', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.tight_layout(pad=2.0)
    plt.title('Electric boiler input temperature', fontsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    ###################### POWER #####################
    plt.plot(1 / 3600 * time_points[:num_steps - 1], 1 / 1000 * Pb_pred[0, :num_steps - 1], linewidth=4)
    plt.plot(1 / 3600 * time_points[:num_steps - 1], 1 / 1000 * Pb_min_gb * np.ones(Pb_pred[1, :num_steps - 1].shape), 'k', linewidth=4)
    plt.plot(1 / 3600 * time_points[:num_steps - 1], 1 / 1000 * Pb_max_gb * np.ones(Pb_pred[1, :num_steps - 1].shape), 'k', linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Power [kW]', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.title('Gas boiler power', fontsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1 / 3600 * time_points[0:num_steps - 1], 1 / 1000 * Pb_pred[1, :num_steps - 1], linewidth=4)
    plt.plot(1 / 3600 * time_points[0:num_steps - 1], 1 / 1000 * Pb_min_eb * np.ones(Pb_pred[1, :num_steps - 1].shape), 'k', linewidth=4)
    plt.plot(1 / 3600 * time_points[0:num_steps - 1], 1 / 1000 * Pb_max_eb * np.ones(Pb_pred[1, :num_steps - 1].shape), 'k', linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Power [kW]', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.title('Electric boiler power', fontsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    for j in range(0, n_out):
        plt.figure(figsize=(10, 7.5))
        for i in range(0, 10):
            plt.plot(
                1 / 3600 * time_points[0:num_steps - 1],
                theta_BNN[0:num_steps - 1, i, j],
                label=f'Parameter {i}',
                linewidth=4,
            )
        plt.legend()
        plt.xlabel('Time (h)')
        plt.ylabel('Theta')
        plt.title(f'All parameters Output {j}')
        plt.tight_layout()
        plt.show()

    potenza_totale = u[0, :num_steps - 1] + u[1, :num_steps - 1] + u[2, :num_steps - 1] + u[3, :num_steps - 1]

    plt.plot(1 / 3600 * time_points[:num_steps - 1], u[0, :num_steps - 1], 'b-')
    plt.plot(1 / 3600 * time_points[:num_steps - 1], u[1, :num_steps - 1], 'r-')
    plt.plot(1 / 3600 * time_points[:num_steps - 1], u[2, :num_steps - 1], 'g-')
    plt.plot(1 / 3600 * time_points[:num_steps - 1], u[3, :num_steps - 1], 'm-')
    plt.xlabel('Time [h]', fontsize=20)
    plt.ylabel('Power [kW]', fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend()
    plt.tight_layout(pad=2.0)
    plt.show()

    plt.plot(1 / 3600 * time_points[:num_steps - 1], potenza_totale, 'k-')
    plt.xlabel('Time [h]', fontsize=8)
    plt.ylabel('Power [kW]')
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend()
    plt.tight_layout(pad=2.0)
    plt.show()

    costo = np.sum(final_cost[:num_steps - 1])
    print(costo)

    media_tempo_computazione = np.mean(computation_time[:num_steps - 1])
    print(media_tempo_computazione)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-process saved BNNExp solution data.')
    parser.add_argument('solution_folder', nargs='?', default=None, help='Path to the solution folder to load')
    args = parser.parse_args()
    main(args.solution_folder)
