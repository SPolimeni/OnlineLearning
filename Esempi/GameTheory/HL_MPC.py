import casadi as ca
import numpy as np
import pickle
import time
import os

from FoRB import InertialFoRB
from utils import proximal
from utils import coupling
from HL_MPC import hl_th, hl_el, hl_eh
import json


class HL_MPC:
    def __init__(self, N, tau, foRB_opts=None):
        """ 
        HL-MPC object is used to handle the MPC iterations of the HL MPC.

        Fields:
        N: Samples in the prediction horizon,
        tau: Sampling time of the MPC,
        dt: Current MPC Iteration starts from 0,
        foRB_opts: options to tune the foRB algorithm,
        P_l_th: Cumulative Thermal load with a window of N,
        P_l_el: Cumulative Electrical load with a window of N,
        P_l_eh_th: Cumulative Energy Hub Thermal load with a window of N,
        P_l_eh_el: Cumulative Energy Hub Electrical load with a window of N,
        C_grid: Sell price imposed by the power grid,
        C_gas: Production cost of consuming gas,
        C_el: Electricity production cost.

        """
        # MPC
        self.N = N
        self.tau = tau
        self.dt = 0
        # Forb
        if foRB_opts is None:
            self.foRB_opts = {
                'MAX_ITER': 1500,
                'lambda_0': 35000,
                'alpha': 0.1,
                'theta': 0.1,
                'beta': 1,
                'tol': 1e-1,
                'verbose': True
            } 
        else:
            self.foRB_opts = foRB_opts
        # High level thermal network
        th_agent_generator = hl_th.HL_TH(N, tau)
        # High level electrical network
        el_agent_generator = hl_el.HL_EL(N, tau)
        # High level energy hub
        eh_agent_generator = hl_eh.HL_EH(N, tau)
        # Proximal problems
        prox_dct = {}
        prox_dct['th'] = proximal.Proximal(self.foRB_opts['alpha'], self.foRB_opts['theta'], th_agent_generator)
        prox_dct['el'] = proximal.Proximal(self.foRB_opts['alpha'], self.foRB_opts['theta'], el_agent_generator)
        prox_dct['eh'] = proximal.Proximal(self.foRB_opts['alpha'], self.foRB_opts['theta'], eh_agent_generator)
        # Coupling matrix
        A_dct = coupling.get_coupling_matrix(prox_dct['th'].index, prox_dct['el'].index, prox_dct['eh'].index, N)
        # Known terms
        b_dct = coupling.get_known_terms(prox_dct['el'].index, prox_dct['eh'].index, prox_dct['th'].index, N)
        
        # Initial iterations values
        lambda_0 = self.foRB_opts['lambda_0']*np.ones(A_dct['th'].shape[0])
        # States
        x_dct = {
            'th': np.ones(prox_dct['th'].index['C_th_eh']+N),
            'el': np.ones(prox_dct['el'].index['C_el_eh']+N),
            'eh': np.ones(prox_dct['eh'].index['P_el_eh']+N)
        }
        # Inertial FoRB
        legend = ['th', 'el', 'eh']
        self.foRB = InertialFoRB(legend, A_dct, x_dct, lambda_0, prox_dct, b_dct, beta=self.foRB_opts['beta'])

        # Disturbances
        self.disturbances = {
            'P_l_th': [],
            'P_l_el': [],
            'P_l_eh_th': [],
            'P_l_eh_el': [],
        }
        # Knwon terms
        self.known_terms = {
            'C_gas': [],
            'C_el': [],
            'C_grid': []
        }
        # Initial state
        self.E_0_dct = {
            'th': 0,
            'el': 0,
            'eh': 0
        }

    def set_disturbances(self, P_l_th: np.ndarray, P_l_el: np.ndarray, P_l_eh_th: np.ndarray, P_l_eh_el: np.ndarray):
        """
        Set the disturbances.
        """
        self.disturbances['P_l_th'] = P_l_th
        self.disturbances['P_l_el'] = P_l_el
        self.disturbances['P_l_eh_th'] = P_l_eh_th
        self.disturbances['P_l_eh_el'] = P_l_eh_el

    def set_known_terms(self, C_gas: np.ndarray, C_el: np.ndarray, C_grid: np.ndarray):
        """
        Set the known terms.
        """
        self.known_terms['C_gas'] = C_gas
        self.known_terms['C_el'] = C_el
        self.known_terms['C_grid'] = C_grid

    def set_initial_state(self, E_0_th: float, E_0_el: float, E_0_eh_el: float):
        """
        Set the initial state.
        """
        self.E_0_dct['th'] = E_0_th
        self.E_0_dct['el'] = E_0_el
        self.E_0_dct['eh'] = E_0_eh_el

        self.save_initial_state()


    def save_initial_state(self):
        """
        Save the initial state to a file.
        """
        # Create Log directory if it doesn't exist
        os.makedirs('Log', exist_ok=True)
        # Create results dictionary to pickle
        initial_state = {
            'E_0_dct': self.E_0_dct
        }

        # Define base filename in Log folder
        base_filename = os.path.join('Log', 'hl_mpc_initial_state_0.json')
        filename = base_filename
        counter = 1
        # Check if file exists and increment filename if needed
        while os.path.exists(filename):
            filename = os.path.join('Log', f"hl_mpc_initial_state_{counter}.json")
            counter += 1
        with open(filename, 'w') as f_json:
            json.dump(initial_state, f_json, indent=4)
        print(f"Initial state saved to {filename}")

    def set_iteration(self, dt: int):
        """
        Set current MPC iteration number.
        """
        self.dt = dt
    
    def run(self):
        """
        Run the MPC.
        """
        # Exctract correct window of disturbances
        P_l_th_i = self.disturbances['P_l_th'][self.dt:self.dt+self.N]
        P_l_el_i = self.disturbances['P_l_el'][self.dt:self.dt+self.N]
        P_l_eh_th_i = self.disturbances['P_l_eh_th'][self.dt:self.dt+self.N]
        P_l_eh_el_i = self.disturbances['P_l_eh_el'][self.dt:self.dt+self.N]
        # Extract correct window of known terms
        C_grid_i = self.known_terms['C_grid'][self.dt:self.dt+self.N]
        C_gas_i = self.known_terms['C_gas']
        C_el_i = self.known_terms['C_el']
        # Set initial state from measurements
        E_0_th = self.E_0_dct['th']
        E_0_el = self.E_0_dct['el']
        E_0_eh_el = self.E_0_dct['eh']

        # Define disturbances and initial conditions
        self.foRB.prox_dct['th'].hl_player_generator.set_tunable_parameters(self.foRB.prox_dct['th'].opti, self.foRB.prox_dct['th'].p_dct, P_l_th_i, E_0_th, C_gas_i, C_grid_i)

        self.foRB.prox_dct['el'].hl_player_generator.set_tunable_parameters(self.foRB.prox_dct['el'].opti, self.foRB.prox_dct['el'].p_dct, P_l_el_i, E_0_el, C_el_i, C_grid_i)

        self.foRB.prox_dct['eh'].hl_player_generator.set_tunable_parameters(self.foRB.prox_dct['eh'].opti, self.foRB.prox_dct['eh'].p_dct, P_l_eh_el_i, P_l_eh_th_i, E_0_eh_el)

        # Run the FoRB
        start_time = time.time()
        self.foRB.run(max_iter=self.foRB_opts['MAX_ITER'], tol=self.foRB_opts['tol'], verbose=self.foRB_opts['verbose'])
        execution_time = time.time() - start_time

        # Extract the next step predictions of the final energy content
        x_th = self.foRB.prox_dct['th'].hl_player_generator.unpack_solution(self.foRB.x_history[-1]['th'], self.foRB.prox_dct['th'].index)
        x_el = self.foRB.prox_dct['el'].hl_player_generator.unpack_solution(self.foRB.x_history[-1]['el'], self.foRB.prox_dct['el'].index)
        x_eh = self.foRB.prox_dct['eh'].hl_player_generator.unpack_solution(self.foRB.x_history[-1]['eh'], self.foRB.prox_dct['eh'].index)

        # Save the results on file:
        
        # Create Log directory if it doesn't exist
        os.makedirs('Log', exist_ok=True)
        # Create results dictionary to pickle
        results = {
            'x_history': self.foRB.x_history,
            'p_history': self.foRB.p_history,
            'lambda_history': self.foRB.lambda_history,
            'N': self.N,
            'tau': self.tau,
            'th_agent_generator': self.foRB.prox_dct['th'].hl_player_generator,
            'el_agent_generator': self.foRB.prox_dct['el'].hl_player_generator,
            'eh_agent_generator': self.foRB.prox_dct['eh'].hl_player_generator,
            'th_index': self.foRB.prox_dct['th'].index,
            'el_index': self.foRB.prox_dct['el'].index,
            'eh_index': self.foRB.prox_dct['eh'].index,
            'execution_time': execution_time
        }

        # Define base filename in Log folder
        base_filename = os.path.join('Log', 'hl_mpc_results_0.pkl')
        filename = base_filename
        counter = 1
        # Check if file exists and increment filename if needed
        while os.path.exists(filename):
            filename = os.path.join('Log', f"hl_mpc_results_{counter}.pkl")
            counter += 1

        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")