"""Core MPC implementation for the thermal-control application.

This module holds the controller state, model-loading logic, optimization setup,
and the receding-horizon control cycle. The class is designed around a custom
combination of model predictions, FMU-based simulation, and neural-network-based
state estimation.

Functional overview:
- The controller is initialized from a configuration dictionary containing the
  plant constraints, economic parameters, and model metadata.
- At each control step it reads the current system state and reconstructs the
  last input/output history used by the predictive model.
- The NNARX/GP/BNNExp model is used to generate a prediction over the horizon and
  the optimizer solves a constrained finite-horizon problem.
- The first control action of the computed sequence is applied to the plant or the
  debug FMU, while the complete optimized trajectory is kept as warm start for the
  next step.
- In the Bayesian variants, uncertainty-aware corrections are incorporated through
  the last-layer update and uncertainty-dependent slack constraints.
"""

import torch

import casadi
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# from pyfmi import load_fmu
import time

from model_strategies import NNARXStrategy, GPStrategy, BNNExpStrategy

_STRATEGY_FACTORY = {
    "NNARX":    NNARXStrategy,  
    "GP":       GPStrategy,
    "BNNExp":   BNNExpStrategy,
}

class MPC:
    """Model-predictive controller for the thermal energy system.

    The controller loads the neural model, stores the previous optimal trajectory,
    and repeatedly solves a constrained optimization problem. The first control
    action from the optimized horizon is applied to the plant or simulation model.

    All behavior specific to a given Param["Model"] ('NNARX', 'GP', 'BNNExp') is
    delegated to a per-model strategy object (see model_strategies.py); this class
    holds only the common orchestration logic, which is the NNARX behavior itself.
    """

    def __init__(self, Param: dict):
        """Initialize controller parameters, model data, and optimization state.

        Parameters are passed as a single dictionary from the runtime entry point.
        The constructor loads model metadata and initializes the prior predictions,
        costs, and control buffers used in the receding-horizon loop.
        """
        print("MPC class initialization")
        self._assign_parameters(Param)

    def _assign_parameters(self, params: dict):
        """Copy configuration values and initialize the derived controller state."""
        for name, value in params.items():
            if name == "c_el":
                value = self.ensure2D(value)
            setattr(self, name, value)

        self.total_time = self.tot_hours * 3600
        self.time_step  = 300
        self.num_steps  = self.total_time // self.time_step + 1
        self.n_uopt     = int(self.N / self.Nb)

        # ``data`` is derived from the model path; it is not a configuration key.
        self.data       = loadmat(self.nnarx_mat)
        self.Potenze    = self.Potenza.T + self.P_Shift

        if self.Model not in _STRATEGY_FACTORY:
            raise ValueError(f"Unsupported Model '{self.Model}'. Supported models: {sorted(_STRATEGY_FACTORY)}")
        
        ### Initial value MPC
        # These values are the warm-start trajectory used at startup before the first
        # optimization solve. They avoid a full cold start and provide a stable initial
        # guess for the solver in the first receding-horizon iteration.
        self.J_prec=0
        y_pred          = np.zeros([self.n_out,self.N])
        y_pred[0,:]     = 70.2690912858482
        y_pred[1,:]     = 60.786245068179
        y_pred[2,:]     = 2.5

        self.y_prec     = y_pred
        u_pred          = np.zeros([self.n_inp,self.N])
        u_pred[0:4,:]   = -32000
        u_pred[4,:]     = 70.5
        u_pred[5,:]     = 70.0
        self.u_prec     = u_pred

        cp = 4186
        Pb_pred         = np.zeros([2, self.N])
        Pb_pred[0,:]    = cp * (self.y_prec[2,:] - self.q_eb) * (self.u_prec[4,:] - self.y_prec[1,:])
        Pb_pred[1,:]    = cp * self.q_eb * (self.u_prec[5,:] - self.y_prec[1,:])
        self.Pb_prec    = Pb_pred
        self.sigma_prec = 0.05 * np.ones((self.N, self.n_out))

        ### bayesian last layer ###
        # The trained network stores the last-layer weights in the MATLAB structure.
        # These are used to construct the Bayesian correction term around the mean
        # prediction and provide uncertainty-aware control inputs in the advanced modes.
        layers= self.data['layers'][0, 0]  
        weight=layers['weights']
        weights=weight[0,0]
        U0 = weights['U0'][0][0]
        b0 = weights['b0'][0][0]

        ### NNARX Hyperparameters Extraction
        output_scaler=self.data['output_scaler']
        output_scaler=output_scaler[0,0]
        self.out_bias=output_scaler['bias'][0]
        self.out_scale=output_scaler['scale'][0]

        self.theta=np.vstack([U0,b0])
        self.L_prev=self.L_prev0*np.eye(10) #TUNING
        self.Q_prev=np.linalg.inv(self.L_prev)@self.theta #MODIFIED
        self.sigma_epsilon2 = self.sigma_epsilon0_2 #MODIFIED - TUNING

        self.strategy = _STRATEGY_FACTORY[self.Model]()
        self.strategy.on_init(self)

    def ensure2D(self, x: np.ndarray):
        """Return a column-oriented 2D array for arrays that are only 1D."""
        if x.ndim == 1:
            return np.expand_dims(x, axis=1)
        return x

    def np2torch(self, x):
        """Convert NumPy data into a PyTorch tensor in the float format used by the model code."""
        return torch.from_numpy(np.asarray(x)).float()

    def SetOptimizationVariables(self):
        """Create the CasADi optimization variables for the receding-horizon problem."""
        # The optimization variables represent the sequence of future control actions
        # and the corresponding predicted thermal trajectory over the whole horizon.
        # Their shape is fixed intentionally because later parts of the MPC logic rely
        # on their indexing and structure when building constraints and warm-starts.
         ### MPC optimization Variables ###
        self.opti = casadi.Opti()
        self.u = self.opti.variable(self.n_inp, self.n_uopt)
        self.y = self.opti.variable(self.n_out, self.N)
        self.y_nnarx = self.opti.variable(self.n_out, self.N)

        self.J  = self.opti.variable(1, 1)
        self.Pb  = self.opti.variable(2, self.N)
        self.slack_y = self.opti.variable(self.n_out, self.N)

        self.strategy.declare_extra_optimization_variables(self)

    def SetConstraints(self):
        """Add temperature, flow, power, and dynamic constraints to the optimization problem.

        These constraints preserve the operational envelope of the thermal system
        while the controller searches for a feasible action sequence.
        """
        # Constraints are the operational envelope of the controller. They ensure that
        # the optimizer stays inside the admissible domain of the district-heating
        # system: temperature bounds, flow limits, power limits, and smoothness of the
        # input sequence between adjacent control intervals.
        # The policy is to keep the problem feasible while still allowing a meaningful
        # economic trade-off between comfort and operating cost.
         ### Constraints ###

        # The generator and boiler outlet temperatures are bounded by the plant
        # operational envelope. This keeps the predicted solution in realistic ranges.
        self.opti.subject_to(self.Ts_min_eb <= self.u[5,:])
        self.opti.subject_to(self.Ts_max_eb>= self.u[5,:])
        self.opti.subject_to(self.Ts_min_gb<= self.u[4,:])
        self.opti.subject_to(self.Ts_max_gb >= self.u[4,:])

        # The first four control inputs represent the thermal power loads for the four
        # plant branches. They are tied to the demand forecast through a shifted load
        # profile along the horizon.
        if self.t > self.num_steps - 4:
            self.opti.subject_to(self.Potenze[:,self.t-2] + self.P_Shift_mpc1 == self.u[0:4, 0])
            self.opti.subject_to(self.Potenze[:,self.t-2+3]+ self.P_Shift_mpc2 == self.u[0:4, 1])
            self.opti.subject_to(self.Potenze[:,self.t-2+6]+ self.P_Shift_mpc3 == self.u[0:4, 2])
            self.opti.subject_to(self.Potenze[:,self.t-2+9]+ self.P_Shift_mpc4 == self.u[0:4, 3])
        else:
            self.opti.subject_to(self.Potenze[:,self.t] + self.P_Shift_mpc1 == self.u[0:4, 0])
            self.opti.subject_to(self.Potenze[:,self.t+3]+ self.P_Shift_mpc2 == self.u[0:4, 1])
            self.opti.subject_to(self.Potenze[:,self.t+6]+ self.P_Shift_mpc3 == self.u[0:4, 2])
            self.opti.subject_to(self.Potenze[:,self.t+9]+ self.P_Shift_mpc4 == self.u[0:4, 3])

        # The nonlinear model is used as the process equation inside the optimization.
        # The predicted outputs y and the surrogate model output y_nnarx must match the
        # recurrent dynamics generated by the NNARX/GP layer.
        y_N,output_rnn,sigma_est= self.model_rnn(self.t,self.data, self.u, self.y_prec,self.u_prec,self.xk,self.uk,self.y,self.N/self.Nb,self.y_rnn,self.theta)

        self.strategy.feedback_constraint(self, y_N, output_rnn)
        self.strategy.output_constraints(self, sigma_est)

        # The last two inputs are outlet temperatures; they are limited to a small
        # step variation with respect to the previous solution so the control action
        # remains smooth and physically realistic from one optimization cycle to the next.
        delta = 5 #TUNING
        val_prec = float(self.u_prec[5, 0])
        val_correnti = self.u[5, 0:self.n_uopt-1]
        vect = casadi.horzcat(val_prec, val_correnti)
        self.opti.subject_to(-delta <= self.u[5,:] - vect)
        self.opti.subject_to( delta >= self.u[5,:] - vect)

        val_prec = float(self.u_prec[4, 0])
        val_correnti = self.u[4, 0:self.n_uopt-1]
        vect = casadi.horzcat(val_prec, val_correnti)
        self.opti.subject_to(-delta <= self.u[4,:] - vect)
        self.opti.subject_to( delta >= self.u[4,:] - vect)

        cp=4186
        # Power balance equations link the predicted outputs and the control actions to
        # the thermal power generated by the heating units. These equations are used to
        # keep the plant power within the admissible operating envelope.
        for j in range(0,self.N):
            ii = min(np.floor(j/self.Nb), self.N/self.Nb-1)
            if j==0:
                self.opti.subject_to( cp * casadi.minus(self.xk[2,self.t],self.q_eb*casadi.DM.ones(1,1))* (self.u[4,j]-self.xk[1, self.t]) == self.Pb[0,0])
                self.opti.subject_to( cp * self.q_eb*casadi.DM.ones(1,1)* (self.u[5,0]-self.xk[1, self.t]) == self.Pb[1,0])

            else:
                self.opti.subject_to( cp * casadi.minus(self.y[2,j-1],self.q_eb*casadi.DM.ones(1,1))* (self.u[4,ii]-self.y[1, j-1]) == self.Pb[0,j])
                self.opti.subject_to( cp * self.q_eb*casadi.DM.ones(1,1)* (self.u[5,ii]-self.y[1,j-1]) == self.Pb[1,j])
        self.opti.subject_to(self.Pb_min <= self.Pb[0,:])
        self.opti.subject_to(self.Pb_max >= self.Pb[0,:])
        self.opti.subject_to(self.Pb_min_eb<= self.Pb[1,:])
        self.opti.subject_to(self.Pb_max_eb >= self.Pb[1,:])

    def SetCostFunction(self):
        """Define the objective used by the MPC controller.

        The cost balances energy cost, thermal tracking, and slack penalties while
        keeping the model output within the controlled operating envelope.
        """
        # The objective function is the mechanism that trades off economic efficiency
        # against comfort and feasibility: lower energy cost is desirable, but the
        # controller must still respect thermal and hydraulic constraints and keep the
        # output trajectory close to the reference conditions. The optimization is not
        # purely economic; it also penalizes deviations from the desired delivery
        # temperature and the use of large slack variables when constraints are relaxed.
        ### Cost Function  ### 

        gamma = self.gamma
        T_ref = self.T_ref 
        c_gas = self.c_gas
        COP = self.COP
        eff_EB = self.eff_EB
        gamma_exp0 = self.gamma_exp0
        gamma_slack0 = self.gamma_slack0
        alpha_slack0 = self.alpha_slack0
        gamma_pred_gas =c_gas*casadi.DM.ones(1,self.N)
        r=int(np.floor(self.t/3)) 
        #gamma_pred_el =self.c_el[r,0]/1000*casadi.DM.ones(1,self.N) #$/KWh

        gamma_pred_el = np.zeros((self.N,1))    #MODIFIED
        for i in range(self.N):
            idx = int(np.floor((self.t+i) * len(self.c_el) / self.num_steps))
            idx = min(idx, len(self.c_el)-1)
            gamma_pred_el[i] = self.c_el[idx]/1000 #c_el[idx] ALB come si vede dai file plot e dai codici test, i prezzi devono essere /1000, così ottimizzazione sembra più sensata
        gamma_pred_el = gamma_pred_el.T

        alpha_slack = alpha_slack0*casadi.MX.ones((1, self.n_slack))  #TUNING PARAMETER
        
        gamma_exp=gamma_exp0*np.ones((1, self.n_out)) #TUNING PARAMETER
        
        #gamma_slack=1000 #TUNING PARAMETER
        gamma_slack = gamma_slack0*np.array([self.out_scale[0], self.out_scale[1], self.out_scale[2]]) #MODIFIED

        self.strategy.cost_expression(self, gamma_pred_gas, gamma_pred_el, alpha_slack, gamma_exp, gamma_slack)

    def mpc_controller(self, t,  T_C0, xk, uk, y_rnn, flag):
        """Solve the MPC optimization over the current prediction horizon.

        Args:
            t: current time step.
            T_C0: time at which control is activated.
            xk: measured state history used as the current system state.
            uk: previous control inputs.
            y_rnn: recent network-based predictions.
            flag: exploration flag used by the controller.

        Returns:
            u_opt, slack, y_nnarx, computation_time, sigma_pred, slack_explo, sol_feas
        """
        # This is the core of the control loop. At each time step it:
        # 1) collects the current plant state and past inputs,
        # 2) builds the optimization problem using the current model,
        # 3) solves the finite-horizon control problem,
        # 4) keeps only the first action and stores the warm-start state for the next step.
        # The logic is intentionally left in a single method to preserve the existing
        # runtime contract and project structure.
        start_time = time.time()
        data = self.data
        Potenze = self.Potenze
        #Potenze = np.transpose(Potenze)
        c_el = self.c_el
        # This flag records whether the optimizer solved the problem directly,
        # needed a relaxed retry, or had to fall back to the last feasible trajectory.
        sol_feas = 0  #MOD_new1 aggiungo questa variabile per salvare stato feasibility

        # The controller solves a prediction problem over N steps and reduces the
        # sequence to a smaller number of control updates using the block size Nb.
        N=self.N #prediction horizon
        n_slack = self.n_slack
        n_inp = self.n_inp
        n_out = self.n_out
        n_uopt = int(self.N / self.Nb)

        # Attach the current runtime data to the controller instance so the constraint
        # and cost builders can access the correct state vector and input history.
        self.xk=xk
        self.uk=uk
        self.t=t
        self.y_rnn=y_rnn
        self.flag= flag
        self.SetOptimizationVariables()

        # Before T_C0 the controller is still in startup mode: the optimization is not
        # yet active, and the previous nominal action is reused to keep the loop stable.
        if t<T_C0:
            u_opt = self.u_prec[:,0]
            u_pred = self.u_prec
            y_pred = self.y_prec
            y_nnarx=self.y_prec
            Pb_pred=self.Pb_prec
            slack= np.zeros([n_slack, 1])
            J_pred = self.J_prec
            computation_time = 0
            sigma_pred = 0.05 * np.ones((self.N, self.n_out))
            slack_explo= np.zeros([n_out, N])

        else:
            # Build the finite-horizon optimization problem using the latest measurements
            # and model predictions. This is where the process constraints, system dynamics,
            # and cost function are embedded in the CasADi problem.
            self.SetConstraints()
            self.SetCostFunction()

            # Warm-start the solver from the previous feasible trajectory. This keeps the
            # optimizer stable and reduces the time needed to converge to the next horizon.
            for i in range(n_inp):
                initial_guess = np.concatenate([self.u_prec[i, 1:n_uopt], [self.u_prec[i, -1]]])
                self.opti.set_initial(self.u[i,:], initial_guess)

            for i in range(n_out):
                y_initial = np.concatenate([ self.y_prec[i, 1:N],[self.y_prec[i, -1]]])
                self.opti.set_initial(self.y[i,:],  y_initial)
                self.opti.set_initial(self.y_nnarx[i,:],  y_initial)
            for i in range(2):
                Pb_initial = np.concatenate([self.Pb_prec[ i,1:], [self.Pb_prec[ i,-1]]])
                self.opti.set_initial(self.Pb[i,:], Pb_initial)
            self.opti.set_initial(self.J, self.J_prec)

            self.strategy.warm_start_extra_variables(self)

            # Objective function is minimized over the horizon while respecting the current
            # model constraints. The optimization uses IPOPT as the underlying solver.
            self.opti.minimize(self.J)
            prob_opts = {
                'expand': True,
                'ipopt': {
                    'print_level': 3,
                },
                'print_time': False
            }

            # IPOPT settings  #MODIFY - TUNING PARAMETER
            ip_opts = {
                'print_level': 3,
                'max_iter': int(1e4),
                'compl_inf_tol': 1e-5
            }

            # Set the solver
            self.opti.solver('ipopt', prob_opts, ip_opts)

            # Solve the FHOCP. The first attempt is the nominal one, but if the solve fails
            # the method relaxes the optimality tolerances and retries. If that still fails,
            # the last known feasible input trajectory is used as a safe fallback.
            try:
                sol = self.opti.solve()
                print('*** Problem solved ***')
                sol_feas = 1 #MOD_new1 aggiunto per salvare stato feasibility
                u_opt = sol.value(self.u[:, 0])
                u_pred = sol.value(self.u)

                y_pred = sol.value(self.y)
                slack, slack_explo, sigma_pred = self.strategy.extract_solution(self, sol)
                Pb_pred=sol.value(self.Pb)
                J_pred = sol.value(self.J)
                y_nnarx=sol.value(self.y_nnarx)

            except Exception as ex:
                print('*** Problem not solved  -  First IPOPT attempt failed, retrying with relaxed tolerances ***')
                try:
                    # The second attempt preserves the same formulation but relaxes the
                    # solver tolerance so the optimizer can recover from a near-feasible
                    # but numerically difficult step.
                    self.opti.minimize(0.01 * self.J)
                    prob_opts = {
                        'expand': True,
                        'ipopt': {
                            'print_level': 0,
                        },
                        'print_time': False
                    }
                    ip_opts = {
                        'print_level': 0,
                        'max_iter': int(1e4),
                        'compl_inf_tol': 1e-4,
                        'acceptable_tol': 1e-4,
                        'acceptable_iter': 10,
                        'mu_strategy': 'adaptive'
                    }
                    self.opti.solver('ipopt', prob_opts, ip_opts)
                    sol = self.opti.solve()
                    sol_feas = 2  #MOD_new1 aggiunto per salvare stato feasibility
                    print('*** Problem solved on second attempt ***')
                    u_opt = sol.value(self.u[:, 0])
                    u_pred = sol.value(self.u)
                    y_pred = sol.value(self.y)
                    slack, slack_explo, sigma_pred = self.strategy.extract_solution(self, sol)
                    Pb_pred = sol.value(self.Pb)
                    J_pred = sol.value(self.J)
                    y_nnarx = sol.value(self.y_nnarx)

                except Exception as ex2:
                    print('*** Problem not solved  -  Takes last Feasible Solution!***')
                    # This fallback keeps the system running even when the optimizer fails.
                    # It reuses the last stable plan instead of producing a broken or
                    # inconsistent control command.
                    sol_feas = 3  #MOD_new1 aggiunto per salvare stato feasibility
                    fallback_u_pred = self.u_prec.copy()
                    fallback_u_pred[0:4, 0] = self.Potenze[:, self.t] + self.P_Shift_mpc1
                    u_opt = fallback_u_pred[:,0]
                    u_pred = fallback_u_pred
                    y_pred = self.y_prec

                    slack = np.zeros([n_slack,1])
                    slack_explo = self.strategy.fallback_slack_explo(self)

                    Pb_pred=self.Pb_prec
                    J_pred = self.J_prec
                    y_nnarx=self.y_prec
                    sigma_pred = self.sigma_prec

        # Save the latest accepted optimization result so the next MPC call has a
        # reliable warm start and a consistent notion of the previous horizon.
        self.J_prec=J_pred
        self.Pb_prec=Pb_pred
        self.y_prec=y_pred
        self.u_prec=u_pred
        self.sigma_prec=sigma_pred

        end_time=time.time()
        computation_time=end_time-start_time

        return  u_opt,slack,y_nnarx,computation_time,sigma_pred, slack_explo, sol_feas  #MOD_new1 aggiunto sol_feas agli output

    def simulate_dynamics(self, k, initial_state, input_vector, A, Bu, Bx, C, steps, weights, n_states, ground_truth, theta):
        """Advance the internal state-space representation over the prediction horizon."""
        n_out = self.n_out
        states = casadi.MX.zeros(n_states)
        outputs = casadi.MX.zeros((n_out, self.N))
        states[:] = initial_state.reshape(24)
        eta = ground_truth[k, :].T
        sigma_k = casadi.MX.zeros((self.N,self.n_out))

        for k_sim in range(0, self.N):

            s1 = casadi.reshape(np.matmul(A, states[:]), (24, 1))
            s2 = casadi.reshape(np.matmul(Bu, input_vector[k_sim, :].T), (24, 1))
            s3 = casadi.reshape(np.matmul(Bx, eta), (24, 1))
            states[:] = s1 + s2 + s3

            eta,sigma = self.activation_fun(input_vector[k_sim, :], weights, states[:],theta)
            outputs[:, k_sim] = eta
            sigma_k[k_sim,:]=sigma
        return states, outputs,sigma_k

    def model_rnn(self, k, data, u, y_pred, u_prec, xk, uk, y, n_uopt, y_rnn, theta): 
        """Build the NNARX state-space prediction used inside the optimization loop.

        The method normalizes inputs, reconstructs the prior state vector, and
        simulates the neural-network-based dynamics that are then used as the model
        prediction of the thermal system.
        """
        # This routine is the bridge between the measured system trajectory and the
        # optimization model embedded in CasADi. It reconstructs a state vector from
        # the last few outputs and inputs, scales it to the NNARX training domain,
        # and then iterates the recurrent dynamics over the prediction horizon.
        # The result is a prediction of future temperatures and flows that can be used
        # directly inside the MPC constraints and objective functions.

        N=self.N
        horizon=3
        Nb=self.Nb
        n_out=self.n_out
        n_inp=5
        q_eb=self.q_eb
        n_states=horizon*(n_out+n_inp)

        ### State space model of NNarx: matrices definitions ##
        # The recurrent state representation is built with a 3-step delay structure:
        # it stores the most recent output and input history, then evolves the latent
        # state through a linear dynamics template before the nonlinear NNARX update.
        # This is a compact way to encode the temporal dependence of the thermal plant.
        _a = np.zeros((horizon, horizon))
        _a[range(0, horizon - 1), range(1, horizon)] = 1
        A = np.kron(_a, np.eye((n_inp + n_out)))
        _b = np.zeros((horizon, 1))
        _b[-1, 0] = 1
        _bx = np.concatenate([np.eye(n_out), np.zeros([n_inp, n_out])], axis=0)
        _bu = np.concatenate([np.zeros([n_out, n_inp]), np.eye(n_inp)], axis=0)
        Bu = np.kron(_b, _bu)
        Bx = np.kron(_b, _bx)
        c=np.concatenate([np.eye(n_out), np.zeros([n_out,n_inp])], axis=1)
        C= np.concatenate([np.zeros([n_out,(n_out+n_inp)*(horizon-1)]),c],axis=1)


        ### NNARX Hyperparameters Extraction
        layers=data['layers'][0, 0] 
        weights=layers['weights'][0,0]
        input_scaler=data['input_scaler']
        input_scaler=input_scaler[0,0]
        inp_bias=input_scaler['bias'][0]
        inp_scale=input_scaler['scale'][0]

        output_scaler=data['output_scaler']
        output_scaler=output_scaler[0,0]
        out_bias=output_scaler['bias'][0]
        out_scale=output_scaler['scale'][0]

        ### Normalization input vector
        U = casadi.MX.zeros(( n_inp,N))
        u_rnn = np.zeros([ n_inp,self.num_steps])
        for i in range(n_inp):
            for j in range(N):
                ii=min(np.floor(j/Nb),N/Nb-1)
                U[i,j]=(u[i,ii]-inp_bias[i]*casadi.MX.ones((1)))/inp_scale[i]
                if i==n_inp-1:
                    if j==0:
                        U[i,j]= (((u[5,j]*q_eb+( xk[2,k] -q_eb*casadi.DM.ones(1))*u[4,j])/casadi.fmax(xk[2,k], 1e-6)) -inp_bias[i])/inp_scale[i]
                    else:
                        U[i,j]= (((u[5,ii]*q_eb+( y[2,j-1] -q_eb*casadi.DM.ones(1))*u[4,ii])/casadi.fmax(y[2,j-1], 1e-6)) -inp_bias[i])/inp_scale[i]
        U=U.T
        for i in range(n_inp):
            for j in range(k):
                
                u_rnn[i,j]=(uk[i,j]-inp_bias[i]*casadi.MX.ones((1)))/inp_scale[i]
                if i==n_inp-1:
                    u_rnn[i,j]= (((uk[5,j]*q_eb+( xk[2,j] -q_eb*casadi.DM.ones(1))*uk[4,j])/xk[2,j]) -inp_bias[i])/inp_scale[i]
        
        ### Current State ###
        initial_state=np.zeros([n_states,1])
        initial_state[0,:] =(xk[0,k-3]-out_bias[0])/out_scale[0]
        initial_state[1,:] = (xk[1,k-3]-out_bias[1])/out_scale[1]
        initial_state[2,:] = (xk[2,k-3]-out_bias[2])/out_scale[2]
        initial_state[3,:] =(uk[0,k-3]-inp_bias[0])/inp_scale[0]
        initial_state[4,:] = (uk[1,k-3]-inp_bias[1])/inp_scale[1]
        initial_state[5,:] = (uk[2,k-3]-inp_bias[2])/inp_scale[2]
        initial_state[6,:] =(uk[3,k-3]-inp_bias[3])/inp_scale[3]
        initial_state[7,:] =(((uk[5,k-3]*q_eb+( xk[2,k-3] -q_eb*np.ones([1,1]))*uk[4,k-3])/xk[2,k-3])-inp_bias[4])/inp_scale[4]
        initial_state[8,:] =(xk[0,k-2]-out_bias[0])/out_scale[0]
        initial_state[9,:] = (xk[1,k-2]-out_bias[1])/out_scale[1]
        initial_state[10,:] = (xk[2,k-2]-out_bias[2])/out_scale[2]
        initial_state[11,:] =(uk[0,k-2]-inp_bias[0])/inp_scale[0]
        initial_state[12,:] = (uk[1,k-2]-inp_bias[1])/inp_scale[1]
        initial_state[13,:] = (uk[2,k-2]-inp_bias[2])/inp_scale[2]
        initial_state[14,:] =(uk[3,k-2]-inp_bias[3])/inp_scale[3]
        initial_state[15,:] =(((uk[5,k-2]*q_eb+( xk[2,k-2] -q_eb*np.ones([1,1]))*uk[4,k-2])/xk[2,k-2])-inp_bias[4])/inp_scale[4]
        initial_state[16,:] =(xk[0,k-1]-out_bias[0])/out_scale[0]
        initial_state[17,:] = (xk[1,k-1]-out_bias[1])/out_scale[1]
        initial_state[18,:] = (xk[2,k-1]-out_bias[2])/out_scale[2]
        initial_state[19,:] = (uk[0,k-1]-inp_bias[0])/inp_scale[0]
        initial_state[20,:] =(uk[1,k-1]-inp_bias[1])/inp_scale[1]
        initial_state[21,:] =(uk[2,k-1]-inp_bias[2])/inp_scale[2]
        initial_state[22,:] =(uk[3,k-1]-inp_bias[3])/inp_scale[3]
        initial_state[23,:] =(((uk[5,k-1]*q_eb+( xk[2,k-1] -q_eb*np.ones([1,1]))*uk[4,k-1])/xk[2,k-1])-inp_bias[4])/inp_scale[4]
        
        ### Output Normalization ###
        # Measured and predicted outputs are normalized to the same scale used during
        # training. This is essential because the learned NNARX/BNN model was trained on
        # normalized signals, while the controller works in physical units.
        Y_pred=casadi.MX(n_out,N)
        Y_RNN=casadi.MX(n_out,N)
        ys=np.zeros([self.num_steps,n_out])
        ground_truth=np.zeros([self.num_steps,n_out])
        for i in range(n_out):
            for j in range(k+1):
                ys[j,i]=(y_rnn[j,i]-out_bias[i])/out_scale[i]
                ground_truth[j,i]=(xk[i,j]-out_bias[i])/out_scale[i]
    
        ### NNARX Prediction ###
        # We simulate the recurrent dynamics over the whole horizon using the current
        # state estimate and the candidate control sequence. The function returns the
        # predicted output trajectory together with the estimated uncertainty term.
        states,outputs,Sigma=self.simulate_dynamics(k,initial_state, U[:,:], A, Bu, Bx, C, N,weights,n_states,ground_truth,theta)

        Y_pred, Y_RNN = self.strategy.predict_outputs(
            self, k, outputs, ys, u_rnn, U, ground_truth,
            out_scale, out_bias, n_out, N, Y_pred, Y_RNN
        )

        return Y_pred,Y_RNN, Sigma

    def activation_fun(self, input_vector, weights, state_vector, theta):
        """Evaluate the neural activation used inside the model-based dynamics simulation."""
        # The prediction model is built as a structured recurrent network: the hidden
        # state is updated from the current state and inputs, then a nonlinear activation
        # produces the output estimate. This is the core computational block used in the
        # forward simulation of the NNARX model.
        # 3 layers total: recurrent dynamics plus hidden layers followed by the output projection.
        U0 = weights['U0'][0][0]
        b0 = weights['b0'][0][0]
        U0=U0
        b0=b0
        #weights dei due hidden layers
        theta = self.strategy.select_theta(self, theta, U0, b0)

        W1 = weights['W.0'][0][0]
        W1=W1
        U1 = weights['U.0'][0][0]
        U1=U1
        b1 = weights['b.0'][0][0]
        b1=b1

        W2 = weights['W.1'][0][0]
    
        U2 = weights['U.1'][0][0]
        
        b2 = weights['b.1'][0][0]
        
    
        n_neurons=9
        # The recurrent network uses two nonlinear hidden layers fed by the current
        # plant state and the candidate input vector. The final feature map appends a
        # constant term to construct the Bayesian last-layer basis used in the adaptive
        # correction step.
        z1=np.matmul(np.transpose(U1[:,:]),state_vector)
        z2=np.matmul(np.transpose(W1[:,:]),input_vector.T) 
        z3=np.transpose(b1[:,:].reshape(n_neurons,))
        z=z1+z2+z3
        e1=np.tanh(z)
        z1=np.matmul(np.transpose(U2[:,:]),e1)
        z2=np.matmul(np.transpose(W2[:,:]),input_vector.T) 
        z3=np.transpose(b2[:,:].reshape(n_neurons,))
        z_l2=z1+z2+z3
        e_vect=np.tanh(z_l2)
        #theta=casadi.vertcat(U0,b0)
        f_tilde=casadi.vertcat(e_vect,1)
        eta=theta.T@f_tilde
        sigma=np.sqrt((1+f_tilde.T@self.L_prev@f_tilde) * self.sigma_epsilon2)

        # e=np.matmul(U0.transpose(),e_vect)

        # eta = e + np.transpose(b0.reshape(3,))

        return eta,sigma

    def bayesian_correction(self, s, y_star, u):
        """Update the Bayesian last layer using recent data from the measured trajectory.

        This is the mechanism that adapts the model when the controller is operating
        with uncertainty-aware corrections. Delegated to the active model strategy;
        only BNNExp implements it.
        """
        return self.strategy.bayesian_correction(self, s, y_star, u)

