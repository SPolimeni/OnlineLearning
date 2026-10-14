"""Per-model strategy classes used by class_MPC.MPC.

Each supported value of Param["Model"] ('NNARX', 'GP', 'BNNExp') maps to one
strategy class here. ModelStrategy defines the default hook behavior, which
is exactly the NNARX case (the common/baseline behavior); GPStrategy and
BNNExpStrategy override only the hooks where their behavior differs from
that baseline, and each owns all of its model-specific state and code in one
place instead of it being scattered across MPC as `if self.Model == ...`
branches.
"""

import gpytorch
import torch

import casadi
import numpy as np


class ModelStrategy:
    """Default hooks = NNARX behavior, the common baseline shared by all models."""

    def on_init(self, mpc):
        """Initialize any strategy-owned state once the MPC instance is built."""
        pass

    def declare_extra_optimization_variables(self, mpc):
        """Create the model-specific CasADi optimization variables."""
        mpc.T_slack = mpc.opti.variable(1, mpc.N)

    def feedback_constraint(self, mpc, y_N, output_rnn):
        """Tie the optimization's output trajectory to the model prediction."""
        mpc.opti.subject_to(mpc.y == output_rnn[:, 0:mpc.N])
        mpc.opti.subject_to(mpc.y_nnarx == output_rnn[:, 0:mpc.N])

    def output_constraints(self, mpc, sigma_est):
        """Add the comfort/operational-envelope constraints on the output trajectory."""
        mpc.opti.subject_to(mpc.Ts_max + mpc.T_slack >= mpc.y[0, :])
        mpc.opti.subject_to(mpc.Tr_min <= mpc.y[1, :])
        mpc.opti.subject_to(mpc.Tr_max >= mpc.y[1, :])
        mpc.opti.subject_to(mpc.m_min <= mpc.y[2, :])
        mpc.opti.subject_to(mpc.m_max >= mpc.y[2, :])
        mpc.opti.subject_to(mpc.y[0, :] >= mpc.Ts_min_supply - mpc.T_slack)
        mpc.opti.subject_to(mpc.T_slack >= 0)
        mpc.opti.subject_to(mpc.T_slack <= 20)

    def cost_expression(self, mpc, gamma_pred_gas, gamma_pred_el, alpha_slack, gamma_exp, gamma_slack):
        """Define the J == ... cost constraint for this model."""
        mpc.opti.subject_to(
            mpc.J == gamma_pred_gas @ (1 / 1000 * mpc.Ts / 3600 * mpc.Pb[0, :].T / mpc.COP)
            + gamma_pred_el @ (1 / 1000 * mpc.Ts / 3600 * mpc.Pb[1, :].T / mpc.eff_EB)
            + mpc.gamma * (mpc.y[0, mpc.N - 1] - mpc.T_ref * casadi.MX.ones((1, 1))) ** 2
            + mpc.T_slack @ mpc.T_slack.T * alpha_slack
        )

    def warm_start_extra_variables(self, mpc):
        """Set the initial guess for the model-specific optimization variables."""
        mpc.opti.set_initial(mpc.slack_y, np.zeros([mpc.n_slack, 1]))

    def extract_solution(self, mpc, sol):
        """Return (slack, slack_explo, sigma_pred) from a successful solve."""
        return np.zeros([mpc.n_slack, 1]), 0, mpc.sigma_prec

    def fallback_slack_explo(self, mpc):
        """Return the slack_explo value used when the solver fails on every attempt."""
        return 0

    def select_theta(self, mpc, theta, U0, b0):
        """Select the last-layer weights used inside the recurrent activation."""
        return casadi.vertcat(U0, b0)

    def predict_outputs(self, mpc, k, outputs, ys, u_rnn, U, ground_truth,
                         out_scale, out_bias, n_out, N, Y_pred, Y_RNN):
        """Denormalize the simulated NNARX outputs into physical units."""
        for i in range(n_out):
            for j in range(N):
                Y_RNN[i, j] = out_scale[i] * outputs[i, j] + out_bias[i] * np.ones([1, 1])
        return Y_pred, Y_RNN


class NNARXStrategy(ModelStrategy):
    """NNARX is the baseline itself: no overrides needed."""
    pass


class GPStrategy(ModelStrategy):
    """Gaussian-process residual correction layered on top of the NNARX prediction."""

    class GPRegressionModel(gpytorch.models.ExactGP):
        """Small GP helper used for the model-error correction pattern."""

        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module    = gpytorch.means.ConstantMean()
            self.covar_module   = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale=1.0,
                    length_scale_bounds=(1e-3, 1e3),
                    ard_num_dims=27
                )
            )

        def forward(self, x):
            mean_x  = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def on_init(self, mpc):
        self.gp_buffer = {
            "y_buffer_1": [],
            "y_buffer_2": [],
            "y_buffer_3": [],
            "X_buffer_1": [],
            "X_buffer_2": [],
            "X_buffer_3": []
        }
        self.y_gp = np.empty((mpc.num_steps + mpc.N, mpc.n_out), dtype=object)

    def feedback_constraint(self, mpc, y_N, output_rnn):
        mpc.opti.subject_to(mpc.y == y_N[:, 0:mpc.N])
        mpc.opti.subject_to(mpc.y_nnarx == output_rnn[:, 0:mpc.N])

    def predict_outputs(self, mpc, k, outputs, ys, u_rnn, U, ground_truth,
                         out_scale, out_bias, n_out, N, Y_pred, Y_RNN):
        output_gp = self.gaussian_process(mpc, k, ys, outputs, u_rnn.T, U, ground_truth)
        output_gp = output_gp.transpose()

        for i in range(n_out):
            for j in range(N):
                r = (output_gp[i, j])
                Y_pred[i, j] = out_scale[i] * r + out_bias[i] * np.ones([1, 1])
                Y_RNN[i, j] = out_scale[i] * outputs[i, j] + out_bias[i] * np.ones([1, 1])

        return Y_pred, Y_RNN

    def gaussian_process(self, mpc, k, ys, y_mpc, input, u_mpc, ground_truth):
        """Apply the GP-based correction pattern used when the controller operates in GP mode.

        This helper is part of the prediction layer and estimates how observed
        errors evolve around the current state values.
        """
        ### Parameters ###
        h=3
        window_size=60
        n_inp=5
        y_mpc=y_mpc.T
        n_out=3
        N=mpc.N
        #### Training of GP ###
        x_k_1=[]
        x_k_2=[]
        x_k_3=[]
        x_tilde_1=[]
        x_tilde_2=[]
        x_tilde_3=[]
        u_1=[]
        u_1_inp=[]
        u_1_out=[]
        u_2=[]
        u_2_inp=[]
        u_2_out=[]
        u_3=[]
        u_3_inp=[]
        u_3_out=[]

        x_k_test_1=[]
        x_k_test_2=[]
        x_k_test_3=[]
        u_k_test_1=[]
        u_k_test_2=[]
        u_k_test_3=[]
        x_tilde_test_1=[]
        x_tilde_test_2=[]
        x_tilde_test_3=[]
        y_final=np.empty((N, n_out), dtype=object)

        y_buffer_1= self.gp_buffer[ "y_buffer_1"]
        y_buffer_2=self.gp_buffer[ "y_buffer_2"]
        y_buffer_3=self.gp_buffer[ "y_buffer_3"]

        X_buffer_1= self.gp_buffer[ "X_buffer_1"]
        X_buffer_2=self.gp_buffer[ "X_buffer_2"]
        X_buffer_3=self.gp_buffer[ "X_buffer_3"]

        ### Reduction of the dataset ###
        if len(X_buffer_1) > window_size-1:
            X_buffer_1.pop(0)
            X_buffer_2.pop(0)
            X_buffer_3.pop(0)

            y_buffer_1.pop(0)
            y_buffer_2.pop(0)
            y_buffer_3.pop(0)


        for j in range(n_out):

            for i in range(h):
                u_stato=[]

                if k-(h-i)-1<0:
                    for s in range(n_inp):
                        u_stato.insert(0,input[0,s])
                    u_out=ys[0,j]
                else:
                    for s in range(n_inp):
                        u_stato.insert(0,input[k-(h-i)-1, s])
                    u_out=(ys[k-(h-i)-1, j])

                if k-(h-i-1)-1<0:
                    x_k=ground_truth[0,j]-ys[0,j]
                else:

                    x_k=ground_truth[k-(h-i-1)-1,j]-ys[k-(h-i-1)-1,j]
                if j==0:

                    u_1_inp=u_stato+u_1_inp
                    u_1_out.insert(0,u_out)
                    x_k_1.insert(0,x_k)
                elif j==1:
                        u_2_inp=u_stato+u_2_inp
                        u_2_out.insert(0,u_out)
                        x_k_2.insert(0,x_k)
                else:
                    u_3_inp=u_stato+u_3_inp
                    u_3_out.insert(0,u_out)

                    x_k_3.insert(0,x_k)


        y_k_1 = ground_truth[k, 0] - ys[k, 0]
        y_k_2 = ground_truth[k, 1] - ys[k, 1]
        y_k_3 = ground_truth[k, 2] - ys[k, 2]

        for s in range(n_inp):
            u_1_inp.insert(0,input[k-1, s])
            u_2_inp.insert(0,input[k-1, s])
            u_3_inp.insert(0,input[k-1, s])
        u_1_out.insert(0,ys[k-1, 0])
        u_2_out.insert(0,ys[k-1, 1])
        u_3_out.insert(0,ys[k-1, 2])
        u_1.extend(u_1_inp)
        u_1.extend(u_1_out)
        u_2.extend(u_2_inp)
        u_2.extend(u_2_out)
        u_3.extend(u_3_inp)
        u_3.extend(u_3_out)
        x_tilde_1.extend(x_k_1)
        x_tilde_1.extend(u_1)
        x_tilde_2.extend(x_k_2)
        x_tilde_2.extend(u_2)
        x_tilde_3.extend(x_k_3)
        x_tilde_3.extend(u_3)


        #update of buffer
        X_buffer_1.append(x_tilde_1)
        X_buffer_2.append(x_tilde_2)
        X_buffer_3.append(x_tilde_3)


        y_buffer_1.append(y_k_1)
        y_buffer_2.append(y_k_2)
        y_buffer_3.append(y_k_3)

        for j in range(n_out):

            if j==0:
                x_tensor = mpc.np2torch(np.stack(X_buffer_1))

                y_tensor = mpc.np2torch(np.array(y_buffer_1) )
            elif j==1:
                x_tensor = mpc.np2torch(np.stack(X_buffer_2))
                y_tensor = mpc.np2torch(np.array(y_buffer_2))
            else:
                x_tensor = mpc.np2torch(np.stack(X_buffer_3))
                y_tensor = mpc.np2torch(np.array(y_buffer_3))



            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            model = self.GPRegressionModel(x_tensor, y_tensor, likelihood)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.3)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            r=0
            delta=10000
            prev_loss=0
            ### optimization of the GP parameters ###
            while delta>0.1 and r <200:
                optimizer.zero_grad()
                output = model(x_tensor)

                loss = -mll(output, y_tensor)
                delta=loss.item()-prev_loss
                delta=np.absolute(delta)


                loss.backward()
                optimizer.step()
                prev_loss=loss.item()
                r=r+1

            if j==0:
                model_1=model
                likelihood_1=likelihood
            elif j==1:
                model_2=model
                likelihood_2=likelihood
            else:
                model_3=model
                likelihood_3=likelihood

            for r in range(0,k+1):
                self.y_gp[r,j] = ground_truth[r,j]-ys[r,j]




        ###### TEST per k+1 e fino a k+N in simulazione ####
        t=k
        r=-1
        for k in range(t,t+N):
            r=r+1
            x_k_1=[]
            x_k_2=[]
            x_k_3=[]
            x_tilde_1=[]
            x_tilde_2=[]
            x_tilde_3=[]
            u_1=[]
            u_1_inp=[]
            u_1_out=[]
            u_2=[]
            u_2_inp=[]
            u_2_out=[]
            u_3=[]
            u_3_inp=[]
            u_3_out=[]

            x_k_test_1=[]
            x_k_test_2=[]
            x_k_test_3=[]
            u_k_test_1=[]
            u_k_test_2=[]
            u_k_test_3=[]
            x_tilde_test_1=[]
            x_tilde_test_2=[]
            x_tilde_test_3=[]


            for j in range(n_out):

                for i in range(h):
                    u_stato=[]


                    if k-(h-i)-1<0:
                        for s in range(n_inp):
                            u_stato.insert(0,input[0,s])
                        u_out=ys[0,j]
                    else:

                        for s in range(n_inp):
                            u_stato.insert(0,input[k-(h-i)-1, s])
                        u_out=(ys[k-(h-i)-1, j])

                    if k-(h-i-1)-1<0:

                        x_k=ground_truth[0,j]-ys[0,j]
                    else:

                        if k-(h-i-1)-1>=t+1:
                            x_k=self.y_gp[k-(h-i-1)-1,j]
                        else:
                            x_k=self.y_gp[k-(h-i-1)-1,j]
                    if j==0:

                        u_1_inp=u_stato+u_1_inp
                        u_1_out.insert(0,u_out)
                        x_k_1.insert(0,x_k)
                    elif j==1:
                        u_2_inp=u_stato+u_2_inp
                        u_2_out.insert(0,u_out)
                        x_k_2.insert(0,x_k)
                    else:
                        u_3_inp=u_stato+u_3_inp
                        u_3_out.insert(0,u_out)

                        x_k_3.insert(0,x_k)
            for s in range(n_inp):
                u_1_inp.insert(0,input[k-1, s])
                u_2_inp.insert(0,input[k-1, s])
                u_3_inp.insert(0,input[k-1, s])
            u_1_out.insert(0,ys[k-1, 0])
            u_2_out.insert(0,ys[k-1, 1])
            u_3_out.insert(0,ys[k-1, 2])

            u_1.extend(u_1_inp)
            u_1.extend(u_1_out)
            u_2.extend(u_2_inp)
            u_2.extend(u_2_out)
            u_3.extend(u_3_inp)
            u_3.extend(u_3_out)
            x_tilde_1.extend(x_k_1)
            x_tilde_1.extend(u_1)
            x_tilde_2.extend(x_k_2)
            x_tilde_2.extend(u_2)
            x_tilde_3.extend(x_k_3)
            x_tilde_3.extend(u_3)

            #vettori test
            x_k_1.pop()
            x_k_2.pop()
            x_k_3.pop()

            if k>=t+1:

                x_k_1.insert(0,self.y_gp[k,0] )


                x_k_2.insert(0,self.y_gp[k,1])
                x_k_3.insert(0,self.y_gp[k,2] )
            else:
                x_k_1.insert(0,self.y_gp[k,0] )


                x_k_2.insert(0,self.y_gp[k,1])
                x_k_3.insert(0,self.y_gp[k,2] )

            x_k_test_1=x_k_1

            x_k_test_2=x_k_2
            x_k_test_3=x_k_3



            for s in range(n_inp):
                u_1_inp.pop()
                u_2_inp.pop()
                u_3_inp.pop()
            u_1_out.pop()
            u_2_out.pop()
            u_3_out.pop()
            if k==t:
                print(k)
                uk=input
                input=casadi.MX.zeros(mpc.num_steps+N+N,5)
                print(input.shape)
                input[0:k,:]=uk[0:k,:]
                input[k:k+N,:]=u_mpc



            if k==t:
                xk=ys
                ys=casadi.MX.zeros(mpc.num_steps+N,3)
                ys[0:k+1,:]=xk[:k+1,:]
                ys[k+1:k+N+1,:]=y_mpc[:,:]


            for s in range(n_inp):
                u_1_inp.insert(0, input[k, s] )
                u_2_inp.insert(0,input[k, s])
                u_3_inp.insert(0,input[k, s] )
            if k>=t+1:

                u_1_out.insert(0,ys[k,0])
                u_2_out.insert(0,ys[k,1])
                u_3_out.insert(0,ys[k,2])

            else:

                u_1_out.insert(0,ys[k,0])

                u_2_out.insert(0,ys[k,1])
                u_3_out.insert(0,ys[k,2])

            u_k_test_1.extend(u_1_inp)
            u_k_test_1.extend(u_1_out)
            a=np.array(u_1_inp)
            b=np.array(u_1_out)
            c=np.array(x_k_test_1)

            u_k_test_2.extend(u_2_inp)
            u_k_test_2.extend(u_2_out)

            u_k_test_3.extend(u_3_inp)
            u_k_test_3.extend(u_3_out)
            x_tilde_test_1.extend(x_k_test_1)
            x_tilde_test_1.extend(u_k_test_1)


            x_tilde_test_2.extend(x_k_test_2)
            x_tilde_test_2.extend(u_k_test_2)


            x_tilde_test_3.extend(x_k_test_3)
            x_tilde_test_3.extend(u_k_test_3)

            for j in range(n_out)  :

                if j==0:
                    test=x_tilde_test_1
                    model=model_1
                    likelihood=likelihood_1
                    z_vect=np.array(y_buffer_1)
                    X_train=(X_buffer_1)
                elif j==1:
                    model=model_2
                    test=x_tilde_test_2
                    likelihood=likelihood_2
                    z_vect=np.array(y_buffer_2)
                    X_train=(X_buffer_2)
                else:
                    model=model_3
                    test=x_tilde_test_3
                    likelihood=likelihood_3
                    z_vect=np.array(y_buffer_3)
                    X_train=(X_buffer_3)
                model.eval()
                l= model.covar_module.base_kernel.lengthscale.detach().numpy()
                l=l.flatten()
                delta=np.diag(l**(-2))
                mean_prior=0
                sigma_F=( model.covar_module.outputscale.detach().numpy()) #squared
                sigma_N=(likelihood.noise_covar.noise.detach().numpy()) #squared
                #sigma squareS
                likelihood.eval()
                kern=casadi.MX(1,len(X_train))
                K=np.zeros([len(X_train),len(X_train)])
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    test_arr=np.array(test)
                    for i in range(len(X_train)):
                        diff_1=(test_arr-np.stack(X_train[i])).transpose()
                        #print(diff_1.shape)
                        kern[0,i]=sigma_F*np.exp(-0.5*diff_1.transpose()@(delta@diff_1))
                        for a in range(len(X_train)):
                                diff_2 =np.stack(X_train[i])-np.stack(X_train[a])

                                if i==a:
                                    delta_kronecker=1
                                else:
                                    delta_kronecker=0
                                K[i,a]=sigma_F*np.exp(-0.5*((diff_2).transpose()@(delta@diff_2)))+sigma_N*delta_kronecker

                    if t==4:

                        inverse_K= 1/K
                    else:
                        inverse_K=np.linalg.inv(K)

                    y_pred_mean=kern@( inverse_K@(z_vect-mean_prior))+mean_prior
                    y_pred_mean=np.array(y_pred_mean)
                    yk=np.array(ys[k+1, j] )

                    y_pred= yk + y_pred_mean

                    if j==0:
                        self.y_gp[k+1,j]=y_pred_mean

                    elif j==1:
                        self.y_gp[k+1,j]=y_pred_mean

                    else:
                        self.y_gp[k+1,j]=y_pred_mean

                    y_final[r, j] = y_pred

        return y_final


class BNNExpStrategy(ModelStrategy):
    """Bayesian last-layer model with uncertainty-aware exploration constraints."""

    def on_init(self, mpc):
        mpc.beta = np.array([
            mpc.beta0_1 / mpc.out_scale[0],
            mpc.beta0_2 / mpc.out_scale[1],
            mpc.beta0_3 / mpc.out_scale[2],
        ])
        mpc.epsilon = mpc.epsilon0 / mpc.beta[0] * mpc.beta

    def declare_extra_optimization_variables(self, mpc):
        mpc.slack_explo = mpc.opti.variable(mpc.n_out, mpc.N)
        mpc.sigmaK = mpc.opti.variable(mpc.N, mpc.n_out)

    def output_constraints(self, mpc, sigma_est):
        # Ts_min_supply <= Ts <= Ts_max, band widened by the uncertainty term and
        # relaxed by a slack variable when the uncertainty becomes too large.
        mpc.opti.subject_to((mpc.Ts_min_supply - mpc.out_bias[0]) / mpc.out_scale[0] + mpc.beta[0] * (mpc.sigmaK[:, 0]).T - mpc.slack_y[0, :] <= (mpc.y[0, :] - mpc.out_bias[0]) / mpc.out_scale[0])
        mpc.opti.subject_to((mpc.Ts_max - mpc.out_bias[0]) / mpc.out_scale[0] - mpc.beta[0] * (mpc.sigmaK[:, 0]).T + mpc.slack_y[0, :] >= (mpc.y[0, :] - mpc.out_bias[0]) / mpc.out_scale[0])

        # Tr_min <= Tr <= Tr_max
        mpc.opti.subject_to((mpc.Tr_min - mpc.out_bias[1]) / mpc.out_scale[1] + mpc.beta[1] * (mpc.sigmaK[:, 1]).T - mpc.slack_y[1, :] <= (mpc.y[1, :] - mpc.out_bias[1]) / mpc.out_scale[1])
        mpc.opti.subject_to((mpc.Tr_max - mpc.out_bias[1]) / mpc.out_scale[1] - mpc.beta[1] * (mpc.sigmaK[:, 1]).T + mpc.slack_y[1, :] >= (mpc.y[1, :] - mpc.out_bias[1]) / mpc.out_scale[1])

        # q_min <= q <= q_max
        mpc.opti.subject_to((mpc.m_min - mpc.out_bias[2]) / mpc.out_scale[2] + mpc.beta[2] * (mpc.sigmaK[:, 2]).T - mpc.slack_y[2, :] <= (mpc.y[2, :] - mpc.out_bias[2]) / mpc.out_scale[2])
        mpc.opti.subject_to((mpc.m_max - mpc.out_bias[2]) / mpc.out_scale[2] - mpc.beta[2] * (mpc.sigmaK[:, 2]).T + mpc.slack_y[2, :] >= (mpc.y[2, :] - mpc.out_bias[2]) / mpc.out_scale[2])

        mpc.opti.subject_to(mpc.sigmaK == sigma_est)

        for i in range(0, mpc.n_out):
            mpc.opti.subject_to(mpc.slack_y[i, :] >= 1e-8)

            mpc.opti.subject_to(mpc.slack_y[0, :] <= 50 / mpc.out_scale[0])
            mpc.opti.subject_to(mpc.slack_y[1, :] <= 50 / mpc.out_scale[1])
            mpc.opti.subject_to(mpc.slack_y[2, :] <= 50 / mpc.out_scale[2])
            mpc.opti.subject_to(mpc.slack_y[0, :2] <= 5 / mpc.out_scale[0])
            mpc.opti.subject_to(mpc.slack_y[1, :2] <= 1 / mpc.out_scale[1])
            mpc.opti.subject_to(mpc.slack_y[2, :2] <= 3 / mpc.out_scale[2])

            mpc.opti.subject_to(mpc.slack_explo[i, :] >= 1e-6)
            mpc.opti.subject_to(mpc.slack_explo[i, :] <= 1e1)

        if mpc.flag == 1:
            ### Exploration Constraints ###
            mpc.opti.subject_to(mpc.beta[0] * (mpc.sigmaK[:, 0]).T >= mpc.epsilon[0] - mpc.slack_explo[0, :] / mpc.out_scale[0])
            mpc.opti.subject_to(mpc.beta[1] * (mpc.sigmaK[:, 1]).T >= mpc.epsilon[1] - mpc.slack_explo[1, :] / mpc.out_scale[1])
            mpc.opti.subject_to(mpc.beta[2] * (mpc.sigmaK[:, 2]).T >= mpc.epsilon[2] - mpc.slack_explo[2, :] / mpc.out_scale[2])

    def cost_expression(self, mpc, gamma_pred_gas, gamma_pred_el, alpha_slack, gamma_exp, gamma_slack):
        mpc.opti.subject_to(
            mpc.J == gamma_pred_gas @ (1 / 1000 * mpc.Ts / 3600 * mpc.Pb[0, :].T / mpc.COP)
            + gamma_pred_el @ (1 / 1000 * mpc.Ts / 3600 * mpc.Pb[1, :].T / mpc.eff_EB)
            + mpc.gamma * (mpc.y[0, mpc.N - 1] - mpc.T_ref * casadi.MX.ones((1, 1))) ** 2
            + gamma_slack[0] * mpc.slack_y[0, :4] @ mpc.slack_y[0, :4].T
            + gamma_slack[1] * mpc.slack_y[1, :4] @ mpc.slack_y[1, :4].T
            + gamma_slack[1] * mpc.slack_y[2, :4] @ mpc.slack_y[2, :4].T
            + 1e-3 * (
                gamma_slack[0] * mpc.slack_y[0, 4:] @ mpc.slack_y[0, 4:].T
                + gamma_slack[1] * mpc.slack_y[1, 4:] @ mpc.slack_y[1, 4:].T
                + gamma_slack[2] * mpc.slack_y[2, 4:] @ mpc.slack_y[2, 4:].T
            )
            + gamma_exp @ mpc.slack_explo @ np.ones((mpc.N, 1)) * mpc.flag
        )

    def warm_start_extra_variables(self, mpc):
        for i in range(mpc.n_out):
            mpc.opti.set_initial(mpc.sigmaK[:, i], 0.05 * np.ones([mpc.N, 1]))
        mpc.opti.set_initial(mpc.slack_explo, np.zeros([mpc.n_out, mpc.N]))
        mpc.opti.set_initial(mpc.slack_y, np.zeros([mpc.n_out, mpc.N]))

    def extract_solution(self, mpc, sol):
        return sol.value(mpc.slack_y), sol.value(mpc.slack_explo), sol.value(mpc.sigmaK)

    def fallback_slack_explo(self, mpc):
        return np.zeros([mpc.n_out, mpc.N])

    def select_theta(self, mpc, theta, U0, b0):
        return theta

    def bayesian_correction(self, mpc, s, y_star, u):
        """Update the Bayesian last layer using recent data from the measured trajectory.

        This is the mechanism that adapts the model when the controller is
        operating with uncertainty-aware corrections. It mutates mpc.L_prev,
        mpc.Q_prev, mpc.theta and mpc.sigma_bnn for the next control cycle.
        """
        L_prec=mpc.L_prev
        Q_prec=mpc.Q_prev

        ####  bayesian correction ###

        data =(mpc.data)
        layers= data['layers'][0, 0]
        weight=layers['weights']
        weights=weight[0,0]
        U0 = weights['U0'][0][0]
        b0 = weights['b0'][0][0]


        output_scaler=data['output_scaler']
        output_scaler=output_scaler[0,0]
        out_bias=output_scaler['bias']
        out_bias=out_bias[0]
        out_scale=output_scaler['scale']

        out_scale=out_scale[0]
        input_scaler=data['input_scaler']
        input_scaler=input_scaler[0,0]
        inp_bias=input_scaler['bias']
        inp_bias=inp_bias[0]
        inp_scale=input_scaler['scale']
        inp_scale=inp_scale[0]
        xk=np.zeros([s+1,3])
        u_norm=np.zeros([5,s])
        print(s)
        q_eb = mpc.q_eb
        n_out=3
        for i in range(n_out):
            for j in range(s+1):
                xk[j,i]=(y_star[i,j]-out_bias[i])/out_scale[i]
        print(u[:,s-1])
        for i in range(5):
            for j in range(s):

                u_norm[i,j]=(u[i,j]-inp_bias[i])/inp_scale[i]
                if i==4:
                    u_norm[i,j]=(((u[4,j]*(y_star[2,j]-q_eb)+u[5,j]*q_eb)/y_star[2,j])-inp_bias[i])/inp_scale[i]


        ####state vector
        state=np.zeros([24,1])
        k=s-1
        state[0,:] =    xk[k-2,0]
        state[1,:] =    (xk[k-2,1])
        state[2,:] =    (xk[k-2,2])
        state[3,:] =    (u_norm[0,k-2])
        state[4,:] =    (u_norm[1,k-2])
        state[5,:] =    (u_norm[2,k-2])
        state[6,:] =    (u_norm[3,k-2])
        state[7,:] =    u_norm[4,k-2]
        state[8,:] =    xk[k-1,0]
        state[9,:] =    xk[k-1,1]
        state[10,:] =   xk[k-1,2]
        state[11,:] =   (u_norm[0,k-1])
        state[12,:] =   (u_norm[1,k-1])
        state[13,:] =   (u_norm[2,k-1])
        state[14,:] =   (u_norm[3,k-1])
        state[15,:] =   (u_norm[4,k-1])
        state[16,:] =   xk[k,0]
        state[17,:] =   xk[k,1]
        state[18,:] =   xk[k,2]
        state[19,:] =   (u_norm[0,k])
        state[20,:] =   (u_norm[1,k])
        state[21,:] =   (u_norm[2,k])
        state[22,:] =   (u_norm[3,k])
        state[23,:] =   u_norm[4,k]

        ### nnarx
        W1 = weights['W.0'][0][0]
        U1 = weights['U.0'][0][0]
        b1 = weights['b.0'][0][0]

        W2 = weights['W.1'][0][0]

        U2 = weights['U.1'][0][0]

        b2 = weights['b.1'][0][0]


        n_neurons=9
        z1=np.matmul(np.transpose(U1[:,:]),state)
        print(z1.shape)
        z2=np.matmul(np.transpose(W1[:,:]),u_norm[:,s-1]).reshape([9,1])
        z3=(b1[:,:].reshape([n_neurons,1]))
        z=z1+z2+z3
        e1=np.tanh(z)

        z1=np.matmul(np.transpose(U2[:,:]),e1)

        z2=np.matmul(np.transpose(W2[:,:]),u_norm[:,s-1]).reshape([9,1])

        z3=(b2[:,:].reshape([n_neurons,1]))

        z_l2=z1+z2+z3
        etaL=np.tanh(z_l2)
        print(etaL.shape)



        f_tilde= np.vstack([etaL, 1]) #10x1
        a=L_prec@f_tilde
        denom = 1 + (f_tilde.T @ L_prec @ f_tilde).item()

        Lambda_inv=L_prec-(1/denom)*(a@a.T)#10x10
        yk_star=(xk[s,:]).reshape([3,1])
        Q=f_tilde@yk_star.T+Q_prec #10x3
        theta_mean= Lambda_inv@Q #10x3
        mpc.Q_prev=Q
        mpc.L_prev=Lambda_inv
        mpc.theta=theta_mean
        mpc.sigma_bnn=np.sqrt((1+f_tilde.T@L_prec@f_tilde)*mpc.sigma_epsilon2)

        return theta_mean
