import gpytorch
import torch

import casadi
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pyfmi import load_fmu
import time


class MPC:
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    length_scale=1.0,
                    length_scale_bounds=(1e-3, 1e3),
                    ard_num_dims=27
                )
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __init__(self,Param):
        ### from param takes the default parameters
        #Blocking Strategy
        print("IN!!!!")
        self.Nb=Param["Nb"]
    
        self.ore_tot = Param["tot_hours"]
        self.total_time = self.ore_tot * 3600
        self.time_step = 300
        self.num_steps = self.total_time // self.time_step + 1
        self.N=Param["N"]
        self.n_out=Param["n_out"]
        self.n_inp=Param["n_inp"]
        self.u_opt=int(self.N/self.Nb)
        self.n_slack=1
        
        ### import from Param
        self.Ts=Param["Ts"]
        self.q_eb=Param["q_eb"]
        self.Tr_max=Param["Tr_max"]
        self.Tr_min=Param["Tr_min"]
        self.Ts_max=Param["Ts_max"]
        self.Ts_max_eb=Param["Ts_max_eb"]
        self.Ts_min=Param["Ts_min"]
        self.m_max=Param["m_max"]
        self.m_min=Param["m_min"]
        self.Pb_max=Param["Pb_max"]
        self.Pb_min=Param["Pb_min"]
        self.Pb_max_eb=Param["Pb_max_eb"]
        self.Pb_min_eb=Param["Pb_min_eb"]
        self.data=loadmat(Param["nnarx_mat"])
        self.c_el=self.ensure2D(Param["c_el"])
        self.Potenze=Param["Potenza"].T #potenza all'istante k
        self.Model=Param["Model"]
        self.T_C0=Param["T_C0"] #istante in cui inizio ad applicare la legge di controllo
        ### initial value MPC
        self.J_prec=0
        Pb_pred = np.zeros([2, self.N])
        Pb_pred[0,:]=self.Pb_min
        Pb_pred[1,:]=self.Pb_min_eb
        self.Pb_prec=Pb_pred
        y_pred=np.zeros([self.n_out,self.N])
        y_pred[0,:]=70
        y_pred[1,:]=60
        y_pred[2,:]=5
        self.y_prec=y_pred
        u_pred=np.zeros([self.n_inp,self.N])
        u_pred[0:4,:]=-32000
        u_pred[4:6,:]=72
        self.u_prec=u_pred
        ### Buffer Gaussian Process ###
        self.gp_buffer = {
        "y_buffer_1": [],
        "y_buffer_2": [],
        "y_buffer_3": [],
        "X_buffer_1": [],
        "X_buffer_2": [],
        "X_buffer_3": []
        }
        self.y_gp= np.empty((self.num_steps + self.N, self.n_out), dtype=object)

        ### bayesian last layer ###
        layers= self.data['layers'][0, 0]  
        weight=layers['weights']
        weights=weight[0,0]
        U0 = weights['U0'][0][0]
        b0 = weights['b0'][0][0]
        self.theta=np.vstack([U0,b0])
        self.L_prev=np.eye(10) #10x10 matrice identità inziale
        self.Q_prev=self.L_prev@self.theta
        
        

        ### MPC optimization Variables ###
        self.opti = casadi.Opti()
        self.u = self.opti.variable(self.n_inp, self.u_opt)
        self.y = self.opti.variable(self.n_out, self.N)
       # self.y_nnarx = self.opti.variable(self.n_out, self.N)
    
        self.s = self.opti.variable(self.n_slack, 1)
        self.J  = self.opti.variable(1, 1)
        self.Pb  = self.opti.variable(2, self.N)
    
    def ensure2D(self, x: np.ndarray):
        if x.ndim == 1:
            return np.expand_dims(x, axis=1)
        return x

    def np2torch(self, x):
        return torch.from_numpy(np.asarray(x)).float()

    def gaussian_process(self,k,ys,y_mpc,input,u_mpc,ground_truth):
        ### Parameters ###
        h=3
        window_size=36
        n_inp=5
        y_mpc=y_mpc.T
        n_out=3
        N=self.N
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
                x_tensor = self.np2torch(np.stack(X_buffer_1))
                
                y_tensor = self.np2torch(np.array(y_buffer_1) )
            elif j==1:
                x_tensor = self.np2torch(np.stack(X_buffer_2))
                y_tensor = self.np2torch(np.array(y_buffer_2))
            else:
                x_tensor = self.np2torch(np.stack(X_buffer_3))
                y_tensor = self.np2torch(np.array(y_buffer_3))
            
            
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            
            model = MPC.GPRegressionModel(x_tensor, y_tensor, likelihood)
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
                input=casadi.MX.zeros(self.num_steps+N+N,5)
                print(input.shape)
                input[0:k,:]=uk[0:k,:]
                input[k:k+N,:]=u_mpc
                
            
            
            if k==t:
                xk=ys
                ys=casadi.MX.zeros(self.num_steps+N,3)
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
    def SetOptimizationVariables(self):
         ### MPC optimization Variables ###
        self.opti = casadi.Opti()
        self.u = self.opti.variable(self.n_inp, self.u_opt)
        self.y = self.opti.variable(self.n_out, self.N)
        self.y_nnarx = self.opti.variable(self.n_out, self.N)
    
        self.s = self.opti.variable(self.n_slack, 1)
        self.J  = self.opti.variable(1, 1)
        self.Pb  = self.opti.variable(2, self.N)
    def SetConstraints(self):
         ### Constraints ###
            self.opti.subject_to(self.Ts_min- self.s[0,:] <= self.y[0,:])
            self.opti.subject_to(self.Ts_max + self.s[0,:] >= self.y[0,:]) 
            self.opti.subject_to(self.Tr_min <= self.y[1,:])
            self.opti.subject_to(self.Tr_max  >= self.y[1,:])   
            self.opti.subject_to(self.Ts_min <= self.u[5,:])
            self.opti.subject_to(self.Ts_max_eb>= self.u[5,:])
            self.opti.subject_to(self.Ts_min<= self.u[4,:])
            self.opti.subject_to(self.Ts_max >= self.u[4,:])
            self.opti.subject_to(self.Potenze[:,self.t]== self.u[0:4, 0])
            self.opti.subject_to(self.Potenze[:,self.t+3]+1000== self.u[0:4, 1])
            self.opti.subject_to(self.Potenze[:,self.t+6]+1000== self.u[0:4, 2])
            self.opti.subject_to(self.Potenze[:,self.t+9]+1000== self.u[0:4, 3])
            self.opti.subject_to(self.m_min<= self.y[2,:])
            self.opti.subject_to(self.m_max >= self.y[2,:])   
            y_N,output_rnn= self.model_rnn(self.t,self.data, self.u, self.y_prec,self.u_prec,self.xk,self.uk,self.y,self.N/self.Nb,self.y_rnn,self.theta)
            # System dynamics constraint
            if self.Model=='GP':
                self.opti.subject_to(self.y == y_N[:,0:self.N])
                self.opti.subject_to(self.y_nnarx == output_rnn[:,0:self.N])
            else:
                self.opti.subject_to(self.y == output_rnn[:,0:self.N])
                self.opti.subject_to(self.y_nnarx== output_rnn[:,0:self.N])

            self.opti.subject_to(self.s[:,0] >= 0)
            delta = 5
            val_prec = float(self.u_prec[5, 0])                       
            val_correnti = self.u[5, 0:self.u_opt-1]                          

            vect = casadi.horzcat(val_prec, val_correnti)
            self.opti.subject_to(-delta <= self.u[5,:] - vect)
            self.opti.subject_to( delta >= self.u[5,:] - vect)
            
            # # #Δu_min < Δu < Δu_max
            delta = 5
            val_prec = float(self.u_prec[4, 0])                       
            val_correnti = self.u[4, 0:self.u_opt-1]                          

            vect = casadi.horzcat(val_prec, val_correnti)
            self.opti.subject_to(-delta <= self.u[4,:] - vect)
            self.opti.subject_to( delta >= self.u[4,:] - vect)

            cp=4186
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
            self.opti.subject_to(self.Pb_min<= self.Pb[1,:])
            self.opti.subject_to(self.Pb_max_eb >= self.Pb[1,:]) 
    def SetCostFunction(self):
        ### Cost Function  ### 
                
            gamma = 0.2
            T_ref = 70
            c_gas=0.034 # $/kWh
            COP = 0.8
            gamma_pred_gas =c_gas*casadi.DM.ones(1,self.N)
            r=int(np.floor(self.t/3)) 
            gamma_pred_el =self.c_el[r,0]/1000*casadi.DM.ones(1,self.N) #$/KWh
            print(self.c_el[r,0])
            alpha_slack = 0.01*casadi.MX.ones((1, self.n_slack)) 
            self.opti.subject_to(self.J ==1*(gamma_pred_gas @(1/1000*self.Ts/3600 * self.Pb[0,:].T/COP)+gamma_pred_el @ (1/1000*self.Ts/3600 * self.Pb[1,:].T/1)+ gamma*(self.y[0,self.N-1] -T_ref*casadi.MX.ones((1, 1)))**2+alpha_slack@self.s))
        
    def mpc_controller(self, t,  T_C0, xk, uk, y_rnn):
        start_time = time.time()
        data = self.data
        Potenze = self.Potenze
        #Potenze = np.transpose(Potenze)
        c_el = self.c_el
        
        
        N=self.N #prediction horizon
    
    
        
        n_slack = self.n_slack
        n_inp = self.n_inp
        n_out = self.n_out
        u_opt = int(self.N / self.Nb)
        self.xk=xk
        self.uk=uk
        self.t=t
        self.y_rnn=y_rnn
        self.SetOptimizationVariables()
        
            
        if t<T_C0:
            u_opt = self.u_prec[:,0]
            u_pred = self.u_prec
            y_pred = self.y_prec
            y_nnarx=self.y_prec
            Pb_pred=self.Pb_prec
            slack= np.zeros([n_slack, 1])
            J_pred = self.J_prec
            t_solver = 0


        else:
            self.SetConstraints()

            self.SetCostFunction()
            for i in range(n_inp) :
                initial_guess = np.concatenate([self.u_prec[i, 1:u_opt], [self.u_prec[i, -1]]])
                self.opti.set_initial(self.u[i,:], initial_guess)

            for i in range(n_out):
                y_initial = np.concatenate([ self.y_prec[i, 1:N],[self.y_prec[i, -1]]])
                self.opti.set_initial(self.y[i,:],  y_initial)
                self.opti.set_initial(self.y_nnarx[i,:],  y_initial)
            self.opti.set_initial(self.s, np.zeros([n_slack,1]))  
            for i in range(2):
                Pb_initial = np.concatenate([self.Pb_prec[ i,1:], [self.Pb_prec[ i,-1]]])
                self.opti.set_initial(self.Pb[i,:], Pb_initial)    
            self.opti.set_initial(self.J, self.J_prec)

            self.opti.minimize(self.J);                  
            prob_opts = {
                'expand': True,
                'ipopt': {
                    'print_level': 0,     # Disable printing
                },
                'print_time': False       # Do not print the timestamp
            }

            # IPOPT settings
            ip_opts = {
                'print_level': 0,           # Disable printing
                'max_iter': int(1e4),       # Maximum iterations. Use int() to avoid float
                'compl_inf_tol': 1e-5      # Desired threshold for the complementarity conditions
            }

            # Set the solver
            self.opti.solver('ipopt', prob_opts, ip_opts)
            
            # SOLVE THE FHOCP
            try:
                sol = self.opti.solve()
                print('*** Problem solved ***')
                # Extract the optimal control action
                u_opt = sol.value(self.u[:, 0])
                # Other variables
                u_pred = sol.value(self.u)
            
                y_pred = sol.value(self.y)
                slack = sol.value(self.s)
                Pb_pred=sol.value(self.Pb)
                J_pred = sol.value(self.J)
                y_nnarx=sol.value(self.y_nnarx)

            except Exception as ex:
                print('*** Problem not solved  -  Takes last Feasible Solution!***')

                u_opt =self.u_prec[:,0]
                # Other variables
                u_pred = self.u_prec
            
                y_pred = self.y_prec
                slack = np.zeros([n_slack,1])
                Pb_pred=self.Pb_prec
                J_pred = self.J_prec
                y_nnarx=self.y_prec
        
        ### save prev values ###      
        self.J_prec=J_pred
        self.Pb_prec=Pb_pred
        self.y_prec=y_pred
        self.u_prec=u_pred

        end_time=time.time()
        computation_time=end_time-start_time

        return  u_opt,slack,y_nnarx,computation_time



    def simulate_dynamics(self, k, initial_state, input_vector, A, Bu, Bx, C, steps, weights, n_states, ground_truth,theta):
        n_out = self.n_out
        states = casadi.MX.zeros(n_states)
        outputs = casadi.MX.zeros((n_out, self.N))
        states[:] = initial_state.reshape(24)
        eta = ground_truth[k, :].T

        for k_sim in range(0, self.N):

            s1 = casadi.reshape(np.matmul(A, states[:]), (24, 1))
            s2 = casadi.reshape(np.matmul(Bu, input_vector[k_sim, :].T), (24, 1))
            s3 = casadi.reshape(np.matmul(Bx, eta), (24, 1))
            states[:] = s1 + s2 + s3

            eta = self.activation_fun(input_vector[k_sim, :], weights, states[:],theta)
            outputs[:, k_sim] = eta

        return states, outputs

    def model_rnn(self,k,data,u,y_pred,u_prec,xk,uk,y,n_uopt,y_rnn,theta): 

        N=self.N
        horizon=3
        Nb=self.Nb
        n_out=self.n_out
        n_inp=5
        q_eb=self.q_eb
        n_states=horizon*(n_out+n_inp)

        ### State space model of NNarx: matrices definitions ##
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
                        U[i,j]= (((u[5,j]*q_eb+( xk[2,k] -q_eb*casadi.DM.ones(1))*u[4,j])/xk[2,k]) -inp_bias[i])/inp_scale[i]
                    else:
                        U[i,j]= (((u[5,ii]*q_eb+( y[2,j-1] -q_eb*casadi.DM.ones(1))*u[4,ii])/y[2,j-1]) -inp_bias[i])/inp_scale[i]
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
        Y_pred=casadi.MX(n_out,N)
        Y_RNN=casadi.MX(n_out,N)
        ys=np.zeros([self.num_steps,n_out])
        ground_truth=np.zeros([self.num_steps,n_out])
        for i in range(n_out):
            for j in range(k+1):
                ys[j,i]=(y_rnn[j,i]-out_bias[i])/out_scale[i]
                ground_truth[j,i]=(xk[i,j]-out_bias[i])/out_scale[i]
    
        ### NNARX Prediction ###
        states,outputs=self.simulate_dynamics(k,initial_state, U[:,:], A, Bu, Bx, C, N,weights,n_states,ground_truth,theta)

        if self.Model=='NNARX' or self.Model=='BNN':
            ### Denormalization Prediction ###
            for i in range(n_out):
                
                for j in range(N):
                       
                        Y_RNN[i,j]=out_scale[i] * outputs[i,j]+out_bias[i]*np.ones([1, 1]) # * per prodotto scalare

        ### GP Correction ###
        elif self.Model=='GP':
        
        
            output_gp=self.gaussian_process(k,ys,outputs,u_rnn.T,U,ground_truth)
            output_gp=output_gp.transpose()
            
            ### Denormalization Prediction ###
            for i in range(n_out):
                
                for j in range(N):
                        r=(output_gp[i,j])
                        #print(r.dtype)
                        Y_pred[i,j]=out_scale[i] * r+out_bias[i]*np.ones([1, 1])
                        Y_RNN[i,j]=out_scale[i] * outputs[i,j]+out_bias[i]*np.ones([1, 1]) # * per prodotto scalare

                
        return Y_pred,Y_RNN

    def activation_fun(self,input_vector, weights,state_vector,theta):
        #3 layers totoali
        U0 = weights['U0'][0][0]
        b0 = weights['b0'][0][0]
        U0=U0
        b0=b0
        #weights dei due hidden layers
        if self.Model=='NNARX' or self.Model=='GP' :
            theta=casadi.vertcat(U0,b0)
        
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

        # e=np.matmul(U0.transpose(),e_vect)

        # eta = e + np.transpose(b0.reshape(3,))
        
        return eta

    def bayesian_correction(self,s,y_star,u):
        L_prec=self.L_prev
        Q_prec=self.Q_prev
    ####  bayesian correction ###

        data =(self.data)
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
        q_eb=3/3.6
        n_out=3
        #print(y_star[:,s])
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
        state[0,:] =xk[k-2,0]
        state[1,:] = (xk[k-2,1])
        state[2,:] = (xk[k-2,2])
        state[3,:] =(u_norm[0,k-2])
        state[4,:] = (u_norm[1,k-2])
        state[5,:] = (u_norm[2,k-2])
        state[6,:] =(u_norm[3,k-2])
        state[7,:] =u_norm[4,k-2]
        state[8,:] ==xk[k-1,0]
        state[9,:] = xk[k-1,1]
        state[10,:] = xk[k-1,2]
        state[11,:] =(u_norm[0,k-1])
        state[12,:] = (u_norm[1,k-1])
        state[13,:] = (u_norm[2,k-1])
        state[14,:] =(u_norm[3,k-1])
        state[15,:] =(u_norm[4,k-1])
        state[16,:] =xk[k,0]
        state[17,:] = xk[k,1]
        state[18,:] = xk[k,2]
        state[19,:] = (u_norm[0,k])
        state[20,:] =(u_norm[1,k])
        state[21,:] =(u_norm[2,k])
        state[22,:] =(u_norm[3,k])
        state[23,:] =u_norm[4,k]

        ### nnarx
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
        #b=(f_tilde.T@L_prec)
        denom = 1 + (f_tilde.T @ L_prec @ f_tilde).item()
        
        Lambda_inv=L_prec-(1/denom)*(a@a.T)#10x10
        yk_star=(xk[s,:]).reshape([3,1])
        Q=f_tilde@yk_star.T+Q_prec #10x3
        #print((f_tilde@yk_star).shape) #ci deve essere corrispondeza temporale tra yk_star e f_tilde-> f_tilde è dell'istante precedente
        theta_mean= Lambda_inv@Q #10x3
        self.Q_prev=Q
        self.L_prev=Lambda_inv
        self.theta=theta_mean

        #return theta_mean
### to add in main mpc
## nel caso dell bnn va chiamata prima la bayesian neural network e passata theta quella nuova calcolata
### nel caso in cui non si usano le bayesain neural network theta deve essere associata a zero pi dentro la funziona verrà ridefinita
# Param={
#         "Ts":300,
#         "N":12,
#         "q_eb":3/3.6,
#         "Tr_max": 66,
#         "Tr_min": 45,
#         "Ts_max": 80,
#         "Ts_max_eb": 75,
#         "Ts_min":65,
#         "m_max":50,
#         "m_min":2,
#         "Pb_max":1e6,
#         "Pb_min":1e3,
#         "Pb_max_eb":1e6,
#         "Pb_min_eb":1e3,
#         "nnarx_mat": 'C:\\Users\\micky\\Desktop\\Poli\\IILM\\Tesi\\RSETEST\\OnlineLearning\\NNARX_9-9_H3_bs20_Ts300_Ns300_20251020_154518',
#         "c_el": np.genfromtxt('C:\\Users\\micky\\Desktop\\Poli\\IILM\\Tesi\\NN_example_CERN\\ssnet\\MPC\\20250501_20250501_MGP_PrezziZonali_Nord.csv', delimiter=';', usecols=(2), skip_header=2),
#         "Potenza":np.genfromtxt('ssnet\\Datasets\\DHN_LOAD\\DHN_ground_truth.csv', delimiter=',', usecols=(3,4,5,6), skip_header=28),#potenza all'istante k
#         "Model":'GP',
#         "T_C0":4,#istante in cui inizio ad applicare la legge di controllo
#         "L_prev": np.eye(10),
#         "Q_prev": np.zeros([10,3])
#     }