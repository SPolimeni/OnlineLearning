import gpytorch
import torch
from sklearn.preprocessing import MinMaxScaler

import casadi
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from pyfmi import load_fmu
import time

def ensure2D(x: np.ndarray):
    if x.ndim == 1:
        return np.expand_dims(x, axis=1)
    return x

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(length_scale=1.0,length_scale_bounds=(1e-3, 1e3),ard_num_dims=27))# specification of the lenght of the kernel -> taken exponential kernel

    def forward(self, x): # give in the return the K matrix
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) # compute the matrxi using the kernel exponential
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def np2torch(x):
    return torch.from_numpy(np.asarray(x)).float()


def mpc_controller(t,N,u_prec,y_prec,J_prec,Pb_prec,T_C0,xk,uk,T_s,gp_buffer,y_gp,y_rnn,theta):
    start_time=time.time()
    data = loadmat('C:\\Users\\micky\\Desktop\\Poli\\IILM\\Tesi\\NN_example_CERN\\training_output\\Munters\\NNARX_9-9_H3_bs20_Ts150_Ns150_20251107_082445\\net.mat')
    Potenze = ensure2D(np.genfromtxt('ssnet\\Datasets\\DHN_LOAD\\DHN_ground_truth.csv', delimiter=',', usecols=(3,4,5,6), skip_header=28))
    Potenze=np.transpose(Potenze)
    c_el=ensure2D(np.genfromtxt('C:\\Users\\micky\\Desktop\\Poli\\IILM\\Tesi\\NN_example_CERN\\ssnet\\MPC\\20250501_20250501_MGP_PrezziZonali_Nord.csv', delimiter=';', usecols=(2), skip_header=2))
    Tr_max=66
    Tr_min=50
    Ts_max=80
    Ts_min=65
    if k==T_C0:
        r=-1
    m_max=20
    m_min=4
    Pb_max=147e3
    Pb_min=60e3
    
    Pb_max_eb=50e3
    Pb_min_eb=30e3
    
    N=N #prediction horizon
    opti = casadi.Opti() #solver
    Nb=3
    n_uopt=int(N/Nb)
    n_slack=1
    n_inp=4 + 2
    n_out=3
    u = opti.variable(n_inp, n_uopt)
    y = opti.variable(n_out, N)
    m_min_out=0
    s = opti.variable(n_slack, 1)
    J  = opti.variable(1, 1)
    Pb  = opti.variable(2, N) 
    end=n_uopt
    q_eb=3/3.6
    
    

    if t<T_C0:
        u_opt = u_prec[:,0]
        u_pred = u_prec
        y_pred = y_prec
        y_nnarx=y_prec
        Pb_pred=Pb_prec
        slack= np.zeros([n_slack, 1])
        J_pred = J_prec
        t_solver = 0


    else:
        
        
        # Ts_min - slack <= Ts <= Ts_max  
        opti.subject_to(Ts_min- s[0,:] <= y[0,:])
        opti.subject_to(Ts_max + s[0,:] >= y[0,:]) 
       
      
        # Tr_min <= Tr <= Tr_max
        opti.subject_to(45<= y[1,:])
        opti.subject_to(66 >= y[1,:]) 

        opti.subject_to(2<= y[2,:])
        opti.subject_to(10 >= y[2,:]) 



        # temperature gas boiler ed eletric boiler 
        opti.subject_to(Ts_min <= u[5,:])
        opti.subject_to(75>= u[5,:])
        opti.subject_to(Ts_min<= u[4,:])
        opti.subject_to(Ts_max >= u[4,:])

        #Disturbance prediction 
        if t>num_steps-4:
            opti.subject_to(Potenze[:,t-2]+3000== u[0:4, 0])
            opti.subject_to(Potenze[:,t-2+3]+1000== u[0:4, 1])
            opti.subject_to(Potenze[:,t-2+6]+1000== u[0:4, 2])
            opti.subject_to(Potenze[:,t-2+9]+1000== u[0:4, 3])
        else:
            opti.subject_to(Potenze[:,t]+3000== u[0:4, 0])
            opti.subject_to(Potenze[:,t+3]+1000== u[0:4, 1])
            opti.subject_to(Potenze[:,t+6]+1000== u[0:4, 2])
            opti.subject_to(Potenze[:,t+9]+1000== u[0:4, 3])
    
        y_N,gp_buffer,y_gp,output_rnn= model_rnn(t,data, u, y_prec,N,u_prec,xk,uk,y,Nb,gp_buffer,y_gp,y_rnn,theta)

        opti.subject_to(y == output_rnn[:,0:N])

        opti.subject_to(s[:,0] >=0)

        #Δu_min < Δu < Δu_max
        delta = 5
        val_prec = float(u_prec[5, 0])                       
        val_correnti = u[5, 0:end-1]                          

        vect = casadi.horzcat(val_prec, val_correnti)
        opti.subject_to(-delta <= u[5,:] - vect)
        opti.subject_to( delta >= u[5,:] - vect)

        # #Δu_min < Δu < Δu_max
        delta = 5
        val_prec = float(u_prec[4, 0])                       
        val_correnti = u[4, 0:end-1]                          

        vect = casadi.horzcat(val_prec, val_correnti)
        opti.subject_to(-delta <= u[4,:] - vect)
        opti.subject_to( delta >= u[4,:] - vect)

        cp=4186
        #Constraint on the power of gas boiler
        for j in range(0,N):
            ii = min(np.floor(j/Nb), N/Nb-1)
        
            if j==0:
                opti.subject_to( cp * casadi.minus(xk[2,t],q_eb*casadi.DM.ones(1,1))* (u[4,j]-xk[1, t]) == Pb[0,0])
                opti.subject_to( cp * q_eb*casadi.DM.ones(1,1)* (u[5,0]-xk[1, t]) == Pb[1,0])

            else:
                opti.subject_to( cp * casadi.minus(y[2,j-1],q_eb*casadi.DM.ones(1,1))* (u[4,ii]-y[1, j-1]) == Pb[0,j])
                opti.subject_to( cp * q_eb*casadi.DM.ones(1,1)* (u[5,ii]-y[1,j-1]) == Pb[1,j])

        opti.subject_to(Pb_min <= Pb[0,:])
        opti.subject_to(Pb_max >= Pb[0,:])   

        
        opti.subject_to(Pb_min_eb<= Pb[1,:])
        opti.subject_to(Pb_max_eb >= Pb[1,:])  

                        
        gamma = 0.2
        T_ref = 75
        c_gas=0.034
        COP = 0.8
        gamma_pred_gas =c_gas*casadi.DM.ones(1,N)
        
        r=int(np.floor(t/3)) 
        gamma_pred_el =c_el[r,0]/1000*casadi.DM.ones(1,N) 
        print(c_el[r,0])
        alpha_slack = 0.01*casadi.MX.ones((1, n_slack))
        opti.subject_to(J == 1*(gamma_pred_gas @(1/1000*T_s/3600 *( Pb[0,:].T)/COP)+gamma_pred_el @ (1/1000*T_s/3600 * Pb[1,:].T/1) +gamma*(y[0,N-1] -T_ref*casadi.MX.ones((1, 1)))**2+alpha_slack@s))
       
        for i in range(n_inp) :
            initial_guess = np.concatenate([u_prec[i, 1:n_uopt], [u_prec[i, -1]]])
            opti.set_initial(u[i,:n_uopt], initial_guess)

        for i in range(n_out):
            y_initial = np.concatenate([ y_prec[i, 1:],[y_prec[i, -1]]])
            opti.set_initial(y[i,:],  y_initial)
        opti.set_initial(s, np.zeros([n_slack,1]))  
        for i in range(2):
            Pb_initial = np.concatenate([Pb_prec[ i,1:], [Pb_prec[ i,-1]]])
            opti.set_initial(Pb[i,:], Pb_initial)    
        opti.set_initial(J, J_prec)

        # Declare the cost function
        opti.minimize(J);                  
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
            'compl_inf_tol': 1e-5       # Desired threshold for the complementarity conditions
        }

        # Set the solver
        opti.solver('ipopt', prob_opts, ip_opts)
      

        # SOLVE THE FHOCP
        try:
            sol = opti.solve()
            print('*** Problem solved ***')
            u_opt = sol.value(u[:, 0])
            u_pred = sol.value(u)
            y_pred = sol.value(y)
            slack = sol.value(s)
            Pb_pred=sol.value(Pb)
            J_pred = sol.value(J)

        except Exception as ex:
            print('*** Problem not solved  - I try again by decreasing alpha slack !!!!!!!!!!!!!!***')
            J2  = opti.variable(1, 1)
            gamma = 0.002
            T_ref = 72
            c_gas=0.034 # MWh
            COP = 0.8
            gamma_pred_gas =c_gas*casadi.DM.ones(1,N)
            r=int(np.floor(k/3)) 
            gamma_pred_el =c_el[r,0]/1000*casadi.DM.ones(1,N)
            alpha_slack = alpha_slack 
            
            opti.subject_to(J2 == 0.01*(gamma_pred_gas @(1/1000*T_s/3600 * Pb[0,:].T/COP)+gamma_pred_el @(1/1000*T_s/3600 *Pb[1,:].T/1) +gamma*(y[0,N-1] -T_ref*casadi.MX.ones((1, 1)))**2))
            
            opti.set_initial(J2, J_prec)
            opti.minimize(J2); 
            try:
                sol = opti.solve()
                print('*** Problem solved ***')
                # Extract the optimal control action
                u_opt = sol.value(u[:, 0])
                u_pred = sol.value(u)
                y_pred = sol.value(y)
                slack = 0
                Pb_pred=sol.value(Pb)
                J_pred = sol.value(J2)
       
            except Exception as ex2:
                print('Second solve failed:', ex2)

    end_time=time.time()
    computation_time=end_time-start_time
    return  u_opt,J_pred,y_pred,u_pred,Pb_pred,slack,gp_buffer,y_gp,computation_time



#activation function: tanh and Non linearity computation
def activation_fun(input_vector, weights,state_vector,theta):

    U0 = weights['U0'][0][0]
    b0 = weights['b0'][0][0]
    U0=U0
    b0=b0

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
    f_tilde=casadi.vertcat(e_vect,1)
    eta=theta.T@f_tilde

    return eta


# Simulation of the dynamics
def simulate_dynamics(initial_state, input_vector, A, Bu, Bx, C, steps,weights,n_states,theta,gt):
    n_out=3
    states = casadi.MX.zeros(n_states)
    outputs = casadi.MX.zeros(( n_out,N))
    # Stato iniziale
    states[:] =initial_state.reshape(24) #k=0
    eta=gt
    # Simulazione temporale
    for k in range(0, steps):
        s1 = casadi.reshape(np.matmul(A, states[:]),(24,1))
        
        s2 = casadi.reshape(np.matmul(Bu, input_vector[k,:].T),(24,1)) # all'istante successivo questo sarà lìinout precedente
       
        s3 =casadi.reshape( np.matmul(Bx, eta),(24,1))
        
        states[:] = s1 + s2 + s3
        eta=activation_fun(input_vector[k,:],weights,states[:],theta)
        outputs[:,k] = eta 
        
    return states, outputs

def model_rnn(k,data,u,y_pred,N,u_prec,xk,uk,y,Nb,gp_buffer,y_gp,y_rnn,theta): 
    horizon=3
    n_out=3
    n_inp=5
    q_eb=3/3.6
    n_states=horizon*(n_out+n_inp)
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

    layers=data['layers']
    layers= layers[0, 0]  
    
    weight=layers['weights']
    weights=weight[0,0]

    ### Normalization ###
    input_scaler=data['input_scaler']
    input_scaler=input_scaler[0,0]
    inp_bias=input_scaler['bias']
    inp_bias=inp_bias[0]
    inp_scale=input_scaler['scale']
    inp_scale=inp_scale[0]
    
    U = casadi.MX.zeros(( n_inp,N))

    u_rnn = np.zeros([ n_inp,num_steps])
    output_scaler=data['output_scaler']
    output_scaler=output_scaler[0,0]
    out_bias=output_scaler['bias']
    out_bias=out_bias[0]
    out_scale=output_scaler['scale']

    out_scale=out_scale[0]
    
    
    for i in range(n_inp):
        for j in range(N):
            ii=min(np.floor(j/Nb),N/Nb-1)
            U[i,j]=(u[i,ii]-inp_bias[i]*casadi.MX.ones((1)))/inp_scale[i]
            if i==n_inp-1:
                
                if j==0:
                    U[i,j]= (((u[5,j]*q_eb+( xk[2,k] -q_eb*casadi.DM.ones(1))*u[4,j])/xk[2,k]) -inp_bias[i])/inp_scale[i]
                else:
                    U[i,j]= (((u[5,ii]*q_eb+( y[2,j-1] -q_eb*casadi.DM.ones(1))*u[4,ii])/y[2,j-1]) -inp_bias[i])/inp_scale[i]

    for i in range(n_inp):
        for j in range(k):
            
            u_rnn[i,j]=(uk[i,j]-inp_bias[i]*casadi.MX.ones((1)))/inp_scale[i]
            if i==n_inp-1:
                u_rnn[i,j]= (((uk[5,j]*q_eb+( xk[2,j] -q_eb*casadi.DM.ones(1))*uk[4,j])/xk[2,j]) -inp_bias[i])/inp_scale[i]
    
    #initial state 
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
    
    U=U.T

    Y_pred=casadi.MX(n_out,N)
    Y_RNN=casadi.MX(n_out,N)
    ys=np.zeros([num_steps,n_out])
    ground_truth=np.zeros([num_steps,n_out])
 
    ### Denormalization ###
    for i in range(n_out):
        for j in range(k+1):
            ys[j,i]=(y_rnn[j,i]-out_bias[i])/out_scale[i]
            ground_truth[j,i]=(xk[i,j]-out_bias[i])/out_scale[i]
    states,outputs=simulate_dynamics(initial_state, U[:,:], A, Bu, Bx, C, N,weights,n_states,theta,ground_truth[k,:])

    for i in range(n_out):
        
        for j in range(N):
           
            Y_RNN[i,j]=out_scale[i] * outputs[i,j]+out_bias[i]*np.ones([1, 1]) # * per prodotto scalare
    return Y_pred,gp_buffer,y_gp,Y_RNN

def bayesian_correction(s,y_star,eta_L,L_prec,Q_prec,u):
####  bayesian correction ###
    sigma_epsilon=0.01*np.ones([3,1])

    data = loadmat('C:\\Users\\micky\\Desktop\\Poli\\IILM\\Tesi\\NN_example_CERN\\training_output\\Munters\\NNARX_9-9_H3_bs20_Ts150_Ns150_20251107_082445\\net.mat')
    layers= data['layers'][0, 0]  
    weight=layers['weights']
    weights=weight[0,0]
    U0 = weights['U0'][0][0]
    b0 = weights['b0'][0][0]
    
    U0=np.array(U0)
    b0=np.array(b0)
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
    for i in range(n_out): 
        for j in range(s+1):
            xk[j,i]=(y_star[i,j]-out_bias[i])/out_scale[i]
    for i in range(5): 
        for j in range(s):

            u_norm[i,j]=(u[i,j]-inp_bias[i])/inp_scale[i]
            if i==4:
                u_norm[i,j]=(((u[4,j]*(y_star[2,j]-q_eb)+u[5,j]*q_eb)/y_star[2,j])-inp_bias[i])/inp_scale[i]
    
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

    if s==T_C0:
        theta=np.vstack([U0,b0])
        Q_prec=L_prec@theta


    f_tilde= np.vstack([etaL, 1]) #10x1
    a=L_prec@f_tilde
    denom = 1 + (f_tilde.T @ L_prec @ f_tilde).item()
    
    Lambda_inv=L_prec-(1/denom)*(a@a.T)#10x10
    yk_star=(xk[s,:]).reshape([3,1])
    Q=f_tilde@yk_star.T+Q_prec #10x3
    theta_mean= Lambda_inv@Q #10x3
    sigma=(1+f_tilde.T@L_prev@f_tilde)*sigma_epsilon 

    
    return theta_mean,Q, Lambda_inv,sigma

   

if __name__ == "__main__":
    r=0
    ore_tot=5
    total_time=ore_tot*3600
    time_step=300
    num_steps = total_time // time_step +1
    print(num_steps)
    
    time_points = np.arange(0, total_time + 1, time_step)[:num_steps]
    N=12
    n_inp=5+1
    n_out=3
    T_C0=4
    y_prec=np.zeros([n_out,N])
    u_prec=np.zeros([n_inp,N])
    Pb_prec=np.zeros([2,N])
    Pb_pred=np.zeros([2,num_steps])

    y=np.zeros([n_out,num_steps])
    s=np.zeros([1,num_steps])

    y_rnn=np.zeros([num_steps+N,n_out])
    y_pred=np.zeros([n_out,num_steps])
    u=np.zeros([n_inp,num_steps])
    q_eb=3/3.6
    u_prec[0,:]=-34576.0789961249
    u_prec[1,:]=-33060.9479618137
    u_prec[2,:]=-32360.2405314816
    u_prec[3,:]=-33257.7579101869
   
    u_prec[4,:]=72.8387963838879
    u_prec[5,:]=69.0
    
    y_prec[0,:]=70.2690912858482
    y_prec[1,:]=60.786245068179
    y_prec[2,:]=3
    cp=4186
    T_net=(u_prec[5,:]*q_eb+( y_prec[2,:]-q_eb*np.ones([1,N]))*u_prec[4,:])/y_prec[2,:]
    print(T_net)
    Pb_prec[0,:]=cp * (y_prec[2,:]-q_eb*np.ones([1,N]))* (u_prec[4,:]-y_prec[1, :])
    Pb_prec[1,:]=cp * q_eb*np.ones([1,N])* (u_prec[5,:]-y_prec[1, :])
    
    J_prec=0

    model=load_fmu('C:\\Users\\micky\\Desktop\\Poli\\IILM\\Tesi\\simulatore\\DHN_MC_V3_FMU_EB_60.fmu')

    opts = model.simulate_options()
    opts['CVode_options']['rtol'] = 1e-4
    opts['CVode_options']['atol'] = 1e-4 
    
    y[:,0:5]=y_prec[:,0:5]
    y_pred[:,0]=y_prec[:,0]
    y_rnn[0]=y_prec[:,0]

    ### memoria gaussinan process###
    gp_buffer={}
    gp_buffer[ "y_buffer_1"]=[]
    gp_buffer[ "y_buffer_2"]=[]
    gp_buffer[ "y_buffer_3"]=[]

    gp_buffer[ "X_buffer_1"]=[]
    gp_buffer[ "X_buffer_2"]=[]
    gp_buffer[ "X_buffer_3"]=[]
    y_gp=np.empty((num_steps+N, n_out), dtype=object)
    
    final_cost=np.zeros([num_steps])
    computation_time=np.zeros([num_steps])
    Q_prev=np.zeros([10,3]) #10x3 
    L_prev=np.eye(10) #10x10 matrice identità inziale
    
    theta_prev=L_prev@Q_prev#10x3 zero mean iniziale 
    phi_prev=np.zeros([9,1])
    sigma=np.zeros([n_out,num_steps])
    for k in range(0,num_steps-1):
        
        print(k)
        if k>3:
            theta_prev,Q_prev,L_prev,sigma_k=bayesian_correction(k,y,phi_prev,L_prev,Q_prev,u)
            sigma[:,k]=sigma_k.reshape(3)
            
        u_control,J_prec,y_prec,u_prec,Pb_prec,slack,gp_buffer,y_gp,computation_time[k]=mpc_controller(k,N,u_prec,y_prec,J_prec,Pb_prec,T_C0,y,u,time_step,gp_buffer,y_gp,y_rnn,theta_prev)

        
        u[:,k]=u_control
        
        print(u_prec[4:6,:])
        if k<T_C0:

            s[:,k]=slack[:,0]
        else:
            s[:,k]=slack
        y_pred[:,k+1]=y_prec[:,0]

        print(y_pred[:,k+1])
        print(y_prec)
        print(slack)
        print(y_rnn[k+1,:])
        Pb_pred[0,k]=Pb_prec[0,0]
        Pb_pred[1,k]=Pb_prec[1,0]
        print(Pb_pred[:,k])
        
        if k>3:
            Time = np.arange(0*900, 300*(k)+1, 300)
            
            input_array=np.column_stack([Time,u[0,:k+1],u[1,:k+1],u[2,:k+1],u[3,:k+1],u[4,:k+1],u[5,:k+1]])
            model.set(['Ts_L4','T_return','mf_supply'],y[:,0])
            res=model.simulate(start_time=300*T_C0, final_time=300.0*(k+1),options=opts,input=(['PowerLoad1','PowerLoad2','PowerLoad3','PowerLoad4','Tref_GB','Tref_EB'],input_array))
       
            y[0,k+1]=res['Ts_L4'][-1]
            y[1,k+1]=res['T_return'][-1]+1
            y[2,k+1]=res['mf_supply'][-1]
            
            print(y[0,k+1])
            print(y[1,k+1])
            print(y[2,k+1])
            model.reset() 
            c_gas=0.034 # MWh
            #c_el=120
            COP = 0.8
            gamma_pred_gas =c_gas
            r=int(np.floor(k/3)) 
            c_el=ensure2D(np.genfromtxt('C:\\Users\\micky\\Desktop\\Poli\\IILM\\Tesi\\NN_example_CERN\\ssnet\\MPC\\20250501_20250501_MGP_PrezziZonali_Nord.csv', delimiter=';', usecols=(2), skip_header=2))
            gamma_pred_el =c_el[r,0]/1000
            T_s=300

            Pb_pred[0,k]=cp * (y[2,k]-q_eb*np.ones([1,1]))* (u[4,k]-y[1, k])
            Pb_pred[1,k]=cp * q_eb*np.ones([1,1])* (u[5,k]-y[1, k])
            final_cost[k] =gamma_pred_gas * (1/1000*T_s/3600 * Pb_pred[0,k]/COP)+gamma_pred_el * (1/1000*T_s/3600 * Pb_pred[1,k]/1)
        
            
            print(final_cost[k])
            print(J_prec)
            T_net=(u[5,k]*q_eb+( y[2,k]-q_eb*np.ones([1,N]))*u[4,k])/y[2,k]
            print(T_net)
    
    Time = np.arange(0*300, 300*(k)+1, 300)
    
    input_array=np.column_stack([Time,u[0,:k+1],u[1,:k+1],u[2,:k+1],u[3,:k+1],u[4,:k+1],u[5,:k+1]])
    model.set(['Ts_L4','T_return','mf_return'],y[:,0])
    
    res=model.simulate(start_time=0, final_time=300.0*(k+1),options=opts,input=(['PowerLoad1','PowerLoad2','PowerLoad3','PowerLoad4','Tref_GB','Tref_EB'],input_array))
    
    y[0,k+1]=res['Ts_L4'][-1]
    
    y[1,k+1]=res['T_return'][-1]
    y[2,k+1]=res['mf_supply'][-1]

    model.reset() 

    plt.plot( res['time'],res['T_gb'])
    plt.xlabel('Tempo')
    plt.ylabel('y')
    plt.show()

    ##################### PLOTTTTTT ##########################
     
    Tr_max=66
    Tr_min=45
    Ts_max=80
    Ts_min=65
    Ts_max_eb=75

    #Ts_max=80
    m_max=50
    m_min=2
    Pb_max=1e6
    Pb_min=1e3
    
    Pb_max_eb=1e6
    Pb_min_eb=1e3

    plt.plot(1/3600*time_points[4:num_steps-1], sigma[0, 4:num_steps-1], 'r', label='output simulatore')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout(pad=2.0)
    plt.show()
    for i in range(n_out):
        plt.subplot(n_out, 1, i+1)
        if i==0:
            label_y_hat=f'Temperature Supply Load4'
        if i==1:
            label_y_hat=f'Temperature Return Line'
        if i==2:
            label_y_hat=f'Mass Flow Return Line'
       
        
        plt.plot(1/3600*time_points, y[i, :], 'r', label='output simulatore')
        plt.plot(1/3600*time_points, y_pred[i,:], 'g-', label='output gp')
        #plt.plot(1/3600*time_points, y_rnn[:num_steps,i], 'b-', label='output RNN')

        plt.title(label_y_hat, fontsize=8)
        plt.xlabel('Time [h]', fontsize=8)
        if i==0 or i==1:
            plt.ylabel('Temperature [°C]')
        else:
            plt.ylabel('Mass flow [kg/s]')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout(pad=2.0)
    plt.show()
    for i in range(n_out):
        plt.figure(figsize=(10, 7.5))
        # Scegli label per il titolo
        if i == 0:
            label_y_hat = 'Temperature Supply Load4'
            ylabel = 'Temperature [°C]'
            max=Ts_max
            min=Ts_min
        elif i == 1:
            label_y_hat = 'Temperature Return Line'
            ylabel = 'Temperature [°C]'
            max=Tr_max
            min=Tr_min
        else:
            label_y_hat = 'Mass Flow Return Line'
            ylabel = 'Mass flow [kg/s]'
            max=m_max
            min=m_min
            
        plt.plot(1/3600*time_points, y[i,:], linewidth=1.8, label='Ground Truth')
        plt.plot(1/3600*time_points, max*np.ones( time_points.shape), 'k-', label='output gp')
        plt.plot(1/3600*time_points, min*np.ones( time_points.shape), 'k-', label='output RNN')

        #plt.title(label_y_hat, fontsize=16)
        plt.xlabel('Time [h]', fontsize=36,fontname= 'Times New Roman')
        plt.ylabel(ylabel, fontsize=36,fontname= 'Times New Roman')
        plt.grid(False)
        plt.tick_params(axis='both', labelsize=28)
        #plt.legend(fontsize=12)
        plt.tight_layout(pad=2.0)
        plt.show()
    
    plt.figure(figsize=(10, 7.5))
    plt.plot(1/3600*time_points[:num_steps-1], u[4,:num_steps-1])
    plt.plot(1/3600*time_points[:num_steps-1], Ts_max*np.ones(Pb_pred[1,:num_steps-1].shape),'k-')
    plt.plot(1/3600*time_points[:num_steps-1], Ts_min*np.ones(Pb_pred[1,:num_steps-1].shape),'k-')
    plt.xlabel('Time [h]',fontsize=36,fontname= 'Times New Roman')
    plt.ylabel('Temperature [°C]',fontsize=36,fontname= 'Times New Roman')
    plt.grid(False)

    plt.tick_params(axis='both', labelsize=28)   
    plt.tight_layout(pad=2.0) 
    plt.show()


    potenza_totale=u[0, :num_steps-1]+u[1, :num_steps-1]+u[2, :num_steps-1]+u[3, :num_steps-1]

    plt.plot(1/3600*time_points[:num_steps-1], u[0, :num_steps-1], 'b-')
    plt.plot(1/3600*time_points[:num_steps-1], u[1, :num_steps-1], 'r-')
    plt.plot(1/3600*time_points[:num_steps-1], u[2, :num_steps-1], 'g-')
    plt.plot(1/3600*time_points[:num_steps-1], u[3, :num_steps-1], 'm-')

    #plt.title(' Power Demand', fontsize=8)
    plt.xlabel('Time [h]', fontsize=20)

    plt.ylabel('Power [kW]', fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=18)
    
    plt.legend()

    plt.tight_layout(pad=2.0)
    plt.show()

    plt.plot(1/3600*time_points[:num_steps-1], potenza_totale, 'k-')
    #plt.plot(1/3600*time_points, y_pred[i, :], 'g-', label='output RNN')
    #plt.title('Potenza reale', fontsize=8)
    plt.xlabel('Time [h]', fontsize=8)

    plt.ylabel('Power [kW]')

    plt.grid(True)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend()

    plt.tight_layout(pad=2.0)
    plt.show()

    ######## CALCOLO DEL COSTO FINALE #######
    costo=0
    for i in range(num_steps-1):
        costo=costo+final_cost[i]
    print(costo)
    plt.plot(1/3600*time_points[:num_steps-1], final_cost[:num_steps-1], 'k-', label='Potenza reale')
    plt.xlabel('Time [h]', fontsize=8)

    plt.ylabel('cost[$/kWh]')

    plt.grid(True)
    plt.legend()

    plt.tight_layout(pad=2.0)
    plt.show()

    media_tempo_computazione=np.mean(computation_time[4:num_steps-1])
    print(media_tempo_computazione)




