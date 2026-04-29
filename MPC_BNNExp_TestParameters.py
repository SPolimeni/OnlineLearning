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

############## Define parameters ##############
data = loadmat('NNmodels/net_1020.mat') #ALB loadmat('net2.mat')
Potenze = ensure2D(np.genfromtxt('DHN_ground_truth.csv', delimiter=',', usecols=(3,4,5,6), skip_header=28)) +4000 
Potenze=np.transpose(Potenze)
c_el=ensure2D(np.genfromtxt('20250501_20250501_MGP_PrezziZonali_Nord.csv', delimiter=';', usecols=(2), skip_header=2))     
# Scaler
output_scaler=data['output_scaler']
output_scaler=output_scaler[0,0]
out_scale=output_scaler['scale']
out_scale=out_scale[0]
out_bias=output_scaler['bias']
out_bias=out_bias[0]
input_scaler=data['input_scaler']
input_scaler=input_scaler[0,0]
inp_scale=input_scaler['scale']
inp_scale=inp_scale[0]
inp_bias=input_scaler['bias']
inp_bias=inp_bias[0]
n_neurons=9
# Parameters
layers= data['layers'][0, 0]  
weight=layers['weights']
weights=weight[0,0]
U0 = weights['U0'][0][0]
b0 = weights['b0'][0][0]

# Simulation
ore_tot=4
total_time=ore_tot*3600
time_step=300
num_steps = total_time // time_step +1

# MPC
T_s=300
N = 12
Nb = 3
n_uopt=int(N/Nb)
gamma = 0.2 # TEST PAR #0.5 # 0.002 ALB aumento altrimenti la temperatura crolla in basso (non è importante, è tuning)
T_ref = 70.5 # TEST PAR 
T_C0 = 4
r = 0

#beta0 = 30      #ALB: visto che aumento L_prev cambio qui 
#beta = np.array([beta0/out_scale[0], beta0/out_scale[1], beta0/out_scale[2]]) 
                
#ALB2 provo a mettere beta diversi per uscita
beta = np.array([35/out_scale[0], 15/out_scale[1], 15/out_scale[2]]) 

# epsilon0= 0.05*out_scale[0]  #ALB: visto che aumento L_prev cambio qui 
# epsilon = np.array([epsilon0/out_scale[0], epsilon0/out_scale[1], epsilon0/out_scale[2]]) #ALB normalizzo epsilon rispetto a out_scale così vincolo exploration rimane consistente, uso lo stesso epsilon per tutti

#ALB2 metto epsilon proporzionali a beta visto che sono diversi per uscita
epsilon = 0.4/beta[0]*beta


slack_tol = 1e-3
delta_u = 5
sigma_epsilon2 = 0.001 #0.005 ALB: visto che aumento L_prev cambio qui

# System
n_inp = 6  # TEST PAR
n_out = 3  # TEST PAR
Ts_max= 80 # TEST PAR
Ts_min= 67 # TEST PAR #ALB2 PROVA
Tr_max= 66 # TEST PAR
Tr_min= 45 # TEST PAR
q_max = 10 # TEST PAR
q_min = 2.0 # TEST PAR
Ts_max_eb = 72 # TEST PAR ALB2 preso dai file test Simone
Ts_min_eb = 65  #TEST PAR
Ts_max_gb = 80 #TEST PAR
Ts_min_gb = 65 #TEST PAR
Pb_max_gb = 147e3 #TEST PAR
Pb_min_gb = 60e3 #TEST PAR
Pb_max_eb = 38e3 #TEST PAR
Pb_min_eb = 22e3 #TEST PAR  
cp=4186 #TEST PAR
q_eb=3/3.6 #TEST PAR
c_gas=0.034 # $/kWh #TEST PAR
COP = 0.8 #TEST PAR
eff_EB = 0.8 #TEST PAR


def mpc_controller(t,N,u_prec,y_prec,J_prec,Pb_prec,T_C0,xk,uk,T_s,gp_buffer,y_gp,y_rnn,theta,L_prev,flag, theta0): #ALB2 aggiungo theta0 come input a MPC per simulare rete originale
    start_time=time.time()

    if t == T_C0:
        r=-1
    
    opti = casadi.Opti() 
    n_slack=1
    u = opti.variable(n_inp, n_uopt)
    y = opti.variable(n_out, N)
    y0 = opti.variable(n_out, N) # ALB2 aggiungo output calcolato con pesi originali
    slack_explo = opti.variable(n_out, N) #opti.variable(n_out, 1) #ALB estendo per N
    sigmaK = opti.variable(N, n_out)
    J  = opti.variable(1, 1)
    Pb  = opti.variable(2, N) 
    end = n_uopt 
    slack_y = opti.variable(n_out, N)
    
    if t < T_C0:
        u_opt = u_prec[:,0]
        u_pred = u_prec
        y_pred = y_prec
        y_pred0 = y_prec# ALB2 aggiungo output calcolato con pesi originali
        Pb_pred=Pb_prec
        slack= np.zeros([n_slack, 1])
        J_pred = J_prec
        sigma_pred=0.05
        slack_explo_pred= np.zeros([n_out, N]) #np.zeros([n_out, 1]) ALB estendo per N

    else:
        # Ts_min <= Ts <= Ts_max
        opti.subject_to((Ts_min - out_bias[0])/out_scale[0] + beta[0]*(sigmaK[:,0]).T - slack_y[0,:] <= (y[0,:] - out_bias[0])/out_scale[0])
        opti.subject_to((Ts_max - out_bias[0])/out_scale[0] - beta[0]*(sigmaK[:,0]).T + slack_y[0,:] >= (y[0,:] - out_bias[0])/out_scale[0]) 
      
        # Tr_min <= Tr <= Tr_max
        opti.subject_to((Tr_min - out_bias[1])/out_scale[1] + beta[1]*(sigmaK[:,1]).T - slack_y[1,:] <= (y[1,:] - out_bias[1])/out_scale[1])
        opti.subject_to((Tr_max - out_bias[1])/out_scale[1] - beta[1]*(sigmaK[:,1]).T + slack_y[1,:] >= (y[1,:] - out_bias[1])/out_scale[1]) 

        # q_min <= q <= q_max
        opti.subject_to((q_min - out_bias[2])/out_scale[2] + beta[2]*(sigmaK[:,2]).T - slack_y[2,:] <= (y[2,:] - out_bias[2])/out_scale[2])
        opti.subject_to((q_max - out_bias[2])/out_scale[2] - beta[2]*(sigmaK[:,2]).T + slack_y[2,:] >= (y[2,:] - out_bias[2])/out_scale[2]) 

        # Temperature eletric boiler and gas boiler
        opti.subject_to(Ts_min_eb <= u[5,:])
        opti.subject_to(Ts_max_eb >= u[5,:])
        opti.subject_to(Ts_min_gb <= u[4,:])
        opti.subject_to(Ts_max_gb >= u[4,:])

        # Slack
        for i in range(0,n_out):
            opti.subject_to(slack_y[i,:] >= 1e-8)
            opti.subject_to(slack_y[i,:] <= 50) #ALB: aiuta l'ottimizzatore, considera normalizzato
            opti.subject_to(slack_explo[i,:] >= 1e-6)
            opti.subject_to(slack_explo[i,:] <= 1e1) #ALB: aiuta l'ottimizzatore

        # Δu_min < Δu < Δu_max
        val_prec = float(u_prec[5, 0])                       
        val_correnti = u[5, 0:end-1]                          
        vect = casadi.horzcat(val_prec, val_correnti)
        opti.subject_to(-delta_u <= u[5,:] - vect)
        opti.subject_to( delta_u >= u[5,:] - vect)

        val_prec = float(u_prec[4, 0])                       
        val_correnti = u[4, 0:end-1]                          
        vect = casadi.horzcat(val_prec, val_correnti)
        opti.subject_to(-delta_u <= u[4,:] - vect)
        opti.subject_to( delta_u >= u[4,:] - vect)

        # Disturbance prediction: impongo che gli input delle potenze siano uguali a quelle programmate
        
        # MC: modificato shift potenza rispetto a file simulazione originale
        if t > num_steps-4:
            opti.subject_to((Potenze[:,t-2] ) == u[0:4, 0]) 
            opti.subject_to((Potenze[:,t-2+3]+1000) == u[0:4, 1])
            opti.subject_to((Potenze[:,t-2+6]+1000) == u[0:4, 2])
            opti.subject_to((Potenze[:,t-2+9]+1000) == u[0:4, 3])
        else:
            opti.subject_to((Potenze[:,t]) == u[0:4, 0])
            opti.subject_to((Potenze[:,t+3]+1000) == u[0:4, 1])
            opti.subject_to((Potenze[:,t+6]+1000) == u[0:4, 2])
            opti.subject_to((Potenze[:,t+9]+1000) == u[0:4, 3])
        

        # System dynamics constraint
        y_N,gp_buffer,y_gp,output_rnn,sigma_est = model_rnn(t,data, u, y_prec,N,u_prec,xk,uk,y,Nb,gp_buffer,y_gp,y_rnn,theta,L_prev)
        opti.subject_to(y == output_rnn[:,0:N])
        
        #ALB2 simulo rete con pesi originali
        y_N0,gp_buffer0,y_gp0,output_rnn0,sigma_est0 = model_rnn(t,data, u, y_prec,N,u_prec,xk,uk,y,Nb,gp_buffer,y_gp,y_rnn,theta0,L_prev)
        opti.subject_to(y0 == output_rnn0[:,0:N])
        
        # Constraint on the power 
        for j in range(0,N):
            ii = min(np.floor(j/Nb), N/Nb-1)
            if j==0:
                opti.subject_to( cp * casadi.minus(xk[2,t],q_eb*casadi.DM.ones(1,1))* (u[4,j]-xk[1, t]) == Pb[0,0])
                opti.subject_to( cp * q_eb*casadi.DM.ones(1,1)* (u[5,0]-xk[1, t]) == Pb[1,0])
            else:
                opti.subject_to( cp * casadi.minus(y[2,j-1],q_eb*casadi.DM.ones(1,1))* (u[4,ii]-y[1, j-1]) == Pb[0,j])
                opti.subject_to( cp * q_eb*casadi.DM.ones(1,1)* (u[5,ii]-y[1,j-1]) == Pb[1,j])

        opti.subject_to(Pb_min_gb <= Pb[0,:])
        opti.subject_to(Pb_max_gb >= Pb[0,:])   

        opti.subject_to(Pb_min_eb <= Pb[1,:])
        opti.subject_to(Pb_max_eb >= Pb[1,:]) 

        opti.subject_to(sigmaK==sigma_est) 
         
        if flag == 1: 
            ### Exploration Constraints ###
            #ALB: per come normalizzato beta e epsilon devo normalizzare anche la slack per averle uguali
            opti.subject_to(beta[0]*(sigmaK[:,0]).T>=epsilon[0]-slack_explo[0,:]/out_scale[0])  
            opti.subject_to(beta[1]*(sigmaK[:,1]).T>=epsilon[1]-slack_explo[1,:]/out_scale[1])  
            opti.subject_to(beta[2]*(sigmaK[:,2]).T>=epsilon[2]-slack_explo[2,:]/out_scale[2])  
       
        # Cost function
        gamma_pred_gas =c_gas*casadi.DM.ones(1,N)
        r=int(np.floor(t/3)) 
        gamma_pred_el = np.zeros((N,1))
        for i in range(N):
            idx = int(np.floor((t+i) * len(c_el) / num_steps))
            idx = min(idx, len(c_el)-1)
            gamma_pred_el[i] = c_el[idx]/1000 #c_el[idx] ALB come si vede dai file plot e dai codici test, i prezzi devono essere /1000, così ottimizzazione sembra più sensata
        gamma_pred_el = gamma_pred_el.T

        gamma_exp=5*np.ones((1, n_out))  #5*np.ones((1, n_out)) ALB2 abbasso slack exploration

        gamma_slack = 1e2*np.array([out_scale[0], out_scale[1], out_scale[2]]) #gamma_slack = 10  ALB dichiarato come vettore per metterlo diverso per variabile
        
        #opti.subject_to(J == gamma_pred_gas@(1/1000*T_s/3600 *( Pb[0,:].T)/COP)+ gamma_pred_el @ (1/1000*T_s/3600 * Pb[1,:].T) + gamma*(y[0,N-1] -T_ref*casadi.MX.ones((1, 1)))**2 + gamma_exp@slack_explo*flag + gamma_slack*casadi.sum1(casadi.sum2(slack_y)))
       
        #ALB cambiata funzione di costo per considerare gamma_slack diversa per ogni slack e slack lungo l'orizzonte
        #opti.subject_to(J == gamma_pred_gas@(1/1000*T_s/3600 *( Pb[0,:].T)/COP)+ gamma_pred_el @ (1/1000*T_s/3600 * Pb[1,:].T) + gamma*(y[0,N-1] -T_ref*casadi.MX.ones((1, 1)))**2 + gamma_exp@slack_explo@np.ones((N, 1))*flag + gamma_slack[0]*casadi.sum1(slack_y[0,:]) + gamma_slack[1]*casadi.sum1(slack_y[1,:]) + gamma_slack[2]*casadi.sum1(slack_y[2,:]))

        #ALB2 metto l'efficienza dell'EB nella funzione di costo
        opti.subject_to(J == gamma_pred_gas@(1/1000*T_s/3600 *( Pb[0,:].T)/COP)+ gamma_pred_el @ (1/1000*T_s/3600 * Pb[1,:].T/eff_EB) + gamma*(y[0,N-1] -T_ref*casadi.MX.ones((1, 1)))**2 + gamma_exp@slack_explo@np.ones((N, 1))*flag + gamma_slack[0]*casadi.sum1(slack_y[0,:]) + gamma_slack[1]*casadi.sum1(slack_y[1,:]) + gamma_slack[2]*casadi.sum1(slack_y[2,:]))

        # Set initial values
        for i in range(n_inp) :
            opti.set_initial(u[i,:n_uopt], np.concatenate([u_prec[i, 1:n_uopt], [u_prec[i, -1]]]))
        for i in range(n_out):
            opti.set_initial(y[i,:],  np.concatenate([ y_prec[i, 1:],[y_prec[i, -1]]]))
        for i in range(2):
            opti.set_initial(Pb[i,:], np.concatenate([Pb_prec[ i,1:], [Pb_prec[ i,-1]]]))    
        for i in range(n_out):
            opti.set_initial(sigmaK[:,i], 0.05*np.ones([N,1]))
        opti.set_initial(J, J_prec)
        opti.set_initial(slack_explo, np.zeros([n_out,N])) # ALB: vale la pena inizializzare al passo prima?
        opti.set_initial(slack_y, np.zeros([n_out,N]))  # ALB: vale la pena inizializzare al passo prima?


        try: #ALB2 abbasso tolleranze se diventa infeasible, dovremmo tracciare quando succede 

            # Declare the cost function
            opti.minimize(J);                  
            prob_opts = {'expand': True, 'ipopt': {'print_level': 0,}, 'print_time': False}

            # IPOPT settings
            #ip_opts = {'print_level': 0, 'max_iter': int(1e4), 'compl_inf_tol': 1e-5}       #ALB: Aggiungo opzioni a ipopt che rendono il solutore più robusto 
            ip_opts = {'print_level': 0, 'max_iter': int(1e4),'mu_strategy': "adaptive" }  
            
            # Set the solver
            opti.solver('ipopt', prob_opts, ip_opts)
        
        # SOLVE THE FHOCP
        #try:
            sol = opti.solve()
            print('*** Problem solved ***')
            u_opt = sol.value(u[:, 0])
            u_pred = sol.value(u)
            y_pred = sol.value(y)
            y_pred0 = sol.value(y0) #ALB2 aggiungo output con pesi originali
            slack = 0
            Pb_pred=sol.value(Pb)
            J_pred = sol.value(J)
            sigma_pred=sol.value(sigmaK)
            slack_explo_pred=sol.value(slack_explo)

        except Exception as ex: #ALB2 riprovo abbassando tolleranze, dovremmo plottare quando succede
            try:
                print('*** Problem not solved - *** **** REDUCE TOLLERANCES **** ***')

                opti.minimize(0.01*J);                  
                prob_opts = {'expand': True, 'ipopt': {'print_level': 0,}, 'print_time': False}

                ip_opts = {'print_level': 0, 'max_iter': int(1e4), 'compl_inf_tol': 1e-4, 'acceptable_tol': 1e-4,'acceptable_iter': 10,'mu_strategy': "adaptive" }  
                
                # Set the solver
                opti.solver('ipopt', prob_opts, ip_opts)
            
                sol = opti.solve()
                print('*** Problem solved ***')
                u_opt = sol.value(u[:, 0])
                u_pred = sol.value(u)
                y_pred = sol.value(y)
                y_pred0 = sol.value(y0) #ALB2 aggiungo output con pesi originali
                slack = 0
                Pb_pred=sol.value(Pb)
                J_pred = sol.value(J)
                sigma_pred=sol.value(sigmaK)
                slack_explo_pred=sol.value(slack_explo)

            except Exception as ex:
                print('*** Problem not solved ***')
            
    end_time=time.time()
    computation_time=end_time-start_time
    return  u_opt,J_pred,y_pred,u_pred,Pb_pred,slack,gp_buffer,y_gp,computation_time,sigma_pred,slack_explo_pred, y_pred0 #ALB2 aggiunto y_pred0 calcolato con i pesi originali


#activation function: tanh and Non linearity computation
def activation_fun(input_vector, weights,state_vector,theta,L_prev):

    #weights dei due hidden layers    
    W1 = weights['W.0'][0][0]
    U1 = weights['U.0'][0][0]
    b1 = weights['b.0'][0][0]
    W2 = weights['W.1'][0][0]   
    U2 = weights['U.1'][0][0]    
    b2 = weights['b.1'][0][0]    
   
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
    sigma=np.sqrt((1+f_tilde.T@L_prev@f_tilde) * sigma_epsilon2)
   
    return eta, sigma


# Simulation of the dynamics
def simulate_dynamics(initial_state, input_vector, A, Bu, Bx, C, steps,weights,n_states,theta,gt,L_prev):
    states = casadi.MX.zeros(n_states)
    outputs = casadi.MX.zeros((n_out,N))
    sigma_k = casadi.MX.zeros((N,n_out))
    
    # Stato iniziale
    states[:] =initial_state.reshape(24) #k=0
    eta=gt
    # Simulazione temporale
    for k in range(0, steps):
        s1 = casadi.reshape(np.matmul(A, states[:]),(24,1))        
        s2 = casadi.reshape(np.matmul(Bu, input_vector[k,:].T),(24,1)) # all'istante successivo questo sarà lìinout precedente
        s3 =casadi.reshape(np.matmul(Bx, eta),(24,1))
        states[:] = s1 + s2 + s3
        eta, sigma=activation_fun(input_vector[k,:],weights,states[:],theta,L_prev)
        # Calcoliamo l'output
        outputs[:,k] = eta #all'istante succesisvo
        sigma_k[k,:]=sigma
        
    return states, outputs, sigma_k

def model_rnn(k,data,u,y_pred,N,u_prec,xk,uk,y,Nb,gp_buffer,y_gp,y_rnn,theta,L_prev): 
    horizon=3
    n_inp=5
    n_states=horizon*(n_out+n_inp)
    #State space model of NNarx: matrices definitions
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

    #rnn layers and weight
    layers=data['layers']
    layers= layers[0, 0]  
    weight=layers['weights']
    weights=weight[0,0]
    
    U = casadi.MX.zeros(( n_inp,N))
    u_rnn = np.zeros([ n_inp,num_steps])    
    
    for i in range(n_inp):
        for j in range(N):
            ii=min(np.floor(j/Nb),N/Nb-1)
            U[i,j]=(u[i,ii]-inp_bias[i]*casadi.MX.ones((1)))/inp_scale[i]
            if i==n_inp-1:
                if j==0:
                    U[i,j]= (((u[5,j]*q_eb+( xk[2,k] -q_eb*casadi.DM.ones(1))*u[4,j])/casadi.fmax(xk[2,k], 1e-6)) -inp_bias[i])/inp_scale[i]
                else:
                    U[i,j]= (((u[5,ii]*q_eb+( y[2,j-1] -q_eb*casadi.DM.ones(1))*u[4,ii])/casadi.fmax(y[2,j-1], 1e-6)) -inp_bias[i])/inp_scale[i]

    for i in range(n_inp):
        for j in range(k):
            u_rnn[i,j]=(uk[i,j]-inp_bias[i]*casadi.MX.ones((1)))/inp_scale[i]
            if i==n_inp-1:
                u_rnn[i,j]= (((uk[5,j]*q_eb+( xk[2,j] -q_eb*casadi.DM.ones(1))*uk[4,j])/xk[2,j]) -inp_bias[i])/inp_scale[i]

    #initial state 
    initial_state=np.zeros([n_states,1])  
    initial_state[0,:] = (xk[0,k-3]-out_bias[0])/out_scale[0]
    initial_state[1,:] = (xk[1,k-3]-out_bias[1])/out_scale[1]
    initial_state[2,:] = (xk[2,k-3]-out_bias[2])/out_scale[2]
    initial_state[3,:] = (uk[0,k-3]-inp_bias[0])/inp_scale[0]
    initial_state[4,:] = (uk[1,k-3]-inp_bias[1])/inp_scale[1]
    initial_state[5,:] = (uk[2,k-3]-inp_bias[2])/inp_scale[2]
    initial_state[6,:] = (uk[3,k-3]-inp_bias[3])/inp_scale[3]
    initial_state[7,:] = (((uk[5,k-3]*q_eb+( xk[2,k-3] -q_eb*np.ones([1,1]))*uk[4,k-3])/xk[2,k-3])-inp_bias[4])/inp_scale[4]
    initial_state[8,:] = (xk[0,k-2]-out_bias[0])/out_scale[0]
    initial_state[9,:] = (xk[1,k-2]-out_bias[1])/out_scale[1]
    initial_state[10,:] = (xk[2,k-2]-out_bias[2])/out_scale[2]
    initial_state[11,:] = (uk[0,k-2]-inp_bias[0])/inp_scale[0]
    initial_state[12,:] = (uk[1,k-2]-inp_bias[1])/inp_scale[1]
    initial_state[13,:] = (uk[2,k-2]-inp_bias[2])/inp_scale[2]
    initial_state[14,:] = (uk[3,k-2]-inp_bias[3])/inp_scale[3]
    initial_state[15,:] = (((uk[5,k-2]*q_eb+( xk[2,k-2] -q_eb*np.ones([1,1]))*uk[4,k-2])/xk[2,k-2])-inp_bias[4])/inp_scale[4]
    initial_state[16,:] = (xk[0,k-1]-out_bias[0])/out_scale[0]
    initial_state[17,:] = (xk[1,k-1]-out_bias[1])/out_scale[1]
    initial_state[18,:] = (xk[2,k-1]-out_bias[2])/out_scale[2]
    initial_state[19,:] = (uk[0,k-1]-inp_bias[0])/inp_scale[0]
    initial_state[20,:] = (uk[1,k-1]-inp_bias[1])/inp_scale[1]
    initial_state[21,:] = (uk[2,k-1]-inp_bias[2])/inp_scale[2]
    initial_state[22,:] = (uk[3,k-1]-inp_bias[3])/inp_scale[3]
    initial_state[23,:] = (((uk[5,k-1]*q_eb+( xk[2,k-1] -q_eb*np.ones([1,1]))*uk[4,k-1])/xk[2,k-1])-inp_bias[4])/inp_scale[4]
    
    U=U.T  
    
    Y_pred=casadi.MX.zeros(n_out,N)
    Y_RNN=casadi.MX.zeros(n_out,N)
    ys=np.zeros([num_steps,n_out])
    ground_truth=np.zeros([num_steps,n_out])
   
    for i in range(n_out):
        for j in range(k+1):
            ys[j,i]=(y_rnn[j,i]-out_bias[i])/out_scale[i]
            ground_truth[j,i]=(xk[i,j]-out_bias[i])/out_scale[i]
    #The outputs is a horizontal vectors, so the orws represents the output variables  the column are the instances 
    states,outputs,sigma_k=simulate_dynamics(initial_state, U[:,:], A, Bu, Bx, C, N,weights,n_states,theta,ground_truth[k,:],L_prev)

    for i in range(n_out):
        for j in range(N):
            Y_RNN[i,j]=out_scale[i] * outputs[i,j]+out_bias[i]*np.ones([1, 1]) # * per prodotto scalare

    return Y_pred,gp_buffer,y_gp,Y_RNN,sigma_k

def bayesian_correction(s,y_star,eta_L,L_prec,Q_prec,u):
    
    xk=np.zeros([s+1,3])
    u_norm=np.zeros([5,s])
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
    state[0,:] = xk[k-2,0]
    state[1,:] = (xk[k-2,1])
    state[2,:] = (xk[k-2,2])
    state[3,:] = (u_norm[0,k-2])
    state[4,:] = (u_norm[1,k-2])
    state[5,:] = (u_norm[2,k-2])
    state[6,:] = (u_norm[3,k-2])
    state[7,:] = u_norm[4,k-2]
    state[8,:] = xk[k-1,0]
    state[9,:] = xk[k-1,1]
    state[10,:] = xk[k-1,2]
    state[11,:] = (u_norm[0,k-1])
    state[12,:] = (u_norm[1,k-1])
    state[13,:] = (u_norm[2,k-1])
    state[14,:] = (u_norm[3,k-1])
    state[15,:] = (u_norm[4,k-1])
    state[16,:] = xk[k,0]
    state[17,:] = xk[k,1]
    state[18,:] = xk[k,2]
    state[19,:] = (u_norm[0,k])
    state[20,:] = (u_norm[1,k])
    state[21,:] = (u_norm[2,k])
    state[22,:] = (u_norm[3,k])
    state[23,:] = u_norm[4,k]

    ### nnarx
    W1 = weights['W.0'][0][0]
    U1 = weights['U.0'][0][0]
    b1 = weights['b.0'][0][0]
    W2 = weights['W.1'][0][0]
    U2 = weights['U.1'][0][0]
    b2 = weights['b.1'][0][0]
   
    n_neurons=9
    z1=np.matmul(np.transpose(U1[:,:]),state)
    z2=np.matmul(np.transpose(W1[:,:]),u_norm[:,s-1]).reshape([9,1])
    z3=(b1[:,:].reshape([n_neurons,1]))
    z=z1+z2+z3
    e1=np.tanh(z)
    z1=np.matmul(np.transpose(U2[:,:]),e1)
    z2=np.matmul(np.transpose(W2[:,:]),u_norm[:,s-1]).reshape([9,1])
    z3=(b2[:,:].reshape([n_neurons,1]))
    z_l2=z1+z2+z3
    etaL=np.tanh(z_l2)

    # if s==T_C0:
    #     theta=np.vstack([U0,b0])
    #     Q_prec=L_prec@theta

    f_tilde= np.vstack([etaL, 1]) #10x1
    a=L_prec@f_tilde
    denom = 1 + (f_tilde.T @ L_prec @ f_tilde).item()    
    Lambda_inv=L_prec-(1/denom)*(a@a.T)#10x10
    yk_star=(xk[s,:]).reshape([3,1])
    Q=f_tilde@yk_star.T+Q_prec #10x3
    theta_mean= Lambda_inv@Q #10x3
    sigma=np.sqrt((1+f_tilde.T@L_prec@f_tilde)*sigma_epsilon2) 
    return theta_mean, Q, Lambda_inv, sigma

   

if __name__ == "__main__":
    
    time_points = np.arange(0, total_time + 1, time_step)[:num_steps]
    y_prec=np.zeros([n_out,N])
    u_prec=np.zeros([n_inp,N])
    Pb_prec=np.zeros([2,N])
    Pb_pred=np.zeros([2,num_steps])
    y=np.zeros([n_out,num_steps])
    s=np.zeros([1,num_steps])
    y_rnn=np.zeros([num_steps+N,n_out])
    y_pred=np.zeros([n_out,num_steps])
    y_pred0=np.zeros([n_out,num_steps]) #ALB2 aggiungo y calcolata con i pesi originali
    u=np.zeros([n_inp,num_steps])
   
    # prendo i valori che misuro
    # u_prec[0,:]=-34576.0789961249 #ALB2 rimetto quelli di Michela
    # u_prec[1,:]=-33060.9479618137
    # u_prec[2,:]=-32360.2405314816
    # u_prec[3,:]=-33257.7579101869
    # u_prec[4,:]=72.8387963838879
    # u_prec[5,:]=70.0

    # ####### COLPEVOLE #ALB2 LI RIMETTO
    u_prec[0,:]=-32000
    u_prec[1,:]=-32000
    u_prec[2,:]=-32000
    u_prec[3,:]=-32000
    u_prec[4,:]=70.5 
    u_prec[5,:]=70.0
    
    y_prec[0,:]=70.2690912858482
    y_prec[1,:]=60.786245068179
    y_prec[2,:]=2.5

    T_net=(u_prec[5,:]*q_eb+( y_prec[2,:]-q_eb*np.ones([1,N]))*u_prec[4,:])/y_prec[2,:]
    Pb_prec[0,:]=cp * (y_prec[2,:]-q_eb*np.ones([1,N]))* (u_prec[4,:]-y_prec[1, :])
    Pb_prec[1,:]=cp * q_eb*np.ones([1,N])* (u_prec[5,:]-y_prec[1, :])
    J_prec=0

    model=load_fmu('FMUs/DHN_MC_V3_FMU_EB_60.fmu')
    opts = model.simulate_options()
    opts['CVode_options']['rtol'] = 1e-4
    opts['CVode_options']['atol'] = 1e-4
    
    y[:,0:5]=y_prec[:,0:5]
    y_pred[:,0]=y_prec[:,0]
    y_pred0[:,0]=y_prec[:,0] #ALB2 aggiunto per simulare la rete con i pesi originali

    y_rnn[0]=y_prec[:,0]


    ### memoria gaussian process###
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

    # Initialization
    # Q_prev=np.zeros([10,3]) #10x3 
    # L_prev=10e-2*np.ones([10,10]) 
    # theta_prev=L_prev@Q_prev#10x3 

    L_prev = 100 * np.eye(10)  # 5*np.eye(10) #ALB aumento per partire da incertezza alta, se aumento troppo diventa infeasible (?)
    theta0 = np.vstack([U0, b0])
    Q_prev = np.linalg.inv(L_prev) @ theta0
    theta_prev = theta0.copy()
    
    phi_prev=np.zeros([9,1])
    sigma=np.zeros([n_out,num_steps])
    exploration=np.zeros([1,num_steps])
    theta_BNN=np.zeros([num_steps,10,3]) #MC
    theta_BNN[0:4,:,:]=theta0 #MC

    for k in range(0,num_steps-1):
        
        print(k)
        if k>3:
            theta_prev,Q_prev,L_prev,sigma_k=bayesian_correction(k,y,phi_prev,L_prev,Q_prev,u)
            theta_BNN[k,:,:]=(theta_prev[:]) #MC


        flag=1
        u_control,J_prec,y_prec,u_prec,Pb_prec,slack,gp_buffer,y_gp,computation_time[k],sigma_pred,slack_explo, y_prec0=mpc_controller(k,N,u_prec,y_prec,J_prec,Pb_prec,T_C0,y,u,time_step,gp_buffer,y_gp,y_rnn,theta_prev,L_prev,flag, theta0) #ALB2 ho aggiunto theta0 tra gli argomenti di mpc_controller e y_prec0 tra le uscite per simulare la rete con i pesi originali  (si può fare di meglio)
        print("Sigma:")
        print(sigma_pred)
        print("Slack exploration:")
        print(slack_explo)


        if np.any(slack_explo <= slack_tol):
            print("exploration ON")

        else:
            flag = 0
            print(" exploration OFF")
            u_control,J_prec,y_prec,u_prec,Pb_prec,slack,gp_buffer,y_gp,computation_time[k],sigma_pred,slack_explo, y_prec0=mpc_controller(k,N,u_prec,y_prec,J_prec,Pb_prec,T_C0,y,u,time_step,gp_buffer,y_gp,y_rnn,theta_prev,L_prev,flag, theta0) #ALB2 ho aggiunto theta0 tra gli argomenti di mpc_controller e y_prec0 tra le uscite per simulare la rete con i pesi originali (si può fare di meglio)

        
        exploration[0,k] = flag

        u[:,k]=u_control

        if k<T_C0:
            s[:,k]=slack[:,0]
        else:
            #sigma[:,k]=sigma_k.reshape(3)
            sigma[:,k]=sigma_pred[0,:].reshape(3)
            s[:,k]=slack

        y_pred[:,k+1]=y_prec[:,0]
        y_pred0[:,k+1]=y_prec0[:,0] #ALB2
        Pb_pred[0,k]=Pb_prec[0,0]
        Pb_pred[1,k]=Pb_prec[1,0]
        
        if k>3:
            Time = np.arange(0*900, 300*(k)+1, 300)            
            input_array=np.column_stack([Time,u[0,:k+1],u[1,:k+1],u[2,:k+1],u[3,:k+1],u[4,:k+1],u[5,:k+1]])
            model.set(['Ts_L4','T_return','mf_supply'],y[:,0])
            #simulation of the system
            res=model.simulate(start_time=300*T_C0, final_time=300.0*(k+1),options=opts,input=(['PowerLoad1','PowerLoad2','PowerLoad3','PowerLoad4','Tref_GB','Tref_EB'],input_array))
            y[0,k+1]=res['Ts_L4'][-1]
            y[1,k+1]=res['T_return'][-1]
            y[2,k+1]=res['mf_supply'][-1]
            model.reset() 
            gamma_pred_gas = c_gas
            r=int(np.floor(k/3))
            gamma_pred_el =c_el[r,0]/1000
            final_cost[k] =gamma_pred_gas * (1/1000*T_s/3600 * Pb_pred[0,k]/COP)+gamma_pred_el * (1/1000*T_s/3600 * Pb_pred[1,k]/eff_EB) #ALB2 aggiunta efficienza e-boiler
            T_net=(u[5,k]*q_eb+( y[2,k]-q_eb*np.ones([1,N]))*u[4,k])/y[2,k]

    Time = np.arange(0*300, 300*(k)+1, 300)
    
    input_array=np.column_stack([Time,u[0,:k+1],u[1,:k+1],u[2,:k+1],u[3,:k+1],u[4,:k+1],u[5,:k+1]])
    model.set(['Ts_L4','T_return','mf_return'],y[:,0])
    #simulation of the system    
    res=model.simulate(start_time=0, final_time=300.0*(k+1),options=opts,input=(['PowerLoad1','PowerLoad2','PowerLoad3','PowerLoad4','Tref_GB','Tref_EB'],input_array))
    y[0,k+1]=res['Ts_L4'][-1]
    y[1,k+1]=res['T_return'][-1]
    y[2,k+1]=res['mf_supply'][-1]

    model.reset() 

    ########################################## PLOT ###############################################

    ##################### UNCERTAINTY #####################
    #ALB cambio qui plottando un grafico per ogni uscita essendo beta epsilon ora vettori
    plt.plot(1/3600*time_points[4:num_steps-1], beta[0]*sigma[0, 4:num_steps-1], linewidth=4)
    plt.plot(1/3600*time_points[4:num_steps-1], epsilon[0] * np.ones(sigma[0, 4:num_steps-1].shape[0]), color='red', linestyle='--', linewidth=4)
    expl = exploration.flatten()
    current_state = expl[0]
    start_idx = 0
    t = time_points / 3600
    for i in range(1, num_steps):
        if expl[i] != current_state:
            plt.axvspan(t[start_idx], t[i], facecolor='lightblue' if current_state == 0 else 'lightcoral', alpha=0.3)
            start_idx = i
            current_state = expl[i]
    plt.axvspan(t[start_idx], t[num_steps-1], facecolor='lightblue' if current_state == 0 else 'lightcoral', alpha=0.3)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'$w$', fontsize=30)
    plt.title(r'beta[0]*sigma[0] ', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1/3600*time_points[4:num_steps-1], beta[1]*sigma[1, 4:num_steps-1], linewidth=4)
    plt.plot(1/3600*time_points[4:num_steps-1], epsilon[1] * np.ones(sigma[0, 4:num_steps-1].shape[0]), color='red', linestyle='--', linewidth=4)
    expl = exploration.flatten()
    current_state = expl[0]
    start_idx = 0
    t = time_points / 3600
    for i in range(1, num_steps):
        if expl[i] != current_state:
            plt.axvspan(t[start_idx], t[i], facecolor='lightblue' if current_state == 0 else 'lightcoral', alpha=0.3)
            start_idx = i
            current_state = expl[i]
    plt.axvspan(t[start_idx], t[num_steps-1], facecolor='lightblue' if current_state == 0 else 'lightcoral', alpha=0.3)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'$w$', fontsize=30)
    plt.title(r'beta[1]*sigma[1] ', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1/3600*time_points[4:num_steps-1], beta[2]*sigma[2, 4:num_steps-1], linewidth=4)
    plt.plot(1/3600*time_points[4:num_steps-1], epsilon[2] * np.ones(sigma[0, 4:num_steps-1].shape[0]), color='red', linestyle='--', linewidth=4)
    expl = exploration.flatten()
    current_state = expl[0]
    start_idx = 0
    t = time_points / 3600
    for i in range(1, num_steps):
        if expl[i] != current_state:
            plt.axvspan(t[start_idx], t[i], facecolor='lightblue' if current_state == 0 else 'lightcoral', alpha=0.3)
            start_idx = i
            current_state = expl[i]
    plt.axvspan(t[start_idx], t[num_steps-1], facecolor='lightblue' if current_state == 0 else 'lightcoral', alpha=0.3)
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
        price[k] = c_el[r,0]
    plt.figure()
    plt.plot(1/3600*time_points, price/1000, linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel('Electricity price', fontsize=30)
    plt.title('Electricity price profile', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()
    
    ###################### OUTPUTS #####################

    y_BLL_max = ((y_pred-out_bias.reshape(n_out,1))/out_scale.reshape(n_out,1)) + beta.reshape(n_out,1) * sigma  
    y_BLL_max = (y_BLL_max*out_scale.reshape(n_out,1)) + out_bias.reshape(n_out,1)
    y_BLL_min = ((y_pred-out_bias.reshape(n_out,1))/out_scale.reshape(n_out,1)) - beta.reshape(n_out,1) * sigma
    y_BLL_min = (y_BLL_min*out_scale.reshape(n_out,1)) + out_bias.reshape(n_out,1)

    plt.plot(1/3600*time_points, y[0,:], label='Simulator', linewidth=4)
    plt.plot(1/3600*time_points, y_pred[0,:], label='Output BNN', linewidth=4)
    plt.plot(1/3600*time_points, y_pred0[0,:], label='Output Original NNARX', linewidth=2) #ALB2 aggiunto per simulatore output con pesi originali
    plt.fill_between(1/3600*time_points, y_BLL_min[0,:], y_BLL_max[0,:], color='orange', alpha=0.3)
    plt.plot(1/3600*time_points, Ts_min * np.ones(((y_pred[0,:]).shape[0],1)), color='k', linewidth=4)
    plt.plot(1/3600*time_points, Ts_max * np.ones(((y_pred[0,:]).shape[0],1)), color='k', linewidth=4)
    plt.legend()
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Temperature [°C]', fontsize=30)
    plt.title('Load supply temperature', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1/3600*time_points, y[1,:], label='Simulator', linewidth=4)
    plt.plot(1/3600*time_points, y_pred[1,:], label='Output BNN', linewidth=4)
    plt.plot(1/3600*time_points, y_pred0[1,:], label='Output Original NNARX', linewidth=2) #ALB2 aggiunto per simulatore output con pesi originali
    plt.fill_between(1/3600*time_points, y_BLL_min[1,:], y_BLL_max[1,:], color='orange', alpha=0.3)
    plt.plot(1/3600*time_points, Tr_min * np.ones(((y_pred[0,:]).shape[0],1)), color='k', linewidth=4)
    plt.plot(1/3600*time_points, Tr_max * np.ones(((y_pred[0,:]).shape[0],1)), color='k', linewidth=4)
    plt.legend()
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Temperature [°C]', fontsize=30)
    plt.title('Return temperature', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1/3600*time_points, y[2,:], label='Simulator', linewidth=4)
    plt.plot(1/3600*time_points, y_pred[2,:], label='Output BNN', linewidth=4)
    plt.plot(1/3600*time_points, y_pred0[2,:], label='Output Original NNARX', linewidth=2) #ALB2 aggiunto per simulatore output con pesi originali
    plt.fill_between(1/3600*time_points, y_BLL_min[2,:], y_BLL_max[2,:], color='orange', alpha=0.3)
    plt.plot(1/3600*time_points, q_min * np.ones(((y_pred[0,:]).shape[0],1)), color='k', linewidth=4)
    plt.plot(1/3600*time_points, q_max * np.ones(((y_pred[0,:]).shape[0],1)), color='k', linewidth=4)
    plt.legend()
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Mass flow rate [kg/s]', fontsize=30)
    plt.title('Flow rate', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    ###################### INPUTS #####################
    plt.plot(1/3600*time_points[:num_steps-1], u[4,:num_steps-1], linewidth=4)
    plt.plot(1/3600*time_points[:num_steps-1], Ts_max_gb*np.ones(Pb_pred[1,:num_steps-1].shape),'k', linewidth=4)
    plt.plot(1/3600*time_points[:num_steps-1], Ts_min_gb*np.ones(Pb_pred[1,:num_steps-1].shape),'k', linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Temperature [°C]', fontsize=30)
    plt.title('Gas boiler input temperature', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)   
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1/3600*time_points[0:num_steps-1], u[5,:num_steps-1], linewidth=4)
    plt.plot(1/3600*time_points[0:num_steps-1], Ts_max_eb*np.ones(Pb_pred[1,:num_steps-1].shape),'k', linewidth=4)
    plt.plot(1/3600*time_points[0:num_steps-1], Ts_min_eb*np.ones(Pb_pred[1,:num_steps-1].shape),'k', linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Temperature [°C]', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.tight_layout(pad=2.0)
    plt.title('Electric boiler input temperature', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    ###################### POWER #####################
    plt.plot(1/3600*time_points[:num_steps-1], 1/1000*Pb_pred[0,:num_steps-1], linewidth=4)
    plt.plot(1/3600*time_points[:num_steps-1], 1/1000*Pb_min_gb*np.ones(Pb_pred[1,:num_steps-1].shape),'k', linewidth=4)
    plt.plot(1/3600*time_points[:num_steps-1], 1/1000*Pb_max_gb*np.ones(Pb_pred[1,:num_steps-1].shape),'k', linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Power [kW]', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.title('Gas boiler power', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()

    plt.plot(1/3600*time_points[0:num_steps-1], 1/1000*Pb_pred[1,:num_steps-1], linewidth=4)
    plt.plot(1/3600*time_points[0:num_steps-1], 1/1000*Pb_min_eb*np.ones(Pb_pred[1,:num_steps-1].shape),'k', linewidth=4)
    plt.plot(1/3600*time_points[0:num_steps-1], 1/1000*Pb_max_eb*np.ones(Pb_pred[1,:num_steps-1].shape),'k', linewidth=4)
    plt.xlabel(r'Time [h]', fontsize=30)
    plt.ylabel(r'Power [kW]', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.title('Electric boiler power', fontsize=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.xlim((0, ore_tot))
    plt.show()
    for j in range(0,n_out):
        plt.figure(figsize=(10, 7.5))

        for i in range(0, 10):
            plt.plot(1/3600 * time_points[0:num_steps-1], 
                    theta_BNN[0:num_steps-1, i, j], 
                    label=f'Parameter {i}', 
                    linewidth=4)

        plt.legend()
        plt.xlabel('Time (h)')
        plt.ylabel('Theta')
        plt.title(f'All parameters Output {j}')
        plt.tight_layout()
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

    # ######## CALCOLO DEL COSTO FINALE #######
    costo=0
    for i in range(num_steps-1):
        costo=costo+final_cost[i]
    print(costo)

    ##### calcolo del tempo di computazione
    media_tempo_computazione=np.mean(computation_time[:num_steps-1])
    print(media_tempo_computazione)

