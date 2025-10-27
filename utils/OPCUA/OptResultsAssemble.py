import json
import os
import pickle
import random as random
import shutil
from itertools import cycle

import numpy as np
import pandas as pd
import math
import xlsxwriter
from struct import Struct
import matplotlib as cm
from matplotlib import pyplot as plt
from screeninfo import get_monitors
from pyomo.environ              import value, Var
from sklearn.metrics import mean_squared_error
from math import sqrt

def CollectingResults(model,Opts):
    
    # path            = Opts['path']
    # Results_path    = os.path.join(path, 'Results')

    MESConfig = model.ConfigParams

    nDigits = 2 # Number of decimals for rounding output values
    
    ####################
    # Result dictonary #
    ####################
    results_dict = {}

    # General info from solver (others might be added like if reaching time limit, etc.) - These are valid only for monolithic model
    results_dict['SolverInfo']  = {}
    if Opts['solver'] == 'cplex':
        results_dict['SolverInfo']['Solver_userTime'] = model.results.solver.user_time

    results_dict['SolverInfo']['MIPgap']          = (model.obj() - model.results.problem.lower_bound) / max(1e-10, abs(model.obj()))*100
    results_dict['SolverInfo']['ObjVal']          = model.obj()

    # Collect components of the Objective function
    results_dict['ObjComponents'] = {}
    for n in model.Networks:
        results_dict['ObjComponents'][n] = {}
        results_dict['ObjComponents'][n]['NetCost']     = value(model.NetCost[n])
        results_dict['ObjComponents'][n]['NetRevenue']  = value(model.NetRevenue[n])
    for v in model.Vectors:
        results_dict['ObjComponents'][n]['VectorSlack'] = value(sum(model.VectorSlack[v,t]*1e6 for t in model.t))

    results_dict['ObjComponents'][n]['VirtualCost']     = value(sum(model.VirtualCost[t] for t in model.t))
    if hasattr(model,'PumpConsumption'):
        results_dict['ObjComponents'][n]['PumpCost']    = value(sum(model.PumpCost[t] for t in model.t))
    
    if Opts['ThermalFlows']:
        results_dict['ObjComponents']['VirtualCostTESShIn']     = value(sum(model.VirtualCostTESShIn[t] for t in model.t))
        results_dict['ObjComponents']['VirtualTESShOutCost']    = value(sum(model.VirtualTESShOutCost[t] for t in model.t))

        results_dict['ObjComponents']['QslackHL']       = {}
        results_dict['ObjComponents']['SlackTminHL']    = {}
        for hl in model.HeatLoads:
            results_dict['ObjComponents']['QslackHL'][hl]       = {}
            results_dict['ObjComponents']['SlackTminHL'][hl]    = {}
            results_dict['ObjComponents']['QslackHL'][hl]       = value(sum(model.QslackHL[hl,t]*1e6 for t in model.t))
            # results_dict['ObjComponents']['SlackTminHL'][hl]    = value(sum(model.SlackTminHL[hl,tl,t]*1e5 for tl in model.Tl_hot for t in model.t))

    if hasattr(model,'TESSSlack') and isinstance(getattr(model, 'TESSSlack'), Var):
        results_dict['ObjComponents']['TESSSlack'] = value(sum(sum(model.TESSSlackCost[ts,t] for t in model.t) for ts in model.Storages_H_HT))

    if hasattr(model,'SOCtrackErrorCost') and isinstance(getattr(model, 'SOCtrackErrorCost'), Var):
        results_dict['ObjComponents']['SOCtrackErrorCost'] = {}
        for s in model.Storages: 
            results_dict['ObjComponents']['SOCtrackErrorCost'][s] = value(sum(model.SOCtrackErrorCost[s,t] for t in model.t_st))

    if hasattr(model,'NetImbalanceCost') and isinstance(getattr(model, 'NetImbalanceCost'), Var):
        results_dict['ObjComponents']['NetImbalanceCost'] = value(sum(model.NetImbalanceCost[n,t] for n in model.Networks for t in model.t))
    
    if hasattr(model,'Qdiss') and isinstance(getattr(model, 'Qdiss'), Var):
        results_dict['ObjComponents']['Qdiss'] = {}
        for g in model.Generators_H_HT:
            results_dict['ObjComponents']['Qdiss'][g] = value(sum(model.Qdiss[g,t]*10e3 for t in model.t))
    
    if hasattr(model,'SlackPowerTracking') and isinstance(getattr(model, 'SlackPowerTracking'), Var):
        results_dict['ObjComponents']['SlackPowerTracking'] = {}
        for ts in model.Storages_H_HT: 
            results_dict['ObjComponents']['SlackPowerTracking'][ts] = value(sum(model.SlackPowerTracking[ts,t]*1e6 for t in model.t))

    # General model features
    results_dict['Tl_hot']      = list(getattr(model, 'Tl_hot', []))      # Tl_hot is not defined for Energy Flow model
    results_dict['Tl_cold']     = list(getattr(model, 'Tl_cold', []))     # Tl_cold is not defined for Energy Flow model
    results_dict['Tl_stor']     = list(getattr(model, 'Tl_stor', []))     # Tl_stor is not defined for Energy Flow model
    results_dict['DT_flow']     = Opts['DT_flows']

    results_dict['Tinit']           = Opts['TstepInit']
    results_dict['InitialState']    = model.InitialState
    results_dict['dt']              = model.dt
    results_dict['nSteps']          = model.nSteps

    # Additional dictionary info
    results_dict['Opts']  = Opts

    # #Collecting model features in dictionar
    results_dict['MESConfig'] = {}
    results_dict['MESConfig'] = MESConfig

    if Opts['BaselineDefinition'] != 'None':
        # Collect prices from baseline for future comparison in post processing with RH results
        results_dict['Economics'] = {}
        for n in model.Networks:
            results_dict['Economics'][n] = {}
            results_dict['Economics'][n]['BuyPrice']     = value(model.NetBuyPrice[n])
            results_dict['Economics'][n]['SellPrice']    = value(model.NetSellPrice[n])

    results_dict['Sets']                = {}
    results_dict['Sets']['Vectors']     = list(model.Vectors)
    results_dict['Sets']['Generators']  = list(model.Generators)
    results_dict['Sets']['Storages']    = list(model.Storages)
    results_dict['Sets']['Networks']    = list(model.Networks)
    results_dict['Sets']['RESgens']     = list(model.RESgens)
    # results_dict['Sets']['Storages_0D'] = list(model.Storages_0D)

    if Opts['ThermalFlows']:
        results_dict['Sets']['Tl_hot']      = list(model.Tl_hot)
        results_dict['Sets']['Tl_cold']     = list(model.Tl_cold)
        results_dict['Sets']['Tl_stor']     = list(model.Tl_stor)
        results_dict['Sets']['Mixers']      = list(model.MixComp)
        results_dict['Sets']['Diverters']   = list(model.DivComp)
        results_dict['Sets']['HotMixComp']  = list(model.HotMixComp)
        results_dict['Sets']['ColdMixComp'] = list(model.ColdMixComp)
        results_dict['Sets']['HotDivComp']  = list(model.HotDivComp)
        results_dict['Sets']['ColdDivComp'] = list(model.ColdDivComp)
        results_dict['Sets']['HeatLoads']   = list(model.HeatLoads)
        results_dict['Sets']['HydrComp']    = list(model.HydrComp)
   
        results_dict['Sets']['Storages_H_HT']       = list(model.Storages_H_HT)
        results_dict['Sets']['Generators_H_HT']     = list(model.Generators_H_HT)
        results_dict['Sets']['Generators_EE_AC']    = list(model.Generators_EE_AC)
        results_dict['Sets']['NL_Vectors']          = list(model.NL_Vectors)
        results_dict['Sets']['L_Vectors']           = list(model.L_Vectors)

    if Opts['ThermalFlows']:
        if len(model.Storages_H_HT):
            results_dict['Sets']['l']   =  [str(element) for element in model.l]    
        else:
            results_dict['Sets']['l']   = 0
        
    
    results_dict['Dimen']   = {}
    results_dict['Dimen']['nVects'] = model.nVects
    results_dict['Dimen']['nGens'] = model.nGens
    results_dict['Dimen']['nStors'] = model.nStors
    results_dict['Dimen']['nNetws'] = model.nNetws

    ##################################################
    # Getting the results from the Energy Flow model #
    ##################################################

    for g in model.Generators: 
        # By definition, all the Generators have at least one output
        # The checks are performed only on the second output
        results_dict[g]             = {}
        results_dict[g]['xOnOff']   = list(np.array(list(model.xGenOn[g,:].value)).round(decimals=nDigits))
        results_dict[g]['xGenSU']   = list(np.array(list(model.xGenSU[g,:].value)).round(decimals=nDigits))
        results_dict[g]['xGenSD']   = list(np.array(list(model.xGenSD[g,:].value)).round(decimals=nDigits))   
        results_dict[g]['xGenOn']   = list(np.array(list(model.xGenOn[g,:].value)).round(decimals=nDigits))      
        results_dict[g]['Out1']     = list(np.array(list(model.GenOut1[g,:].value)).round(decimals=nDigits))
        if isinstance(MESConfig['Generators'][g]['Output 2'],str) and MESConfig['Generators'][g]['Output 2'] in model.Vectors:
            results_dict[g]['Out2']     = list(np.array(list(model.GenOut2[g,:].value)).round(decimals=nDigits))
        else:
            results_dict[g]['Out2']     = list(np.zeros(int(model.nSteps)))
        results_dict[g]['In']       = list(np.array(list(model.GenIn[g,:].value)).round(decimals=nDigits))

    if Opts['EnergyFlows']:
        for s in model.Storages: 
            results_dict[s]             = {}
            results_dict[s]['StorIn']   = list(np.array(list(model.StorIn[s,:].value)).round(decimals=nDigits))
            results_dict[s]['StorOut']  = list(np.array(list(model.StorOut[s,:].value)).round(decimals=nDigits))
            results_dict[s]['StorC']    = list(np.array(list(model.StorC[s,:].value)).round(decimals=nDigits)) 
            # results_dict[s]['StorC'].insert(0, model.StorC0[s].value)
            results_dict[s]['xStorCH']  = list(np.array(list(model.xStorCH[s,:].value)).round(decimals=nDigits))
            results_dict[s]['xStorDH']  = list(np.array(list(model.xStorDH[s,:].value)).round(decimals=nDigits))

    if Opts['ThermalFlows']:
        for ts in model.Storages_H_HT:
            results_dict[ts]             = {}
            results_dict[s]['xStorCH']  = list(np.array(list(model.xStorCH[ts,:].value)).round(decimals=nDigits))
            results_dict[s]['xStorDH']  = list(np.array(list(model.xStorDH[ts,:].value)).round(decimals=nDigits))

        for hl in model.HeatLoads: 
            results_dict[hl]             = {}
            results_dict[hl]['Demand']   = list(np.array(list(model.QHL[hl,:].value)).round(decimals=nDigits))
        
    for n in model.Networks:
        results_dict[n]             = {}
        results_dict[n]['NetBuy']   = list(np.array(list(model.NetBuy[n,:].value)).round(decimals=nDigits))  
        results_dict[n]['NetSell']  = list(np.array(list(model.NetSell[n,:].value)).round(decimals=nDigits))  

    for v in model.Vectors:
        results_dict[v]             = {}
        try:
            ExternalDemand = list(np.array(list(model.DemandsDict[v])[0+Opts['TstepInit']:int(model.nSteps+Opts['TstepInit'])]).round(decimals=nDigits))
            results_dict[v]['External demand'] = [float(x) for x in ExternalDemand]
        except:
            results_dict[v]['External demand'] = list(np.zeros(model.nSteps))
        results_dict[v]['Slack']    = list(np.array(list(model.VectorSlack[v,:].value)).round(decimals=nDigits))

    for r in model.RESgens:
        results_dict[r]             = {}
        results_dict[r]['Out']      = list(np.array(list(model.RESout[r,:].value)).round(decimals=nDigits))  
        results_dict[r]['Curt']     = list(np.array(list(model.RESout[r,:].value)).round(decimals=nDigits))

    ###################################################
    # Getting the results from the Thermal Flow model #
    ###################################################
        
    if Opts['ThermalFlows']:
        results_dict['MassFlows'] = {}  # Dictionary to store the mass flows for each component with values at the non null temperature levels (up to 2 - SOS2 conditions)

        def process_mass_flows(Temp, Tl_Vec, name, t , results_dict): # Function to extract the mass flows from the pyomo model with the corresponding temperatures 
            NonNullTemp = Temp[Temp >= 1e-3]
            NonNullIdx  = np.nonzero(Temp >= 1e-3)[0]
            if len(NonNullTemp) == 0:
                Vres = {1: 0, 2:0, 3:1}
                Tres = {1: 0, 2:0, 3:1}
            elif len(NonNullTemp) == 1:
                Vres = {1: NonNullTemp[0], 2:0, 3:1}
                Tres = {1: Tl_Vec[NonNullIdx.item()], 2:0, 3:1}
            elif len(NonNullTemp) == 2:
                Vres = {1: NonNullTemp[0], 2:NonNullTemp[1], 3:2}
                Tres = {1: Tl_Vec[NonNullIdx[0].item()], 2:Tl_Vec[NonNullIdx[1].item()], 3:2}
            else:
                raise Exception('More than 2 non null values as flow rate, check '+name+' at time '+t)

            return Vres, Tres
        
        for g in model.Generators_H_HT:   # To collect the mass flows for each Genine (up to 2 mass flows at adjacent temperature levels)
            results_dict['MassFlows'][g] = {} 
            for t in model.t: 
                results_dict['MassFlows'][g][t] = {}

                Temp = np.array(list(model.GenVout[g,:,t].value)).round(decimals=nDigits*2)
                results_dict['MassFlows'][g][t]['Vout'], results_dict['MassFlows'][g][t]['Tout'] = process_mass_flows(Temp, results_dict['Tl_hot'], g, t , results_dict)

                Temp = np.array(list(model.GenVin[g,:,t].value)).round(decimals=nDigits*2)
                results_dict['MassFlows'][g][t]['Vin'], results_dict['MassFlows'][g][t]['Tin'] = process_mass_flows(Temp, results_dict['Tl_cold'], g, t , results_dict)

        for hl in model.HeatLoads:      # To collect the mass flows for each heat load (up to 2 mass flows at adjacent temperature levels)
            results_dict['MassFlows'][hl] = {} 
            for t in model.t: 
                results_dict['MassFlows'][hl][t] = {}

                Temp = np.array(list(model.HLVin[hl,:,t].value)).round(decimals=nDigits*2)
                results_dict['MassFlows'][hl][t]['Vin'], results_dict['MassFlows'][hl][t]['Tin'] = process_mass_flows(Temp, results_dict['Tl_hot'], hl, t , results_dict)

                Temp = np.array(list(model.HLVout[hl,:,t].value)).round(decimals=nDigits*2)
                results_dict['MassFlows'][hl][t]['Vout'], results_dict['MassFlows'][hl][t]['Tout'] = process_mass_flows(Temp, results_dict['Tl_cold'], hl, t , results_dict)

        for mix in model.MixComp:       # To collect the mass flows for each mixer (up to 2 mass flows at adjacent temperature levels)
            results_dict['MassFlows'][mix] = {}
            for t in model.t:
                results_dict['MassFlows'][mix][t] = {}

                if mix in model.HotMixComp:
                    Temp = np.array(list(model.hVMix[mix,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassFlows'][mix][t]['Vmix'], results_dict['MassFlows'][mix][t]['Tmix'] = \
                        process_mass_flows(Temp, results_dict['Tl_hot'], mix, t , results_dict)
                elif mix in model.ColdMixComp:
                    Temp = np.array(list(model.cVMix[mix,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassFlows'][mix][t]['Vmix'], results_dict['MassFlows'][mix][t]['Tmix'] = \
                        process_mass_flows(Temp, results_dict['Tl_cold'], mix, t , results_dict)

        for div in model.DivComp:       # To collect the mass flows for each diverter (up to 2 mass flows at adjacent temperature levels)
            results_dict['MassFlows'][div] = {}
            for t in model.t:
                results_dict['MassFlows'][div][t] = {}

                if div in model.HotDivComp:
                    Temp = np.array(list(model.hVDiv[div,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassFlows'][div][t]['Vdiv'], results_dict['MassFlows'][div][t]['Tdiv'] = \
                        process_mass_flows(Temp, results_dict['Tl_hot'], div, t , results_dict)
                elif div in model.ColdDivComp:
                    Temp = np.array(list(model.cVDiv[div,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassFlows'][div][t]['Vdiv'], results_dict['MassFlows'][div][t]['Tdiv'] = \
                        process_mass_flows(Temp, results_dict['Tl_cold'], div, t , results_dict)
        
        for ts in model.Storages_H_HT:
            results_dict['MassFlows'][ts] = {}
            for t in model.t:
                results_dict['MassFlows'][ts][t] = {}

                if Opts['StratStorModel'] == '2L': # To collect the mass flows for each storage (up to 2 mass flows at adjacent temperature levels)
                    Temp = np.array(list(model.TESShVout[ts,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassFlows'][ts][t]['hVout'], results_dict['MassFlows'][ts][t]['hTout'] = \
                        process_mass_flows(Temp, results_dict['Tl_stor'], ts, t , results_dict)
                                    
                    Temp = np.array(list(model.TESScVout[ts,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassFlows'][ts][t]['cVout'], results_dict['MassFlows'][ts][t]['cTout'] = \
                        process_mass_flows(Temp, results_dict['Tl_stor'], ts, t , results_dict)
                
                else:
                    Temp = np.array(list(model.TESShVout[ts,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassFlows'][ts][t]['hVout'], results_dict['MassFlows'][ts][t]['hTout'] = \
                        process_mass_flows(Temp, results_dict['Tl_stor'], ts, t , results_dict)
                                    
                    Temp = np.array(list(model.TESScVout[ts,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassFlows'][ts][t]['cVout'], results_dict['MassFlows'][ts][t]['cTout'] = \
                        process_mass_flows(Temp, results_dict['Tl_stor'], ts, t , results_dict)

                
                Temp = np.array(list(model.TESShVin[ts,:,t].value)).round(decimals=nDigits*2)
                results_dict['MassFlows'][ts][t]['hVin'], results_dict['MassFlows'][ts][t]['hTin'] = \
                    process_mass_flows(Temp, results_dict['Tl_hot'], ts, t , results_dict)
                
                Temp = np.array(list(model.TESScVin[ts,:,t].value)).round(decimals=nDigits*2)
                results_dict['MassFlows'][ts][t]['cVin'], results_dict['MassFlows'][ts][t]['cTin'] = \
                    process_mass_flows(Temp, results_dict['Tl_cold'], ts, t , results_dict)
            
            if Opts['StratStorModel'] == 'CLV':
                results_dict['MassStor'] = {}
                results_dict['MassConv'] = {}
                results_dict['MassStor'][ts] = {}
                results_dict['MassConv'][ts] = {}
                results_dict['MassConv'][ts]['Vup'] = {}
                results_dict['MassConv'][ts]['Vdw'] = {}

                for l in model.l:
                    results_dict['MassStor'][ts][l] = {}
                    results_dict['MassConv'][ts]['Vup'][l]  = {}
                    results_dict['MassConv'][ts]['Vdw'][l]  = {}
                    for t in model.t_st:
                        results_dict['MassStor'][ts][l][t] = {}

                        Temp = np.array(list(model.TESS_Ml[ts,l,:,t].value)).round(decimals=nDigits*2)
                        results_dict['MassStor'][ts][l][t]['M'], results_dict['MassStor'][ts][l][t]['T'] = \
                            process_mass_flows(Temp, results_dict['Tl_stor'], ts, t , results_dict)
                        
                        results_dict['MassConv'][ts]['Vup'][l][t] = {}
                        results_dict['MassConv'][ts]['Vdw'][l][t] = {}

                        Temp = np.array(list(model.TESS_Vup[ts,l,:,t].value)).round(decimals=nDigits*2)
                        results_dict['MassConv'][ts]['Vup'][l][t]['M'], results_dict['MassConv'][ts]['Vup'][l][t]['T'] = \
                            process_mass_flows(Temp, results_dict['Tl_stor'], ts, t , results_dict)
                        
                        Temp = np.array(list(model.TESS_Vdw[ts,l,:,t].value)).round(decimals=nDigits*2)
                        results_dict['MassConv'][ts]['Vdw'][l][t]['M'], results_dict['MassConv'][ts]['Vdw'][l][t]['T'] = \
                            process_mass_flows(Temp, results_dict['Tl_stor'], ts, t , results_dict)

            elif Opts['StratStorModel'] == '2L':
                results_dict['MassStor_Hot'] = {}
                results_dict['MassStor_Hot'][ts] = {}
                for t in model.t_st:
                    results_dict['MassStor_Hot'][ts][t] = {}

                    Temp = np.array(list(model.TESS_MlHot[ts,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassStor_Hot'][ts][t]['M'], results_dict['MassStor_Hot'][ts][t]['T'] = \
                        process_mass_flows(Temp, results_dict['Tl_hot'], ts, t , results_dict)
                    
                results_dict['MassStor_Cold'] = {}
                results_dict['MassStor_Cold'][ts] = {}
                for t in model.t_st:
                    results_dict['MassStor_Cold'][ts][t] = {}

                    Temp = np.array(list(model.TESS_MlCold[ts,:,t].value)).round(decimals=nDigits*2)
                    results_dict['MassStor_Cold'][ts][t]['M'], results_dict['MassStor_Cold'][ts][t]['T'] = \
                        process_mass_flows(Temp, results_dict['Tl_cold'], ts, t , results_dict)
                    

            elif Opts['StratStorModel'] == 'CLT':
                raise('CLT model not implemented yet')
        
            # For 2L model, it is not necessary to collect the internal mass flows


        results_dict['UniFlows'] = {}  # Dictionary to store the mass flows for each component as unique values at the mixing temperature 
        
        def UnifyFlows(VtempDict, TtempDict): # Unification of the couples from ['MassFlows'] dictionary
            CheckFlows = VtempDict[3]
            if CheckFlows == 1:
                Vtemp = VtempDict[1]
                Ttemp = TtempDict[1]
            elif CheckFlows == 2:
                Vtemp = VtempDict[1] + VtempDict[2]
                if Vtemp != 0:
                    Ttemp = (VtempDict[1]*TtempDict[1] + VtempDict[2]*TtempDict[2])/Vtemp
                else:
                    Ttemp = 0
            
            return Vtemp, Ttemp
        
        for g in model.Generators_H_HT:  # To compute actual mass flow and temperature (weighted average) for each Genine 
            results_dict['UniFlows'][g] = {}
            Vin, Vout, Tin, Tout = [], [], [], []

            for t in model.t:
                VtempDict   = results_dict['MassFlows'][g][t]['Vin']
                TtempDict   = results_dict['MassFlows'][g][t]['Tin']
                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                Vin.append(Vtemp)
                Tin.append(Ttemp)

                VtempDict   = results_dict['MassFlows'][g][t]['Vout']
                TtempDict   = results_dict['MassFlows'][g][t]['Tout']
                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                Vout.append(Vtemp)
                Tout.append(Ttemp)

            results_dict['UniFlows'][g]['Vin']  = Vin
            results_dict['UniFlows'][g]['Tin']  = Tin
            results_dict['UniFlows'][g]['Vout'] = Vout
            results_dict['UniFlows'][g]['Tout'] = Tout

        for hl in model.HeatLoads:  # To compute actual mass flow and temperature (weighted average) for each heat load
            results_dict['UniFlows'][hl] = {}
            Vin, Vout, Tin, Tout = [], [], [], []

            for t in model.t:
                VtempDict   = results_dict['MassFlows'][hl][t]['Vin']
                TtempDict   = results_dict['MassFlows'][hl][t]['Tin']
                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                Vin.append(Vtemp)
                Tin.append(Ttemp)


                VtempDict   = results_dict['MassFlows'][hl][t]['Vout']
                TtempDict   = results_dict['MassFlows'][hl][t]['Tout']
                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                Vout.append(Vtemp)
                Tout.append(Ttemp)
        
            results_dict['UniFlows'][hl]['Vin']  = Vin
            results_dict['UniFlows'][hl]['Tin']  = Tin
            results_dict['UniFlows'][hl]['Vout'] = Vout
            results_dict['UniFlows'][hl]['Tout'] = Tout

        for mix in model.MixComp:   # To compute actual mass flow and temperature (weighted average) for each mixer
            results_dict['UniFlows'][mix] = {}
            Vmix, Tmix = [], []

            for t in model.t:
                VtempDict = results_dict['MassFlows'][mix][t]['Vmix']
                TtempDict = results_dict['MassFlows'][mix][t]['Tmix']

                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                Vmix.append(Vtemp)
                Tmix.append(Ttemp)
        
            results_dict['UniFlows'][mix]['Vmix']  = Vmix
            results_dict['UniFlows'][mix]['Tmix']  = Tmix

        for div in model.DivComp:   # To compute actual mass flow and temperature (weighted average) for each diverter
            results_dict['UniFlows'][div] = {}
            Vdiv, Tdiv = [], []

            for t in model.t:
                VtempDict = results_dict['MassFlows'][div][t]['Vdiv']
                TtempDict = results_dict['MassFlows'][div][t]['Tdiv']

                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                Vdiv.append(Vtemp)
                Tdiv.append(Ttemp)

            results_dict['UniFlows'][div]['Vdiv']  = Vdiv
            results_dict['UniFlows'][div]['Tdiv']  = Tdiv

        for ts in model.Storages_H_HT:
            results_dict['UniFlows'][ts] = {}
            hVout, hTout, hVin, hTin, cVout, cTout, cVin, cTin = [], [], [], [], [], [], [], []

            for t in model.t:
                VtempDict = results_dict['MassFlows'][ts][t]['hVout']
                TtempDict = results_dict['MassFlows'][ts][t]['hTout']
                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                hVout.append(Vtemp)
                hTout.append(Ttemp)

                VtempDict = results_dict['MassFlows'][ts][t]['hVin']
                TtempDict = results_dict['MassFlows'][ts][t]['hTin']
                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                hVin.append(Vtemp)
                hTin.append(Ttemp)

                VtempDict = results_dict['MassFlows'][ts][t]['cVout']
                TtempDict = results_dict['MassFlows'][ts][t]['cTout']
                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                cVout.append(Vtemp)
                cTout.append(Ttemp)

                VtempDict = results_dict['MassFlows'][ts][t]['cVin']
                TtempDict = results_dict['MassFlows'][ts][t]['cTin']
                Vtemp, Ttemp = UnifyFlows(VtempDict, TtempDict)
                cVin.append(Vtemp)
                cTin.append(Ttemp)
            
            results_dict['UniFlows'][ts]['hVout']   = hVout
            results_dict['UniFlows'][ts]['hTout']   = hTout
            results_dict['UniFlows'][ts]['hVin']    = hVin
            results_dict['UniFlows'][ts]['hTin']    = hTin
            results_dict['UniFlows'][ts]['cVout']   = cVout
            results_dict['UniFlows'][ts]['cTout']   = cTout
            results_dict['UniFlows'][ts]['cVin']    = cVin
            results_dict['UniFlows'][ts]['cTin']    = cTin
                                    
        
        results_dict['UniStor'] = {}
        for ts in model.Storages_H_HT:
            results_dict['UniStor'][ts]  = {}
            if Opts['StratStorModel'] == 'CLT':
                results_dict['UniStor'][ts]['M'] = {}
                for Tl in model.Tl_stor:
                    results_dict['UniStor'][ts]['M'][Tl] = list(model.TESS_Ml[ts,Tl,:].value)

            if Opts['StratStorModel'] == 'CLV':
                results_dict['UniStor'][ts]['M']    = {}
                results_dict['UniStor'][ts]['T']    = {}
                results_dict['UniStor'][ts]['Vup']  = {}
                results_dict['UniStor'][ts]['Tup']  = {}
                results_dict['UniStor'][ts]['Vdw']  = {}
                results_dict['UniStor'][ts]['Tdw']  = {}

                for l in model.l:
                    Ml, Tl, Vup, Tup, Vdw, Tdw = [], [], [], [], [], []

                    for t in model.t:
                        Mltemp = results_dict['MassStor'][ts][l][t]['M']
                        Tltemp = results_dict['MassStor'][ts][l][t]['T']
                        Vtemp, Ttemp = UnifyFlows(Mltemp, Tltemp)
                        Ml.append(Vtemp)
                        Tl.append(Ttemp)

                        VupTemp = results_dict['MassConv'][ts]['Vup'][l][t]['M']
                        TupTemp = results_dict['MassConv'][ts]['Vup'][l][t]['T']
                        Vtemp, Ttemp = UnifyFlows(VupTemp, TupTemp)
                        Vup.append(Vtemp)
                        Tup.append(Ttemp)

                        VdwTemp = results_dict['MassConv'][ts]['Vdw'][l][t]['M']
                        TdwTemp = results_dict['MassConv'][ts]['Vdw'][l][t]['T']
                        Vtemp, Ttemp = UnifyFlows(VdwTemp, TdwTemp)
                        Vdw.append(Vtemp)
                        Tdw.append(Ttemp)

                    results_dict['UniStor'][ts]['M'][l]     = Ml
                    results_dict['UniStor'][ts]['T'][l]     = Tl
                    results_dict['UniStor'][ts]['Vup'][l]   = Vup
                    results_dict['UniStor'][ts]['Tup'][l]   = Tup
                    results_dict['UniStor'][ts]['Vdw'][l]   = Vdw
                    results_dict['UniStor'][ts]['Tdw'][l]   = Tdw

            elif Opts['StratStorModel'] == '2L':
                results_dict['UniStor'][ts]['M_hot']    = {}
                results_dict['UniStor'][ts]['M_cold']   = {}
                for Tl in model.Tl_hot:
                    results_dict['UniStor'][ts]['M_hot'][Tl]    = list(model.TESS_MlHot[ts,Tl,:].value)
                for Tl in model.Tl_cold:
                    results_dict['UniStor'][ts]['M_cold'][Tl]   = list(model.TESS_MlCold[ts,Tl,:].value)

                results_dict['UniStor'][ts]['M_hotMerge'] = {}
                results_dict['UniStor'][ts]['T_hotMerge'] = {}

                Ml, Tl = [], []

                for t in model.t:
                    Mltemp = results_dict['MassStor_Hot'][ts][t]['M']
                    Tltemp = results_dict['MassStor_Hot'][ts][t]['T']
                    Vtemp, Ttemp = UnifyFlows(Mltemp, Tltemp)
                    Ml.append(Vtemp)
                    Tl.append(Ttemp)
                
                results_dict['UniStor'][ts]['M_hotMerge'] = Ml
                results_dict['UniStor'][ts]['T_hotMerge'] = Tl

                results_dict['UniStor'][ts]['M_coldMerge'] = {}
                results_dict['UniStor'][ts]['T_coldMerge'] = {}

                Ml, Tl = [], []

                for t in model.t:
                    Mltemp = results_dict['MassStor_Cold'][ts][t]['M']
                    Tltemp = results_dict['MassStor_Cold'][ts][t]['T']
                    Vtemp, Ttemp = UnifyFlows(Mltemp, Tltemp)
                    Ml.append(Vtemp)
                    Tl.append(Ttemp)

                results_dict['UniStor'][ts]['M_coldMerge'] = Ml
                results_dict['UniStor'][ts]['T_coldMerge'] = Tl

    return results_dict

def CumulatingRHResults(model,Opts,RHresults_dict,results_dict):

    Adv_dict    = {}
    nStepsAdv   = Opts['nStepsAdv'] 

    # Collecting results for the control horizon
    for g in model.Generators: 
        Adv_dict[g] = {}
        Adv_dict[g]['xOnOff']   = results_dict[g]['xOnOff'][0:nStepsAdv]
        Adv_dict[g]['xGenSU']   = results_dict[g]['xGenSU'][0:nStepsAdv]
        Adv_dict[g]['xGenSD']   = results_dict[g]['xGenSD'][0:nStepsAdv]
        Adv_dict[g]['xGenOn']   = results_dict[g]['xGenOn'][0:nStepsAdv]
        Adv_dict[g]['Out1']     = results_dict[g]['Out1'][0:nStepsAdv]
        Adv_dict[g]['Out2']     = results_dict[g]['Out2'][0:nStepsAdv]
        Adv_dict[g]['In']       = results_dict[g]['In'][0:nStepsAdv]

    ## TODO: Add info required in the commented section
    # for s in model.Storages:
    #     Adv_dict[s] = {}
    #     Adv_dict[s]['StorIn']  = results_dict[s]['StorIn'][0:nStepsAdv]
    #     Adv_dict[s]['StorOut'] = results_dict[s]['StorOut'][0:nStepsAdv]
    #     Adv_dict[s]['xStorCHDH'] = results_dict[s]['xStorCHDH'][0:nStepsAdv]
    #     if Opts['TstepInit'] == 0:
    #         Adv_dict[s]['StorC']   = results_dict[s]['StorC'][0:nStepsAdv+1]
    #     else: 
    #         Adv_dict[s]['StorC']   = results_dict[s]['StorC'][1:nStepsAdv+1]

    if Opts['ThermalFlows']:
        for ts in model.Storages_H_HT:
            Adv_dict[ts] = {}
            Adv_dict[ts]['xStorCH'] = results_dict[ts]['xStorCH'][0:nStepsAdv]
            Adv_dict[ts]['xStorDH'] = results_dict[ts]['xStorDH'][0:nStepsAdv]
    
    for n in model.Networks:
        Adv_dict[n] = {}
        Adv_dict[n]['NetBuy']  = results_dict[n]['NetBuy'][0:nStepsAdv]
        Adv_dict[n]['NetSell'] = results_dict[n]['NetSell'][0:nStepsAdv]
    
    for v in model.Vectors:
        Adv_dict[v] = {}
        Adv_dict[v]['External demand'] = results_dict[v]['External demand'][0:nStepsAdv]
        Adv_dict[v]['Slack']  = results_dict[v]['Slack'][0:nStepsAdv]

    for r in model.RESgens:
        Adv_dict[r] = {}
        Adv_dict[r]['Out']   = results_dict[r]['Out'][0:nStepsAdv] 
        Adv_dict[r]['Curt']  = results_dict[r]['Curt'][0:nStepsAdv]

    if Opts['ThermalFlows']:
        Adv_dict['UniFlows'] = {}

        for g in model.Generators_H_HT:
            Adv_dict['UniFlows'][g] = {}
            Adv_dict['UniFlows'][g]['Vin']  = results_dict['UniFlows'][g]['Vin'][0:nStepsAdv]
            Adv_dict['UniFlows'][g]['Tin']  = results_dict['UniFlows'][g]['Tin'][0:nStepsAdv]
            Adv_dict['UniFlows'][g]['Vout'] = results_dict['UniFlows'][g]['Vout'][0:nStepsAdv]
            Adv_dict['UniFlows'][g]['Tout'] = results_dict['UniFlows'][g]['Tout'][0:nStepsAdv]

        for hl in model.HeatLoads:
            Adv_dict['UniFlows'][hl] = {}
            Adv_dict['UniFlows'][hl]['Vin']     = results_dict['UniFlows'][hl]['Vin'][0:nStepsAdv]
            Adv_dict['UniFlows'][hl]['Tin']     = results_dict['UniFlows'][hl]['Tin'][0:nStepsAdv]
            Adv_dict['UniFlows'][hl]['Vout']    = results_dict['UniFlows'][hl]['Vout'][0:nStepsAdv]
            Adv_dict['UniFlows'][hl]['Tout']    = results_dict['UniFlows'][hl]['Tout'][0:nStepsAdv]
            # Adv_dict['UniFlows'][hl]['Tslack']  = value(sum(model.SlackTminHL[hl,tl,t] for tl in model.Tl_hot for t in range(0,nStepsAdv)))
            try:
                Tdel = results_dict['UniFlows']['hMIX_TESS']['Tmix'][0:nStepsAdv]
                Tin  = results_dict['UniFlows']['HL1']['Tin']
                DTdel = [Tdel[i] - Tin[i] for i in range(nStepsAdv)]
                Adv_dict['UniFlows'][hl]['DTslack']  =  DTdel
                for i in range(nStepsAdv):
                    if Tdel[i] < model.ConfigParams['Thermal Loads']['HL1']['Tmin'] - 0.1:
                        Adv_dict['UniFlows'][hl]['#LowDeliveryTemp'] = 1
                    else:
                        Adv_dict['UniFlows'][hl]['#LowDeliveryTemp'] = 0
            except:
                pass

        for mix in model.MixComp:
            Adv_dict['UniFlows'][mix] = {}
            Adv_dict['UniFlows'][mix]['Vmix']  = results_dict['UniFlows'][mix]['Vmix'][0:nStepsAdv]
            Adv_dict['UniFlows'][mix]['Tmix']  = results_dict['UniFlows'][mix]['Tmix'][0:nStepsAdv]
        
        for div in model.DivComp:
            Adv_dict['UniFlows'][div] = {}
            Adv_dict['UniFlows'][div]['Vdiv']  = results_dict['UniFlows'][div]['Vdiv'][0:nStepsAdv]
            Adv_dict['UniFlows'][div]['Tdiv']  = results_dict['UniFlows'][div]['Tdiv'][0:nStepsAdv]
        
        for ts in model.Storages_H_HT:
            Adv_dict['UniFlows'][ts] = {}
            Adv_dict['UniFlows'][ts]['hVout']   = results_dict['UniFlows'][ts]['hVout'][0:nStepsAdv]
            Adv_dict['UniFlows'][ts]['hTout']   = results_dict['UniFlows'][ts]['hTout'][0:nStepsAdv]
            Adv_dict['UniFlows'][ts]['hVin']    = results_dict['UniFlows'][ts]['hVin'][0:nStepsAdv]
            Adv_dict['UniFlows'][ts]['hTin']    = results_dict['UniFlows'][ts]['hTin'][0:nStepsAdv]
            Adv_dict['UniFlows'][ts]['cVout']   = results_dict['UniFlows'][ts]['cVout'][0:nStepsAdv]
            Adv_dict['UniFlows'][ts]['cTout']   = results_dict['UniFlows'][ts]['cTout'][0:nStepsAdv]
            Adv_dict['UniFlows'][ts]['cVin']    = results_dict['UniFlows'][ts]['cVin'][0:nStepsAdv]
            Adv_dict['UniFlows'][ts]['cTin']    = results_dict['UniFlows'][ts]['cTin'][0:nStepsAdv]
        
        Adv_dict['UniStor'] = {}
        for ts in model.Storages_H_HT:
            Adv_dict['UniStor'][ts] = {}
            if Opts['StratStorModel'] == 'CLV':
                Adv_dict['UniStor'][ts]['M']    = {}
                Adv_dict['UniStor'][ts]['T']    = {}
                Adv_dict['UniStor'][ts]['Vup']  = {}
                Adv_dict['UniStor'][ts]['Tup']  = {}
                Adv_dict['UniStor'][ts]['Vdw']  = {}
                Adv_dict['UniStor'][ts]['Tdw']  = {}
                for l in model.l:
                    if Opts['TstepInit'] == 0:
                        Adv_dict['UniStor'][ts]['M'][l]     = results_dict['UniStor'][ts]['M'][l][0:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['T'][l]     = results_dict['UniStor'][ts]['T'][l][0:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['Vup'][l]   = results_dict['UniStor'][ts]['Vup'][l][0:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['Tup'][l]   = results_dict['UniStor'][ts]['Tup'][l][0:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['Vdw'][l]   = results_dict['UniStor'][ts]['Vdw'][l][0:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['Tdw'][l]   = results_dict['UniStor'][ts]['Tdw'][l][0:nStepsAdv+1]
                    else:
                        Adv_dict['UniStor'][ts]['M'][l]     = results_dict['UniStor'][ts]['M'][l][1:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['T'][l]     = results_dict['UniStor'][ts]['T'][l][1:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['Vup'][l]   = results_dict['UniStor'][ts]['Vup'][l][1:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['Tup'][l]   = results_dict['UniStor'][ts]['Tup'][l][1:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['Vdw'][l]   = results_dict['UniStor'][ts]['Vdw'][l][1:nStepsAdv+1]
                        Adv_dict['UniStor'][ts]['Tdw'][l]   = results_dict['UniStor'][ts]['Tdw'][l][1:nStepsAdv+1]

            elif Opts['StratStorModel'] == '2L':
                Adv_dict['UniStor'][ts]['M_hot']    = {}
                Adv_dict['UniStor'][ts]['M_cold']   = {}

                for Tl in model.Tl_hot:
                    if Opts['ConvertCLVto2L']:
                        if Opts['TstepInit'] == 0:
                            Adv_dict['UniStor'][ts]['M_hot'][Tl]    = results_dict['UniStor'][ts]['M_hot'][Tl][0:nStepsAdv]
                        else:
                            Adv_dict['UniStor'][ts]['M_hot'][Tl]    = results_dict['UniStor'][ts]['M_hot'][Tl][0:nStepsAdv]
                    else:
                        if Opts['TstepInit'] == 0 or Opts['ConvertCLVto2L']:
                            Adv_dict['UniStor'][ts]['M_hot'][Tl]    = results_dict['UniStor'][ts]['M_hot'][Tl][0:nStepsAdv+1]
                        else:
                            Adv_dict['UniStor'][ts]['M_hot'][Tl]    = results_dict['UniStor'][ts]['M_hot'][Tl][1:nStepsAdv+1]


                for Tl in model.Tl_cold:
                    if Opts['ConvertCLVto2L']:
                        if Opts['TstepInit'] == 0:
                            Adv_dict['UniStor'][ts]['M_cold'][Tl]   = results_dict['UniStor'][ts]['M_cold'][Tl][0:nStepsAdv]
                        else:
                            Adv_dict['UniStor'][ts]['M_cold'][Tl]   = results_dict['UniStor'][ts]['M_cold'][Tl][0:nStepsAdv]
                    else:
                        if Opts['TstepInit'] == 0:
                            Adv_dict['UniStor'][ts]['M_cold'][Tl]   = results_dict['UniStor'][ts]['M_cold'][Tl][0:nStepsAdv+1]
                        else:
                            Adv_dict['UniStor'][ts]['M_cold'][Tl]   = results_dict['UniStor'][ts]['M_cold'][Tl][1:nStepsAdv+1]
                    
            elif Opts['StratStorModel'] == 'CLT':
                raise('CLT model not implemented yet')

            else: 
                raise Exception('Storage model not supported for the advanced horizon - Only CLV is currently implemented')

    if Opts['TstepInit'] == 0: # First RH timestep
        RHresults_dict = Adv_dict
    else:   # Cumulating the results
        for key in Adv_dict.keys():
            for subkey in Adv_dict[key].keys():
                if isinstance(Adv_dict[key][subkey],dict):
                    for subkeyL2 in Adv_dict[key][subkey].keys():
                        if isinstance(Adv_dict[key][subkey][subkeyL2],dict):
                            for subkeyL3 in Adv_dict[key][subkey][subkeyL2].keys():
                                RHresults_dict[key][subkey][subkeyL2][subkeyL3] += Adv_dict[key][subkey][subkeyL2][subkeyL3]
                        else:
                            RHresults_dict[key][subkey][subkeyL2] += Adv_dict[key][subkey][subkeyL2]
                else: 
                    RHresults_dict[key][subkey] += Adv_dict[key][subkey]

    
    RHresults_dict['Opts']      = Opts
    RHresults_dict['MESConfig'] = results_dict['MESConfig']
    RHresults_dict['Sets']      = results_dict['Sets']
    RHresults_dict['nSteps']    = results_dict['nSteps']
    RHresults_dict['Tl_hot']    = results_dict['Tl_hot']      
    RHresults_dict['Tl_cold']   = results_dict['Tl_cold']     
    RHresults_dict['Tl_stor']   = results_dict['Tl_stor']   
    RHresults_dict['Dimen']     = results_dict['Dimen']

    return RHresults_dict

def CreateComparisonDict(RH_dict,Baseline_dict,model,Opts):
    Comparison_dict = {}
    Tlmin = [model.ConfigParams['Thermal Loads'][hl]['Tmin'] for hl in model.HeatLoads]
    Comparison_dict['TminDelivery'] =  sum(Tlmin)/len(Tlmin) # Average of the minimum delivery temperature for the thermal loads
    Comparison_dict['Baseline']  = {}
    Comparison_dict['RH']        = {}

    VarPrices = Baseline_dict['Economics']

    for n in model.Networks:
        Comparison_dict['Baseline'][n]  = {}
        Comparison_dict['RH'][n]        = {}

        BuyBase     = 0
        SellBase    = 0
        BuyRH       = 0
        SellRH      = 0
        for t in range(0,Opts['Tbaseline']):
            if len(VarPrices[n]['BuyPrice']) == 1:
                BuyPrice = VarPrices[n]['BuyPrice']['0']
            elif len(VarPrices[n]['BuyPrice']) == 0:
                BuyPrice = 0
            else:
                BuyPrice = VarPrices[n]['BuyPrice'][str(t)]
            BuyBase += Baseline_dict[n]['NetBuy'][t]*BuyPrice*model.dt
            BuyRH   += RH_dict['DT_flows'][str(Opts['DT_flows'])][n]['NetBuy'][t]*BuyPrice*model.dt

            if len(VarPrices[n]['SellPrice']) == 1:
                SellPrice = VarPrices[n]['SellPrice']['0']
            elif len(VarPrices[n]['SellPrice']) == 0:
                SellPrice = 0
            else:
                SellPrice = VarPrices[n]['SellPrice'][str(t)]
            SellBase += Baseline_dict[n]['NetSell'][t]*SellPrice*model.dt
            SellRH   += RH_dict['DT_flows'][str(Opts['DT_flows'])][n]['NetSell'][t]*SellPrice*model.dt

        Comparison_dict['Baseline'][n]['Buy']  = BuyBase
        Comparison_dict['Baseline'][n]['Sell'] = SellBase
        Comparison_dict['RH'][n]['Buy']       = BuyRH
        Comparison_dict['RH'][n]['Sell']      = SellRH

    ObjBase = 0
    ObjRH   = 0
    for n in model.Networks:
        ObjBase += Comparison_dict['Baseline'][n]['Buy'] - Comparison_dict['Baseline'][n]['Sell']
        ObjRH   += Comparison_dict['RH'][n]['Buy'] - Comparison_dict['RH'][n]['Sell']

    Comparison_dict['Baseline']['Obj']  = ObjBase
    Comparison_dict['RH']['Obj']        = ObjRH


    Comparison_dict['Grid_Baseline']   = [a-b for a,b in zip(Baseline_dict['Grid']['NetBuy'][0:int(Opts['Tbaseline']/Opts['dt'])],Baseline_dict['Grid']['NetSell'][0:int(Opts['Tbaseline']/Opts['dt'])])]
    Comparison_dict['Grid_RH']         = [a-b for a,b in zip(RH_dict['DT_flows'][str(Opts['DT_flows'])]['Grid']['NetBuy'][0:int(Opts['Tbaseline']/Opts['dt'])],RH_dict['DT_flows'][str(Opts['DT_flows'])]['Grid']['NetSell'][0:int(Opts['Tbaseline']/Opts['dt'])])]
    Comparison_dict['Grid_Imbalances'] = [a-b for a,b in zip(Comparison_dict['Grid_Baseline'],Comparison_dict['Grid_RH'])]

    if len([i for i in Comparison_dict['Grid_Imbalances'] if i > 0]) > 0:
        Comparison_dict['AveGrid_PosImbalances'] = sum([i for i in Comparison_dict['Grid_Imbalances'] if i >= 0])/len([i for i in Comparison_dict['Grid_Imbalances'] if i > 0])
    else: 
        Comparison_dict['AveGrid_PosImbalances'] = 0

    if len([i for i in Comparison_dict['Grid_Imbalances'] if i < 0]) > 0:
        Comparison_dict['AveGrid_NegImbalances'] = sum([i for i in Comparison_dict['Grid_Imbalances'] if i < 0])/len([i for i in Comparison_dict['Grid_Imbalances'] if i < 0])
    else: 
        Comparison_dict['AveGrid_NegImbalances'] = 0

    Comparison_dict['RMSE_Imbalance']       = sqrt(mean_squared_error(Comparison_dict['Grid_Baseline'], Comparison_dict['Grid_RH']))

    return Comparison_dict