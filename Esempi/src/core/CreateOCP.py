import os
import sys
import platform

# Add the main directory to sys.path
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(main_dir)

import casadi as ca
from src.L_MPC.EL.core.CasADI_functions     import CasADI_sets, CasADI_Generators, CasADI_Storages, CasADI_Renewables, CasADI_ElectricalLoads, CasADI_Network

def CreateCasADIModel(Opts,ConfigParams):

    # Create the optimization problem
    ocp  = ca.Opti()

    Sets = CasADI_sets(Opts,ConfigParams)

    # Create variables and components with internal constraints
    Generators      = {}
    for GenName in ConfigParams['Generators'].keys():
        Generators[GenName]     = CasADI_Generators(ocp,GenName,ConfigParams['Generators'][GenName],Sets)

    Storages        = {}
    for StorageName in ConfigParams['Storages'].keys():
        Storages[StorageName]   = CasADI_Storages(ocp,StorageName,ConfigParams['Storages'][StorageName],Sets)

    Renewables     = {}
    for ResName in ConfigParams['RES'].keys():
        Renewables[ResName]     = CasADI_Renewables(ocp,ResName,ConfigParams['RES'][ResName],Sets)

    ElectricalLoads = {}
    for LoadName in ConfigParams['ElectricalLoads'].keys():
        ElectricalLoads[LoadName] = CasADI_ElectricalLoads(ocp,LoadName,Sets)

    EL = CasADI_Network(ocp,Generators,Storages,Renewables,ElectricalLoads,Sets)

    return [ocp,EL]

def SetSolverOptions(ocp):

    os_name = platform.system()
    if os_name == 'Windows':
        ipopt_options = {
            'ipopt.max_iter': 1000000,
            'ipopt.tol': 1e-1,
            'ipopt.acceptable_tol': 1e-1,
            'ipopt.acceptable_constr_viol_tol': 1e-1,
            'ipopt.print_level': 5,
            'print_time': True,
            'ipopt.output_file': 'ipopt_debug.txt'
        }
    elif os_name == 'Linux':
        ipopt_options = {
            'ipopt.max_iter': 1000000,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_constr_viol_tol': 1e-3,
            'ipopt.print_level': 5,
            'print_time': True,
            'ipopt.output_file': '/dev/null'
        }
    ocp.solver('ipopt', ipopt_options)
    return ocp