import json
from DERTF_SetPointsWriter import SetPoint, OPCUA_SetPoint




from opcua import Client
import os

client = Client("opc.tcp://172.25.101.183:4840/freeopcua/server/")  # metti il tuo endpoint
client.connect()
node = client.get_node("ns=4;s=CO.GB.r_Ctrl_setpointGB")  # il tuo NodeID
datatype = node.get_data_type_as_variant_type()
print(datatype)  # Ti dirà se è Float, Double, Int32, ecc.
client.disconnect()

# Simulazione dei risultati di ottimizzazione
results_dict = {
    'S100': {
        'FlowRate': 12.5,   # valore ottimo trovato dall'ottimizzazione
        'Tout': 80.0        # valore ottimo trovato dall'ottimizzazione
    },
    'S400': {
        'FlowRate': 12.5,   # valore ottimo trovato dall'ottimizzazione
        'Tout': 80.0        # valore ottimo trovato dall'ottimizzazione
    },
    'S900': {
        'FlowRate': 12.5,
        'ControlloPressione': False,
        'SetpointPressione': 3
    },
    'S700_HL1': {
        'Tuser': 60.0,
        'Puser': 30.0
    },
    'S700_HL2': {
        'Tuser': 60.0,
        'Puser': 30.0
    },
    'S700_HL3': {
        'Tuser': 60.0,
        'Puser': 30.0
    },
    'S700_HL4': {
        'Tuser': 60.0,
        'Puser': 30.0
    },
}

MeasDict_path = os.path.join(os.getcwd(),'DebugResults', 'L_MPC', 'TH', 'SolutionData',f'Results_tStep0.json')
# Saving the results
with open(MeasDict_path, 'r') as file:
    results_dict = json.load(file)

# Inizializza le classi
sp = SetPoint()
opcua_writer = OPCUA_SetPoint()
opcua_writer.connect()


values_dict_S100 = sp.MatchResults(results_dict, 'S100')
# values_dict_S900 = sp.MatchResults(results_dict, 'S900')

# Scrittura dei valori
opcua_writer.write_value('S100', values_dict_S100)
# opcua_writer.write_value('S900', values_dict_S900)


values_dict_S200 = sp.MatchResults(results_dict, 'S200')
values_dict_S200['Status'] = 'CHARGE'
opcua_writer.write_value('S200', values_dict_S200)

opcua_writer.disconnect()
print("Test completato: valori scritti per S100 e S900.")
