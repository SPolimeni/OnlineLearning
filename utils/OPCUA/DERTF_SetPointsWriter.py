import os
import json
import pandas as pd
from opcua import Client, ua
from opcua.ua.uaerrors import UaStatusCodeError
import io
import warnings

def Read_SetPointsJson(Opts):
    """
    Legge il file Excel e restituisce un dizionario di DataFrame, uno per ogni sistema.
    """
    JsonFilePath = os.path.join(Opts['SetPointsDataPath'],Opts['json_OutputFileName_SetPoints'])
    with open(JsonFilePath, 'r') as file:
        SetPointsJson = json.load(file)
    
    SetPoints_df = {}
    for Tag in SetPointsJson.keys():
        SetPoints_df[Tag] = pd.read_json(io.StringIO(SetPointsJson[Tag]), orient='records')

    return SetPoints_df

class SetPoint:
    def __init__(self, **kwargs):
        SetPointsDataPath = os.path.join(os.getcwd(),'utils','Communication', 'OPCUA','SetPointsData')
        Opts = {'json_OutputFileName_SetPoints'     : 'SetPoints.json',
                'SetPointsDataPath'                 : SetPointsDataPath}

        self.SetPoints_df = Read_SetPointsJson(Opts)
        rho   = 1000
        cp    = 1.162*1e-3
        self.FlowRateConvM3h = 1/(rho*cp)
        self.ConvM3h_Lmin    = 16.6667

    def MatchResults(self, results_dict, system):
        """
        Restituisce un dict {Name: valore} per il sistema, pronto per la scrittura OPC UA.
        Per S200 restituisce anche lo stato.
        """
        values_dict = {}

        if system == 'S100' or system == 'S300' or system == 'S400':

            FlowRate = results_dict[system]['m'][0]
            Tout = results_dict[system]['T_out'][0]

            values_dict['FlowRate'] = min(max(FlowRate,3),15)
            warnings.warn("Limitaazione flow rate e T out S100")
            if system == 'S100':
                values_dict['Tout'] = max(Tout+2,70)
            else:
                values_dict['Tout'] = max(Tout,70)

        elif system == 'S200':
            m_ch = results_dict[system]['m_ch'][0] if 'm_ch' in results_dict[system] else 0
            m_dch = results_dict[system]['m_dch'][0] if 'm_dch' in results_dict[system] else 0
            soglia = 1

            # Determina lo stato
            if m_ch > soglia and m_dch < soglia:
                status = 'CHARGE'
                FlowRate = m_ch
            elif m_dch > soglia and m_ch < soglia:
                status = 'DISCHARGE'
                FlowRate = m_dch
            else:
                status = 'STANDBY'
                FlowRate = 0

            values_dict['Status'] = status
            values_dict['FlowRate'] = FlowRate

            # Per ogni riga del DataFrame, scegli il valore giusto
            df = self.SetPoints_df[system]
            col = f'VALUE {status}'
            for _, row in df.iterrows():
                # name = row['Name'].strip()
                name = row['Name']
                val = row[col]
                # Se FlowRate e valore "EMS", sostituisci con quello calcolato
                if name == 'FlowRate' and val == 'EMS':
                    val = FlowRate
                values_dict[name] = val

        elif system == 'S500':
            FlowRate_Network = results_dict[system]['m'][0]
            FlowRate_Network = min(max(FlowRate_Network,2.5),5)  # Limit to 15
            Pel      = results_dict[system]['Out1'][0]
            Pel      = min(max(Pel,25),50)  # Limit to 15

            values_dict['FlowRate_Network'] = FlowRate_Network
            values_dict['FlowRate_CHPloop'] = (FlowRate_Network+1)*16.667  # TODO: Check if this works
            values_dict['Pel'] = Pel

        elif system == 'S900':
            FlowRate = results_dict[system]['m'][0]

            values_dict['FlowRate'] = FlowRate
        
        elif system == 'S700_HL1' or system == 'S700_HL2' or system == 'S700_HL3' or system == 'S700_HL4': 
            Tuser = results_dict[system]['T_out'][0]
            Puser  = results_dict[system]['ThermalPower'][0]

            values_dict['Tuser'] = Tuser
            values_dict['Puser'] = Puser


        return values_dict

class OPCUA_SetPoint:
    def __init__(self, **kwargs):
        self.server_url = "opc.tcp://172.25.101.183:4840/freeopcua/server/" 
        self.connected = False
        
        SetPointsDataPath = os.path.join(os.getcwd(),'utils','Communication', 'OPCUA','SetPointsData')
        Opts = {'json_OutputFileName_SetPoints'     : 'SetPoints.json',
                'SetPointsDataPath'                 : SetPointsDataPath}

        self.SetPoints_df = Read_SetPointsJson(Opts)

    def connect(self):
        if not self.connected:
            self.client = Client(self.server_url)
            self.client.connect()
            self.connected = True

    def disconnect(self):
        if self.connected:
            self.client.disconnect()
            self.connected = False

    def write_node(self, node_data):
        for data in node_data:
            node = self.client.get_node(data['NodeID'])
            # Ricava il tipo atteso dal server OPC UA
            expected_uaDataType = node.get_data_type_as_variant_type()
            try:
                val = data['Value']
                # Cast automatico in base al tipo atteso
                if expected_uaDataType.name in ['Float', 'Double']:
                    val = float(val)
                elif expected_uaDataType.name == 'Int32':
                    val = int(val)
                elif expected_uaDataType.name == 'Boolean':
                    if isinstance(val, str):
                        val = val.strip().lower() in ['true', '1']
                    else:
                        val = bool(val)
                value = ua.DataValue(ua.Variant(val, expected_uaDataType))

                node.set_value(value)
            except UaStatusCodeError as e:
                print(f"Error writing value to node {data['NodeID']}: {e}")
                continue
            except Exception as e:
                print(f"Generic error on {data['NodeID']}: {e}")
                continue

    def write_value(self, system, values_dict):
        if system == 'S200':
            df = self.SetPoints_df[system]
            if values_dict['Status'] == 'CHARGE':
                df = df.drop(columns=["VALUE STANDBY", "VALUE DISCHARGE"])
            elif values_dict['Status'] == 'DISCHARGE':
                df = df.drop(columns=["VALUE STANDBY", "VALUE CHARGE"])
            elif values_dict['Status'] == 'STANDBY':
                df = df.drop(columns=["VALUE CHARGE", "VALUE DISCHARGE"])
        else:
            df = self.SetPoints_df[system]
        node_data = []
        for _, row in df.iterrows():
            name = row['Name']
            name = name.replace(' ', '')  # Rimuove gli spazi dal nome
            nodeid = row['NodeID']
            datatype = row['DATA TYPE']
            if name not in values_dict:
                warnings.warn(f"Attenzione: '{name}' non trovato nei risultati per {system}, salto.")
                continue
            value = values_dict[name]
            node_data.append({'NodeID': nodeid, 'Value': value, 'DataType': datatype})
        self.write_node(node_data)


def CollectingResults(solution_dict):
    results_dict = {}

    # Generators
    for name in solution_dict['Generators']:   
        results_dict[name] = solution_dict['Generators'][name]
    
    # HeatingLoads
    for name in solution_dict['HeatingLoads']:
        results_dict[name] = solution_dict['HeatingLoads'][name]
    
    # Storages
    for name in solution_dict['Storages']:
        results_dict[name] = solution_dict['Storages'][name]


    try:
        for name in solution_dict['ElectricalLoads']:
            results_dict[name] = solution_dict['ElectricalLoads'][name]
    except KeyError:
        pass

    return results_dict