import os
import json
import pandas as pd
from opcua import Client, ua
from opcua.ua.uaerrors import UaStatusCodeError
import io
import warnings

class OPCUA_SetPoint:
    def __init__(self, **kwargs):
        self.server_url = "opc.tcp://172.25.101.183:4840/freeopcua/server/" 
        self.connected = False
        self.connect()

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

