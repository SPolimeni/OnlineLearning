import os
import platform
os_name = platform.system()
import numpy as np
import pandas as pd
import re
import json
import copy
import time
import inspect

from datetime import datetime as dt
import pytz
from opcua                  import Client, ua
from opcua.ua.uaerrors      import UaStatusCodeError
from io import StringIO

import influxdb_client
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

import concurrent.futures



##################
# InfluxDB Class #
##################

class InfluxDB_connect:
   def __init__(self,**kwargs):
      self.token  = "btjVn84rLctD0R_2WjhUxPKUeOawZFuxRhG4bqbUCsI45RTqYb7q_8-Ib_Q7SQMLycyVXipygmtkHklrL1Q5qg==" #"7bIWbhcIEs891yPyNnYDcRP-33aWUYFvbBgYiu770r-nGzq0agBCdhnMWjCU5Kfx54B6bdqsJ6xpdWmgNwj5Yg==" #
      self.org    = "RSE"
      self.url    = "http://172.25.102.65:8086"
      self.Bucket = "raw_data"

      self.InfluxDBclient  = InfluxDBClient(url=self.url, token=self.token, org=self.org)
      self.WriteAPI        = self.InfluxDBclient.write_api(write_options=SYNCHRONOUS)

      self.DataToWrite           = []
      self.ExecutedWriting       = 0

      # List of all nodes to read and their corresponding metadata
      ConfigDataPath = os.path.join(os.getcwd(),'utils','InfluxDB','ConfigData')
      FileName       = 'DBdataConfig.json'
      with open(os.path.join(ConfigDataPath,FileName), 'r') as file:
         DBdataConfig = json.load(file)

      Zones    = []
      Classes  = []
      for key in DBdataConfig.keys():
         for subkey in DBdataConfig[key].keys():
            Zones.append(DBdataConfig[key][subkey]['Zone'])
            Classes.append(DBdataConfig[key][subkey]['Class'])

      Zones    = list(set(Zones))
      Classes = list(set(Classes))

      self.metadata = {'zones':Zones, 'classes':Classes}


   def write_data(self):
      self.ExecutedWriting += 1
      print(f"Writing data - attempt # {self.ExecutedWriting}")

      Data = self.DataToWrite
   
      for key in Data.keys():
         time_start = time.time()
         Points = self.define_points(Data[key])

         try:
            self.WriteAPI.write(bucket=self.Bucket, org=self.org, record=Points)

         except Exception as e:
            error_msg = f"Error writing data to InfluxDB: {e} (Line {inspect.currentframe().f_lineno} in InfluxDB_mgmt.py)"
            print(error_msg)
            LogError(error_msg)
         
         time_stop = time.time()
         print(f"Time to write all points in {key}: {time_stop - time_start}")


   def define_points(self, DataKey):
      Points = []
      for subKey in DataKey.keys():
         for i in range(len(DataKey[subKey]['Value'])):
            Name = DataKey[subKey]['Tag']

            point = influxdb_client.Point(Name).field("value", DataKey[subKey]['Value'][i]).time(DataKey[subKey]['Time'][i])

            for tag_key, tag_value in DataKey[subKey].items():
               if tag_key != 'Value' and tag_key != 'Time' and tag_key != 'nomeFRER':
                  point.tag(tag_key, tag_value)

            Points.append(point)

      return Points

   
   def shutdown(self):
      self.InfluxDBclient.close()
      

#################
# OPCUA Classes #
#################

class OPCUA_connect:
   def __init__(self,**kwargs):

      self.server_url = "opc.tcp://172.25.101.183:4840/freeopcua/server/" 
      self.connected  = False

      # OPC UA Client reconnection parameters
      self.MaxRetries = 5
      self.RetryDelay = 5
      self.BufferSize = 20

      # Aux
      self.ExecutedReadings   = 0
      self.verbose            = 0

      # List of all nodes to read and their corresponding metadata
      ConfigDataPath = os.path.join(os.getcwd(),'utils','InfluxDB','ConfigData')
      FileName  = 'DBdataConfig.json'
      with open(os.path.join(ConfigDataPath,FileName), 'r') as file:
         self.DBdataConfig = json.load(file)

      # Dictionary with blank for values to be read (buffer?)
      self.DBdataRead = copy.deepcopy(self.DBdataConfig)

      for key in self.DBdataRead.keys():
         for subKey in self.DBdataRead[key].keys():
               self.DBdataRead[key][subKey]['Value'] = [] # empty list for values to be read and buffered
               self.DBdataRead[key][subKey]['Time']  = [] # empty list for timestamps related to the values

      csv_list = ['Pressure','Temperature','Flow','Power','Valve']  # TODO: Add FRER quantities
      self.DBdata_csv = {key: None for key in csv_list}

      self.DBtoCSVmap = {}

      for csv_key in csv_list:
         TagList = []
         for DB_key in self.DBdataRead.keys():
            for subKey in self.DBdataRead[DB_key].keys():
               if csv_key in self.DBdataRead[DB_key][subKey]['Type'] and self.DBdataRead[DB_key][subKey]['Class'] != 'Meter':
                  
                  Tag = self.DBdataRead[DB_key][subKey]['Tag']
                  self.DBtoCSVmap[Tag] = {'Key' : DB_key, 'nodeID' : subKey}
                  TagList.append(Tag)

         self.DBdata_csv[csv_key] = pd.DataFrame(columns=TagList)

      self.csv_data_path = os.path.join(os.getcwd(),'Acquistions')

   def check_connection(self):                          
      if self.client.get_endpoints():
         print("Client is connected to the server.")
         self.connected = True
      else:
         print("Failed to retrieve server endpoints, client may not be connected.")
         self.connected = False

   def connect(self): # Connecting to the OPC UA Server
      if not self.connected:
            try:
               self.client = Client(self.server_url)
               self.client.connect()
               self.check_connection()
            except Exception as e:
               error_msg = f"Error connecting to OPC UA Server: {e} (Line {inspect.currentframe().f_lineno} in InfluxDB_mgmt.py)"
               print(error_msg)
               LogError(error_msg)
                              
               self.connected = False

   def reconnect(self):
      """Attempts to reconnect to the OPC UA server with retries."""
      for attempt in range(self.MaxRetries):
         try:
               print(f"Reconnecting attempt {attempt + 1}/{self.MaxRetries}...")
               self.client.connect()
               print("Reconnected successfully.")
               return True
         except Exception as e:
               error_msg = f"Reconnect attempt {attempt + 1} failed: {e} (Line {inspect.currentframe().f_lineno} in InfluxDB_mgmt.py)"
               print(error_msg)
               LogError(error_msg)

               time.sleep(self.RetryDelay)
      print("Exceeded maximum retries. Could not reconnect.")
      return False
   
   def disconnect(self): # Disconnecting from the OPC UA Server
      if self.connected or self.client.uaclient.is_connected():
            self.client.disconnect()
            print("Disconnected from OPC UA Server")
            self.connected = False

   def read_nodes(self):
      self.ExecutedReadings += 1

      # Check ping and reconnect if needed
      if not self.ping_server():
         print("OPC UA ping failed. Trying to reconnect.")
         LogError("OPC UA ping failed. Trying to reconnect.")
         self.connected = False
         if not self.reconnect():
               LogError("Failed to reconnect. Skipping read.")
               return None

      if self.verbose > 0:
         print(f"Reading nodes - attempt # {self.ExecutedReadings}")

      try:
         for key in self.DBdataRead.keys():
               start_time = time.time()

               if key == 'CHP':
                  for subKey in self.DBdataRead[key].keys():
                     try:
                           node_id = subKey
                           if 'ADAM' in node_id:
                              adam_pos = node_id.find('ADAM')
                              NodeName = node_id[:adam_pos + len('ADAM') + 1]
                              node = self.client.get_node(NodeName)
                              idx = int(node_id[adam_pos + len('ADAM') + 1:].strip('()'))
                              value = node.get_value()[idx]
                           else:
                              value = self.client.get_node(node_id).get_value()

                           timestamp = time.time_ns()
                           self._update_buffer(key, subKey, value, timestamp)

                     except Exception as e:
                           LogError(f"Error reading {subKey}: {e}")
                           continue

               else:
                  node_ids = list(self.DBdataRead[key].keys())
                  with concurrent.futures.ThreadPoolExecutor() as executor:
                     futures = [executor.submit(self.read_node_value, nid) for nid in node_ids]
                     results = []
                     for f in futures:
                           try:
                              results.append(f.result(timeout=2))
                           except Exception as e:
                              results.append(None)
                              LogError(f"Timeout reading node: {e}")

                  timestamp = time.time_ns()
                  for subKey, value in zip(node_ids, results):
                     if value is not None:
                           self._update_buffer(key, subKey, value, timestamp)

               if self.verbose > 0:
                  print(f"Finished reading group {key} in {time.time() - start_time:.2f} sec")

         return self.DBdataRead

      except Exception as e:
         LogError(f"Unhandled error during read_nodes: {e}")
         return None

   def _update_buffer(self, key, subKey, value, timestamp):
      buf = self.DBdataRead[key][subKey]
      if len(buf['Value']) >= self.BufferSize:
         index = self.ExecutedReadings % self.BufferSize
         buf['Value'][index] = value
         buf['Time'][index] = timestamp
      else:
         buf['Value'].append(value)
         buf['Time'].append(timestamp)

   def ping_server(self):
      try:
         self.client.get_endpoints()
         return True
      except Exception:
         return False
   
   def read_node_value(self,node_id):
      node = self.client.get_node(node_id)
      return node.get_value()

   def populate_csv(self):

      for quantity in self.DBdata_csv.keys():
         TimeIndex      = None
         QuantityPass   = 0
         for column in self.DBdata_csv[quantity].columns:
            key      = self.DBtoCSVmap[column]['Key']
            nodeID   = self.DBtoCSVmap[column]['nodeID']

            Value    = self.DBdataRead[key][nodeID]['Value']
            Time_ns  = self.DBdataRead[key][nodeID]['Time']

            if TimeIndex is None:
               timestamp_sec  = [x / 1e9 for x in Time_ns]
               dt_object      = [dt.utcfromtimestamp(x).replace(microsecond=0) for x in timestamp_sec]
               rome_tz        = pytz.timezone('Europe/Rome')
               dt_object_rome = [pytz.utc.localize(x).astimezone(rome_tz) for x in dt_object]


            df_col         = pd.DataFrame({column: Value})
            df_col.index   = dt_object_rome

            self.DBdata_csv[quantity][column] = df_col      


   def save_csv(self):
      
      self.populate_csv()
      
      datetime = dt.now()
      date     = datetime.strftime('%Y-%m-%d')

      csv_path = os.path.join(self.csv_data_path,f'Data_{date}')

      if not(os.path.exists(csv_path)):
         os.makedirs(csv_path)

      for key in self.DBdata_csv.keys():
         csv_file = os.path.join(csv_path,f'{key}.csv')
         if os.path.isfile(csv_file):
            previous_data = pd.read_csv(csv_file, index_col=0)
            new_data = pd.concat([previous_data,self.DBdata_csv[key]])
            new_data.to_csv(csv_file)
         else:
            self.DBdata_csv[key].to_csv(csv_file)

      

#######################
# Auxiliary Functions #
#######################

def LogError(error_msg):
   LogDataPath = os.path.join(os.getcwd(),'utils','InfluxDB','Logs')
   Date = dt.now()
   timezone = pytz.timezone('Europe/Rome')
   Date_tz = timezone.localize(Date)
   FileName    = 'ErrorLog_'+str(Date_tz.strftime('%Y-%m-%d')) + '.txt'

   file_path = os.path.join(LogDataPath,FileName)
   if not os.path.exists(file_path):
      with open(file_path, 'w') as file:
         pass

   error_msg = Date_tz.strftime('%Y-%m-%d %H:%M:%S') + " - " + error_msg

   with open(file_path, 'a') as file:
      file.write(error_msg + '\n')



        