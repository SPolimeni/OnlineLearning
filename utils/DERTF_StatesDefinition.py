import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from utils.InfluxDB.InfluxDB_mgmt import InfluxDB_connect
import casadi as ca

def SettingUpInfluxDBconnection():
    InfluxDB_connector = InfluxDB_connect()
    return InfluxDB_connector

def CheckInfluxDBconnection(InfluxDB_connector):
    HealthCheck = False
    try:
        health = InfluxDB_connector.InfluxDBclient.health()
        if health.status == "pass":
            HealthCheck = True
            print("InfluxDB is healthy and connected.")
        else:
            print(f"InfluxDB connection issue: {health}")
    except Exception as e:
        print(f"Error connecting to InfluxDB: {e}")
    return HealthCheck

def SimpleStatusReadings(InfluxDB_connector):
    """
    Legge Tin, Tout, FlowRate, Power per ciascun sistema da InfluxDB.
    Tin = TT*1, Tout = TT*2, FlowRate = FT*1, Power = Power*
    Restituisce: {sistema: {'T_in_sol': val, 'T_out_sol': val, 'm_sol': val, 'Power': val}}
    """
    Bucket = InfluxDB_connector.Bucket
    timeRead = f"-{5}m"
    timeAggr = f"{5}m"
    query_api = InfluxDB_connector.InfluxDBclient.query_api()

    try:
        query_Treturn = f'''
            from(bucket: "{Bucket}")
                |> range(start: {timeRead})
                |> filter(fn: (r) => r["Tag"] == "TT901")
                |> aggregateWindow(every: {timeAggr}, fn: mean, createEmpty: false)
        '''
        tables_Tret= query_api.query(query_Treturn)
        try:
            Tret = [record.values['_value'] for table in tables_Tret for record in table.records][-1]
        except IndexError:
            Tret = 60
    except Exception as e:
        Tret = 60


    try:
        query_Tdelivery = f'''
            from(bucket: "{Bucket}")
                |> range(start: {timeRead})
                |> filter(fn: (r) => r["Tag"] == "TT732")
                |> aggregateWindow(every: {timeAggr}, fn: mean, createEmpty: false)
        '''
        tables_Tdel = query_api.query(query_Tdelivery)
        try:
            Tdel = [record.values['_value'] for table in tables_Tdel for record in table.records][-1]  
        except IndexError:
            Tdel = 70
    except Exception as e:
        Tdel = 70

    try:
        query_Flow = f'''
            from(bucket: "{Bucket}")
                |> range(start: {timeRead})
                |> filter(fn: (r) => r["Tag"] == "FT901")
                |> aggregateWindow(every: {timeAggr}, fn: mean, createEmpty: false)
        '''
        tables_Flow = query_api.query(query_Flow)
        try:
            Flow = [record.values['_value'] for table in tables_Flow for record in table.records][-1]
        except IndexError:
            Flow = 10
    except Exception as e:
        Flow = 10

    y_dict = {
        'T_return'  : Tret,
        'T_delivery': Tdel,
        'flow'      : Flow/3.6,
    }

    return y_dict

def main_status_readings():
    InfluxDB_connector = SettingUpInfluxDBconnection()

    if not CheckInfluxDBconnection(InfluxDB_connector):
        print("Connessione a InfluxDB fallita.")
        exit()

    y_dict = SimpleStatusReadings(InfluxDB_connector)

    return y_dict



if __name__ == "__main__":
    InfluxDB_connector = SettingUpInfluxDBconnection()

    if not CheckInfluxDBconnection(InfluxDB_connector):
        print("Connessione a InfluxDB fallita.")
        exit()

    y_dict = SimpleStatusReadings(InfluxDB_connector)

    stop = 1
