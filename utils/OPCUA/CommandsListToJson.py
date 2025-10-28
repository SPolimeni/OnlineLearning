import os
import pandas as pd
import json

def ExcelToSimpleJson(excel_path, json_path):
    # Leggi tutti gli sheet
    xls = pd.read_excel(excel_path, sheet_name=None)
    json_dict = {}
    for sheet, df in xls.items():
        # Colonne minime richieste
        colonne_base = ['Name', 'NodeID', 'DATA TYPE']
        # Colonne opzionali
        colonne_extra = ['VALUE STANDBY', 'VALUE CHARGE', 'VALUE DISCHARGE']
        # Seleziona solo le colonne presenti
        colonne_presenti = [col for col in colonne_base + colonne_extra if col in df.columns]
        df = df[colonne_presenti].copy()
        # Prefisso NodeID
        prefix = 'ns=4;s='
        if 'NodeID' in df.columns:
            df['NodeID'] = df['NodeID'].apply(lambda x: x if str(x).startswith(prefix) else prefix + str(x))
        # Salva come stringa JSON
        json_dict[sheet] = df.to_json(orient='records')
    # Salva il file JSON
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=2)

def ExcelToJson_prova_carichi(EXCEL_PATH, JSON_PATH):
    # Leggi il foglio Values
    df_values = pd.read_excel(EXCEL_PATH, sheet_name='Values', header=[0, 1])
    # Leggi il foglio NodeOPC
    df_nodes = pd.read_excel(EXCEL_PATH, sheet_name='NodeOPC')

    # Dizionario dei valori
    data_dict = {}
    for idx, row in df_values.iterrows():
        timestep = row[('Timestep', 'Timestep')]
        data_dict[timestep] = {}
        for col in df_values.columns[1:]:
            system, var = col
            value = row[col]
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
            if system not in data_dict[timestep]:
                data_dict[timestep][system] = {}
            data_dict[timestep][system][var] = value

    # Dizionario dei NodeID
    node_map = {
        f"{str(row['System'])}|{str(row['Name'])}": f"ns=4;s={str(row['NodeID']).strip()}"
        for _, row in df_nodes.iterrows()
    }

    # Salva entrambi nel JSON
    # Convert np.int64 keys in data_dict to int
    data_dict_int_keys = {int(k): v for k, v in data_dict.items()}
    out = {
        "values": data_dict_int_keys,
        "node_map": node_map
    }
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"File JSON salvato in: {JSON_PATH}")

if __name__ == "__main__":
    # excel_path = os.path.join(os.getcwd(), 'utils', 'Communication', 'OPCUA', 'SetPointsData', 'MappaComandiTF_EMS_Centralizzato_NoOnOff.xlsx')
    # json_path = os.path.join(os.getcwd(), 'utils', 'Communication', 'OPCUA', 'SetPointsData', 'SetPoints.json')
    # ExcelToSimpleJson(excel_path, json_path)
    EXCEL_PATH = os.path.join(os.getcwd(), 'utils', 'Communication', 'OPCUA', 'SetPointsData', 'Setpoints_prova_carichi_OnlineLearning.xlsx')
    JSON_PATH = os.path.join(os.getcwd(), 'utils', 'Communication', 'OPCUA', 'SetPointsData', 'Setpoints_prova_carichi_OnlineLearning.json')
    ExcelToJson_prova_carichi(EXCEL_PATH, JSON_PATH)
    print(f"Creato il file JSON: {JSON_PATH}")