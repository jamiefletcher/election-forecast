import csv
import numpy as np
import pandas as pd

# Constants
CENSUS_2013R = "data/CanCensus2021_2013Ridings/98-401-X2021010_English_CSV_data.csv"
ELECTION_2021 = "data/Results/44th_table_tableau12.csv"
ELECTION_2019 = "data/Results/43rd_table_tableau12.csv"
ELECTION_2015 = "data/Results/42nd_table_tableau12.csv"
ZERO_COUNTS = 0.25
BLANK_RECORD = {
    "id": 0 , 
    "year": 0, 
    "winner": "", 
    "LIB": 0, 
    "CON": 0, 
    "NDP": 0, 
    "GRN": 0, 
    "BQ":0, 
    "OTH": 0
}

# Classes
class CensusData:
    def __init__(self, names, fields, data):
        self.names = names
        self.fields = fields
        self.data = data

def make_numeric(str):
    try:
        return int(str)
    except ValueError:
        try:
            return float(str)
        except ValueError:
            return -999

def fields_to_drop(df, zero_thresh = 0.2, missing = -999):
    fields = []
    for col in df.columns:
        vc = df[col].value_counts(normalize=True, sort=True).to_dict()
        if 0 in vc.keys() and vc[0] > zero_thresh:
            fields.append(col)
        if missing in vc.keys():
            fields.append(col)
    return fields

def load_census(filepath, geo_level="Province"):
    name_map = {}
    field_map = {}
    data_map = {}
    curr_id = ""
    with open(filepath, encoding="latin_1") as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        csvreader = csv.DictReader(csvfile, dialect=dialect)
        for row in csvreader:
            id = row["ALT_GEO_CODE"]
            guid = row["DGUID"]
            name = row["GEO_NAME"]
            level = row["GEO_LEVEL"]
            field_id = int(row["CHARACTERISTIC_ID"])
            field = row["CHARACTERISTIC_NAME"].strip()
            value = row["C1_COUNT_TOTAL"]
            if geo_level in level:
                if field_id not in field_map:
                    field_map[field_id] = field
                if curr_id != id:
                    curr_id = id
                    name_map[id] = name
                    data_map[id] = {"id": id, "guid": guid, "name": name}
                data_map[id][field_id] = make_numeric(value)
    # convert to list for Pandas compatibility
    data_list = [record for record in data_map.values()]
    return CensusData(name_map, field_map, data_list)

def prase_party(candidate_name):
    if "Liberal" in candidate_name:
        return "LIB"
    elif "Conservative" in candidate_name:
        return "CON"
    elif "New Democratic Party" in candidate_name:
        return "NDP"
    elif "Green Party" in candidate_name:
        return "GRN"
    elif "Bloc Québécois" in candidate_name:
        return "BQ"
    else:
        return "OTH"

def load_results_t12(filepath, year):
    results = []
    curr_id = ""
    record = BLANK_RECORD.copy()
    with open(filepath) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        csvreader = csv.DictReader(csvfile, dialect=dialect)
        for row in csvreader:
            id = row["Electoral District Number/Numéro de circonscription"]
            party = prase_party(row["Candidate/Candidat"])
            winner = party if row["Majority Percentage/Pourcentage de majorité"] != "" else ""
            pct = float(row["Percentage of Votes Obtained /Pourcentage des votes obtenus"])/100
            if curr_id != id:
                if curr_id != "":
                    results.append(record)                
                curr_id = id
                record = BLANK_RECORD.copy()
                record["id"] = id
                record["year"] = year
            if winner != "":
                record["winner"] = winner
            record[party] += record[party] + pct
        # append final record
        results.append(record) 
    return results

def main():
    print("Reading Census Data ... ", end="", flush=True)
    census_2013_ridings = load_census(CENSUS_2013R, geo_level="Federal electoral district")
    df = pd.DataFrame.from_records(census_2013_ridings.data)
    df.set_index("id", inplace=True)
    print("Done")

    print("Reading Election Data ... ", end="", flush=True)
    df_results_2021 = pd.DataFrame.from_records(data = load_results_t12(ELECTION_2021, year=2021))
    df_results_2021.set_index("id", inplace=True)
    df_results_2021.drop(['year'], axis = 1, inplace = True)
    df_results_2021.rename(
        columns={"winner":"winner-21", "LIB":"LIB-21", "CON":"CON-21", "NDP":"NDP-21", "GRN":"GRN-21", "BQ":"BQ-21", "OTH":"OTH-21"},
        inplace = True
    )

    df_results_2019 = pd.DataFrame.from_records(data = load_results_t12(ELECTION_2019, year=2019))
    df_results_2019.set_index("id", inplace=True)
    df_results_2019.drop(['year'], axis = 1, inplace = True)
    df_results_2019.rename(
        columns={"winner":"winner-19", "LIB":"LIB-19", "CON":"CON-19", "NDP":"NDP-19", "GRN":"GRN-19", "BQ":"BQ-19", "OTH":"OTH-19"},
        inplace = True
    )

    df_results_2015 = pd.DataFrame.from_records(load_results_t12(ELECTION_2015, year=2015))
    df_results_2015.set_index("id", inplace=True)
    df_results_2015.drop(['year'], axis = 1, inplace = True)
    df_results_2015.rename(
        columns={"winner":"winner-15", "LIB":"LIB-15", "CON":"CON-15", "NDP":"NDP-15", "GRN":"GRN-15", "BQ":"BQ-15", "OTH":"OTH-15"},
        inplace = True
    )
    print("Done")

    print(df_results_2021)
    print(df_results_2019)
    print(df_results_2015)

    # print(df.shape)
    print(f"Dropping census features with missing data ... ", end="")
    df.drop(fields_to_drop(df, zero_thresh=ZERO_COUNTS), axis = 1, inplace=True)
    # print(df.shape)
    print("Done")

    # df.info()
    print(df)



if __name__ == "__main__":
    main()