import csv
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Constants
CENSUS_2013R = "data/CanCensus2021_2013Ridings/98-401-X2021010_English_CSV_data.csv"
ELECTION_2021_T12 = "data/Results/44th_table_tableau12.csv"
ELECTION_2019_T12 = "data/Results/43rd_table_tableau12.csv"
ELECTION_2015_T12 = "data/Results/42nd_table_tableau12.csv"
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

def prepare_census():
    print("Reading Census Data ... ", end="", flush=True)
    census_2013_ridings = load_census(CENSUS_2013R, geo_level="Federal electoral district")
    df_census = pd.DataFrame.from_records(census_2013_ridings.data)
    df_census.columns = df_census.columns.astype(str)
    print("Done")

    print(f"Dropping census features with missing data ... ", end="")
    df_census.drop(fields_to_drop(df_census, zero_thresh=ZERO_COUNTS), axis = 1, inplace=True)
    print("Done")

    print(f"Dropping guid and name features ... ", end="")
    df_census.drop(["guid", "name"], axis = 1, inplace = True)
    print("Done")

    print(f"Rescale census data by Z-score ... ", end="")
    num_cols = df_census.select_dtypes(include='number').columns.to_list()
    num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    preprocessing = ColumnTransformer(
        [('num', num_pipeline, num_cols)], remainder='passthrough'
    )
    df_census_scaled = preprocessing.fit_transform(df_census)
    # Convert back to pd.DataFrame
    feature_names = preprocessing.get_feature_names_out()
    id_idx = np.argwhere(feature_names == "remainder__id")[0]   # id column needs to keep its name
    feature_names[id_idx] = "id"
    df_census_scaled = pd.DataFrame(data=df_census_scaled, columns=feature_names)
    print("Done")

    return df_census_scaled

def prepare_results():
    print("Reading Election Data ... ", end="", flush=True)    
    years = {
        2021: ELECTION_2021_T12,
        2019: ELECTION_2019_T12,
        2015: ELECTION_2015_T12
    }
    dfs = {}    
    for year, path in years.items():
        df = pd.DataFrame.from_records(data = load_results_t12(path, year))
        df.drop(["year"], axis = 1, inplace = True)
        dfs[year] = df
    print("Done")
    return dfs

def main():
    df_census = prepare_census()
    print(df_census.info())

    df_results = prepare_results()
    print(df_results[2021].info())
    print(df_results[2019].info())
    print(df_results[2015].info())

    # predict winner of next election based on previous election results + census
    Xy_2015 = pd.merge(
        df_results[2015].drop(["winner"], axis=1),  # drop 2015 winner category
        df_results[2019][["id", "winner"]],         # merge 2019 winner as target
        on="id"
    )
    Xy_2019 = pd.merge(
        df_results[2019].drop(["winner"], axis=1),  # drop 2019 winner category
        df_results[2021][["id", "winner"]],         # merge 2021 winner as target
        on="id"
    )
    Xy = pd.concat([Xy_2015, Xy_2019], ignore_index=True)
    Xy = pd.merge(Xy, df_census, on="id")

    ids = Xy["id"]
    y = Xy["winner"]
    X = Xy.drop(["id", "winner"], axis = 1)

    print(ids)
    print(y)
    print(X)

if __name__ == "__main__":
    main()
