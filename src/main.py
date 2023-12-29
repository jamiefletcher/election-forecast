import csv
import numpy as np
import pandas as pd

# Constants
CENSUS_2013R = "data/CanCensus2021_2013Ridings/98-401-X2021010_English_CSV_data.csv"
ZERO_COUNTS = 0.25

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

def load_census_data(filepath, geo_level="Province"):
    name_map = {}
    field_map = {}
    data_map = {}
    curr_id = ""
    with open(filepath, encoding="latin_1") as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        csvreader = csv.DictReader(csvfile, dialect=dialect)
        for row in csvreader:
            id = row["DGUID"]
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
                    data_map[id] = {"id" : id, "name" : name}
                data_map[id][field_id] = make_numeric(value)
    # convert to list for Pandas compatibility
    data_list = [record for record in data_map.values()]
    return CensusData(name_map, field_map, data_list)

def main():
    print("Reading Census Data ... ", end="", flush=True)
    census_2013_ridings = load_census_data(CENSUS_2013R, geo_level="Federal electoral district")
    print("Done")

    df = pd.DataFrame.from_records(census_2013_ridings.data)
    df.set_index("id", inplace=True)
    # print(df.shape)
    print(f"Dropping features with >{ZERO_COUNTS * 100}% zero counts ... ", end="")
    df.drop(fields_to_drop(df, zero_thresh=ZERO_COUNTS), axis = 1, inplace=True)
    # print(df.shape)
    print("Done")

    # df.info()
    print(df.head())



if __name__ == "__main__":
    main()