import csv

# Constants
census_2013r = "data/CanCensus2021_2013Ridings/98-401-X2021010_English_CSV_data.csv"
# census_2013r = "data/CanCensus2021_2013Ridings/tmp.csv"

def make_numeric(str):
    try:
        return int(str)
    except ValueError:
        try:
            return float(str)
        except ValueError:
            return -999

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
                    data_map[id] = {"name" : name}
                data_map[id][field_id] = make_numeric(value)
    return {"names": name_map, "fields": field_map, "data": data_map}

def main():
    print("Reading Census Data ... ", end="", flush=True)
    census_2013_ridings = load_census_data(census_2013r, geo_level="Federal electoral district")
    print("Done")

if __name__ == "__main__":
    main()