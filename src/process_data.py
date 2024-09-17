import csv

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import ScalingOp, make_numeric, scale_df, parse_party, fix_other


class CensusData:
    def __init__(self, names, fields, data):
        self.names = names
        self.fields = fields
        self.data = data


# Constants
BLANK_RECORD = {"id": "", "LIB": 0, "CON": 0, "NDP": 0, "GRN": 0, "BQ": 0, "OTH": 0}


# Table 12 - Local Results
T12_RIDING = "Electoral District Number/Numéro de circonscription"
T12_CANDIDATE = "Candidate/Candidat"
T12_WIN_MARGIN = "Majority Percentage/Pourcentage de majorité"
T12_VOTE_PCT = "Percentage of Votes Obtained /Pourcentage des votes obtenus"

# Table 9 - National Results
T9_PARTY = "\ufeffPolitical affiliation/Appartenance politique"
T9_VOTE_PCT = "Total"

ZERO_COUNT_THRESH = 0.25


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
            riding_id = row["ALT_GEO_CODE"]
            guid = row["DGUID"]
            name = row["GEO_NAME"]
            level = row["GEO_LEVEL"]
            field_id = int(row["CHARACTERISTIC_ID"])
            field = row["CHARACTERISTIC_NAME"].strip()
            value = row["C1_COUNT_TOTAL"]
            if geo_level in level:
                if field_id not in field_map:
                    field_map[field_id] = field
                if curr_id != riding_id:
                    curr_id = riding_id
                    name_map[riding_id] = name
                    data_map[riding_id] = {"id": riding_id, "guid": guid, "name": name}
                data_map[riding_id][field_id] = make_numeric(value)
    # convert to list for Pandas compatibility
    data_list = [record for record in data_map.values()]
    return CensusData(name_map, field_map, data_list)


def load_results_t12(filepath):
    results = []
    curr_id = ""
    record = BLANK_RECORD.copy()
    with open(filepath, encoding="utf-8") as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        csvreader = csv.DictReader(csvfile, dialect=dialect)
        for row in csvreader:
            riding_id = row[T12_RIDING]
            party = parse_party(row[T12_CANDIDATE])
            winner = party if row[T12_WIN_MARGIN] != "" else ""
            pct = float(row[T12_VOTE_PCT]) / 100
            if curr_id != riding_id:
                if curr_id != "":
                    results.append(record)
                curr_id = riding_id
                record = BLANK_RECORD.copy()
                record["id"] = riding_id
            if winner != "":
                record["winner"] = winner
            record[party] += pct
        # append final record
        results.append(record)
    return results


def load_results_t9(filepath):
    record = BLANK_RECORD.copy()
    # not needed at national level
    record.pop("id")
    with open(filepath, encoding="utf-8") as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        csvreader = csv.DictReader(csvfile, dialect=dialect)
        for row in csvreader:
            party = parse_party(row[T9_PARTY])
            pct = float(row[T9_VOTE_PCT]) / 100
            record[party] += pct
    return record


def fields_to_drop(df, zero_thresh=0.2, missing=-999):
    fields = []
    for col in df.columns:
        vc = df[col].value_counts(normalize=True, sort=True).to_dict()
        if 0 in vc.keys() and vc[0] > zero_thresh:
            fields.append(col)
        if missing in vc.keys():
            fields.append(col)
    return fields


def prepare_census(input_file):
    print("- Read census data ...", flush=True)
    census_2013_ridings = load_census(
        input_file, geo_level="Federal electoral district"
    )
    df_census = pd.DataFrame.from_records(census_2013_ridings.data)
    df_census.columns = df_census.columns.astype(str)

    print("- Drop census features with missing data ...")
    df_census.drop(
        fields_to_drop(df_census, zero_thresh=ZERO_COUNT_THRESH), axis=1, inplace=True
    )

    print("- Drop guid and name features ...")
    df_census.drop(["guid", "name"], axis=1, inplace=True)

    print("- Rescale census data by Z-score ...")
    num_cols = df_census.select_dtypes(include="number").columns.to_list()
    num_pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
    preprocessing = ColumnTransformer(
        [("num", num_pipeline, num_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    df_census_scaled = pd.DataFrame(
        data=preprocessing.fit_transform(df_census),
        columns=preprocessing.get_feature_names_out(),
    )
    return df_census_scaled


def prepare_elections(riding_files, national_files):
    riding_results = {}
    riding_results_scaled = {}
    national_results = {}
    print("- Read local election results ...")
    for year, path in riding_files.items():
        df = pd.DataFrame.from_records(data=load_results_t12(path))
        riding_results[year] = df

    print("- Read national election results ...")
    for year, path in national_files.items():
        national_results[year] = load_results_t9(path)

    print("- Rescale local election results ...")
    for year in riding_files.keys():
        riding_results_scaled[year] = scale_df(
            riding_results[year], national_results[year], ScalingOp.div
        )
    return riding_results_scaled, national_results


def merge_dfs(df_census, df_ridings, df_national, target_class, merge_class):
    Xy = []

    print("- Scale each set of local results by national totals ...")
    for year_target, df_target in df_ridings.items():
        # select winner as target
        target = df_target[[merge_class, target_class]]
        
        for df in df_ridings.values():
            # drop existing winner cat
            tmp = df.drop([target_class], axis=1)
            # scale riding back up by target national result
            tmp = scale_df(tmp, df_national[year_target], ScalingOp.mul)
            # re-compute "OTH" so totals sum to 1.0
            tmp = fix_other(tmp)
            # merge target winner cat
            Xy.append(pd.merge(tmp, target, on=merge_class))

    # concat all individual Xy into a single df and merge with census
    Xy = pd.concat(Xy, ignore_index=True)

    print("- Merge election results with census data ...")
    Xy = pd.merge(Xy, df_census, on="id")
    return Xy
