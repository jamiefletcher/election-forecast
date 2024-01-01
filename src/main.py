import csv
import enum

# import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Filepaths
CENSUS_2013R = "data/CanCensus2021_2013Ridings/98-401-X2021010_English_CSV_data.csv"
ELECTION_2021_T12 = "data/Results/44th_table_tableau12.csv"  # riding results
ELECTION_2019_T12 = "data/Results/43rd_table_tableau12.csv"
ELECTION_2015_T12 = "data/Results/42nd_table_tableau12.csv"
ELECTION_2021_T9 = "data/Results/44th_table_tableau09.csv"  # national totals
ELECTION_2019_T9 = "data/Results/43rd_table_tableau09.csv"
ELECTION_2015_T9 = "data/Results/42nd_table_tableau09.csv"


# Constants
T12_RIDING = "Electoral District Number/Numéro de circonscription"
T12_CANDIDATE = "Candidate/Candidat"
T12_WIN_MARGIN = "Majority Percentage/Pourcentage de majorité"
T12_VOTE_PCT = "Percentage of Votes Obtained /Pourcentage des votes obtenus"
T9_PARTY = "\ufeffPolitical affiliation/Appartenance politique"
T9_VOTE_PCT = "Total"
ZERO_COUNTS = 0.25
BLANK_RECORD = {"id": "", "LIB": 0, "CON": 0, "NDP": 0, "GRN": 0, "BQ": 0, "OTH": 0}


# Classes
class CensusData:
    def __init__(self, names, fields, data):
        self.names = names
        self.fields = fields
        self.data = data


class Op(enum.Enum):
    mul = 1
    div = 2


def make_numeric(string):
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            return -999


def fields_to_drop(df, zero_thresh=0.2, missing=-999):
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
            party = prase_party(row[T12_CANDIDATE])
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
            party = prase_party(row[T9_PARTY])
            pct = float(row[T9_VOTE_PCT]) / 100
            record[party] += pct
    return record


def prepare_census():
    print("Reading census data ... ", end="", flush=True)
    census_2013_ridings = load_census(
        CENSUS_2013R, geo_level="Federal electoral district"
    )
    df_census = pd.DataFrame.from_records(census_2013_ridings.data)
    df_census.columns = df_census.columns.astype(str)
    print("Done")

    print("Dropping census features with missing data ... ", end="")
    df_census.drop(
        fields_to_drop(df_census, zero_thresh=ZERO_COUNTS), axis=1, inplace=True
    )
    print("Done")

    print("Dropping guid and name features ... ", end="")
    df_census.drop(["guid", "name"], axis=1, inplace=True)
    print("Done")

    print("Rescale census data by Z-score ... ", end="")
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
    print("Done")
    return df_census_scaled


def scale_df(df, scaling_factors, op=Op.mul):
    cat_cols = df.select_dtypes(exclude="number").columns.to_list()
    df_cat = df[cat_cols]
    df_num = df.drop(cat_cols, axis=1)
    if op == Op.mul:
        df_num *= scaling_factors
    elif op == Op.div:
        df_num /= scaling_factors
    else:
        raise ValueError("Unexpected scaling operation")
    return df_num.join(df_cat)


def fix_other(df):
    df.drop(["OTH"], axis=1, inplace=True)
    df["OTH"] = 1.0 - df["LIB"] - df["CON"] - df["NDP"] - df["GRN"] - df["BQ"]
    df["OTH"] = df["OTH"].clip(lower=0)
    return df


def prepare_elections():
    riding_result_files = {
        2021: ELECTION_2021_T12,
        2019: ELECTION_2019_T12,
        2015: ELECTION_2015_T12,
    }
    national_result_files = {
        2021: ELECTION_2021_T9,
        2019: ELECTION_2019_T9,
        2015: ELECTION_2015_T9,
    }
    riding_results = {}
    riding_results_scaled = {}
    national_results = {}
    print("Reading election data - riding level ... ", end="", flush=True)
    for year, path in riding_result_files.items():
        df = pd.DataFrame.from_records(data=load_results_t12(path))
        riding_results[year] = df
    print("Done")
    print("Reading election data - national level ... ", end="", flush=True)
    for year, path in national_result_files.items():
        national_results[year] = load_results_t9(path)
    print("Done")
    print("Rescale riding level data ... ", end="", flush=True)
    for year in riding_result_files.keys():
        riding_results_scaled[year] = scale_df(
            riding_results[year], national_results[year], Op.div
        )
    print("Done")
    return riding_results_scaled, national_results


def make_xy(df_census, df_elections, target_class, merge_class):
    Xy = []
    riding_results, national_results = df_elections
    for year_target, df_target in riding_results.items():
        # select winner as target
        target = df_target[[merge_class, target_class]]
        # non-target dfs
        df_feat = [v for k, v in riding_results.items() if k != year_target]
        for df in df_feat:
            # drop existing winner cat
            tmp = df.drop([target_class], axis=1)
            # scale riding back up by target national result
            tmp = scale_df(tmp, national_results[year_target], Op.mul)
            # re-compute "OTH" so totals sum to 1.0
            tmp = fix_other(tmp)
            # merge target winner cat
            Xy.append(pd.merge(tmp, target, on=merge_class))

    # concat all individual Xy into a single df and merge with census
    Xy = pd.concat(Xy, ignore_index=True)
    Xy = pd.merge(Xy, df_census, on="id")
    return Xy["id"], Xy.drop(["id", "winner"], axis=1), Xy["winner"]


def feature_select(X, y, random=None):
    n_trees = 10
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=random)
    select = SelectFromModel(estimator=rf).fit(X, y)
    return pd.DataFrame(
        data=select.transform(X), columns=select.get_feature_names_out()
    )


def main():
    df_census = prepare_census()
    df_elections = prepare_elections()
    _, X, y = make_xy(df_census, df_elections, target_class="winner", merge_class="id")
    # print(ids)
    print(y.value_counts())
    print(X.shape)
    X_select = feature_select(X, y)
    print(X_select.shape)
    print(X_select.head())


if __name__ == "__main__":
    main()
