import enum


class ScalingOp(enum.Enum):
    mul = 1
    div = 2


PARTIES = {
    "LIB": "Liberal",
    "CON": "Conservative",
    "NDP": "New Democratic Party",
    "BQ": "Bloc Québécois",
}


def scale_df(df, scaling_factors, op=ScalingOp.mul):
    cat_cols = df.select_dtypes(exclude="number").columns.to_list()
    df_cat = df[cat_cols]
    df_num = df.drop(cat_cols, axis=1)
    if op == ScalingOp.mul:
        df_num *= scaling_factors
    elif op == ScalingOp.div:
        df_num /= scaling_factors
    else:
        raise ValueError("Unexpected scaling operation")
    return df_num.join(df_cat)


def make_numeric(string):
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            return -999


def parse_party(candidate_name):
    for party_id, party_name in PARTIES.items():
        if party_name in candidate_name:
            return party_id
    return "OTH"


def fix_other(df):
    df.drop(["OTH"], axis=1, inplace=True)
    df["OTH"] = 1.0 - df["LIB"] - df["CON"] - df["NDP"] - df["GRN"] - df["BQ"]
    df["OTH"] = df["OTH"].clip(lower=0)
    return df
