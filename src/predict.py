import warnings

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils import ScalingOp, fix_other, scale_df

# Disable annoying warning -- see: https://stackoverflow.com/a/76306267
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

wiki_table_attrs = {"class": "wikitable"}
wiki_table_cols = {"CPC": "CON", "LPC": "LIB", "NDP": "NDP", "BQ": "BQ", "GPC": "GRN"}


def scale_polling(df_census, df_ridings, df_polls, target_class, merge_class):
    print("- Scale local results by national poll totals ...")
    tmp = df_ridings.drop([target_class], axis=1)
    tmp = scale_df(tmp, df_polls, ScalingOp.mul)
    tmp = fix_other(tmp)

    print("- Merge election results with census data ...")
    return pd.merge(tmp, df_census, on=merge_class)


def project_ridings(df, scaling_factors):
    print("- Re-project dataset from 2013 -> 2023 riding boundaries ...")
    new_rows = []
    for id_new, v in scaling_factors.items():
        for id_old, factor in v.items():
            tmp = df[df["id"] == str(id_old)]
            tmp = scale_df(tmp, factor, ScalingOp.mul)
            tmp["id"] = str(id_new)
            new_rows.append(tmp)
    new_df = pd.concat(new_rows)
    new_df = new_df.groupby("id", as_index=False).sum()
    return new_df


def remove_empty(str_list):
    return [item for item in str_list if len(item) > 0]


def average_polls(df):
    df = df[list(wiki_table_cols)]
    df = df.rename(columns=wiki_table_cols)
    df = df.astype(np.float16)
    df = fix_other(df)
    return df.mean()


def latest_polls(url, n_recent=5):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", attrs=wiki_table_attrs)

    header = [" ".join(item.stripped_strings) for item in table.tbody.tr.children]
    header = remove_empty(header)

    data = []
    for tr in table.tbody.find_all("tr"):
        row = [" ".join(td.stripped_strings) for td in tr.find_all("td")]
        row = remove_empty(row)
        if len(row) == len(header):  # Ignore rows without data
            data.append(row)

    df = pd.DataFrame(data=data[:n_recent], columns=header)
    df = average_polls(df)
    return df.to_dict()
