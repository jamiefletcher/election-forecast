import pandas as pd
from utils import ScalingOp, scale_df, fix_other

import warnings

# Disable annoying warning -- see: https://stackoverflow.com/a/76306267
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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