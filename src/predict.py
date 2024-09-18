import pandas as pd
from utils import ScalingOp, scale_df, fix_other

def scale_polling(df_census, df_ridings, df_polls, target_class, merge_class):
    # Xy = []

    print("- Scale local results by national poll totals ...")
    # df_target = df_ridings
    # for year_target, df_target in df_ridings.items():
    # select winner as target
    # target = df_target[[merge_class, target_class]]
    
    # df = df_ridings 
    # for df in df_ridings.values():
    # drop existing winner cat
    tmp = df_ridings.drop([target_class], axis=1)
    # scale riding back up by target national result
    tmp = scale_df(tmp, df_polls, ScalingOp.mul)
    # re-compute "OTH" so totals sum to 1.0
    tmp = fix_other(tmp)
    # merge target winner cat
    # Xy.append(pd.merge(tmp, target, on=merge_class))

    # concat all individual Xy into a single df and merge with census
    # Xy = pd.concat(Xy, ignore_index=True)

    print("- Merge election results with census data ...")
    Xy = pd.merge(tmp, df_census, on=merge_class)
    return Xy