import numpy as np
import pandas as pd

def reshape_load_curve_lit(config, df_original):

    target_shape_data = config['demand_data']['target_shape_own']
    target_shape = np.array(target_shape_data)/1000
    target_shape /= target_shape.sum()

    reshaped_days = []
    for date, group in df_original.groupby(df_original.index.date):
        if len(group) != 24:
            continue
        reshaped_day = pd.DataFrame(index=group.index)
        for col in group.columns:
            daily_total = group[col].sum()
            reshaped_day[col] = daily_total * target_shape
        reshaped_days.append(reshaped_day)

    reshaped_df = pd.concat(reshaped_days)

    return reshaped_df