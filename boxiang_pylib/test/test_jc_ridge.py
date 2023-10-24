import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from databricks.connect import DatabricksSession
from typing import Tuple
from sklearn.linear_model import Ridge

from boxiang_pylib.jc.regression.ridge import train_predict


def test_jc_ridge() -> Tuple[Ridge, pd.DataFrame, Figure]:
    spark = DatabricksSession.builder.getOrCreate()
    game = 'Emoji Blitz'
    market = 'GO'
    user_source_type_cd = 'MK'
    target_var = 'INSTALL'
    feature_vars = ['SPEND', 'FEATURING']
    by = 'MONTH'
    start_dt = '2021-01-01'
    pred_length = 12
    future_feature_vals = np.array([[200000, 0]] * pred_length)
    tuning_params = ['alpha', 'fit_intercept', 'positive', 'max_weight']

    ridge, pred_df, fig = train_predict(
        spark=spark,
        game=game,
        market=market,
        user_source_type_cd=user_source_type_cd,
        target_var=target_var,
        by=by,
        start_dt=start_dt,
        pred_length=pred_length,
        future_feature_vals=future_feature_vals,
        tuning_params=tuning_params
    )

    return ridge, pred_df, fig


if __name__ == "__main__":
    ridge, pred_df, fig = test_jc_ridge()
    fig.show()