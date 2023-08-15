import numpy as np
import pandas as pd
import pmdarima.arima as pm
from matplotlib.figure import Figure
from typing import Tuple
import sys
from library.setting import PACKAGE_PATH
if PACKAGE_PATH not in sys.path:
    sys.path.append(PACKAGE_PATH)
from library.jc.ts.sarimax import predict_metric
from databricks.connect import DatabricksSession


def test_jc_ts_sarimax() -> Tuple[pm.arima.ARIMA, pd.DataFrame, Figure]:
    spark = DatabricksSession.builder.getOrCreate()
    game = 'Emoji Blitz'
    market = ['GO']
    user_source_type_cd = ['MK']
    channel = 'ADWORDS'
    country = ['UNITED STATES', 'CANADA']
    target_var = 'LTV'
    ext_vars = ['SPEND', 'FEATURING']
    m = 52
    by = 'WEEK'
    start_dt = '2021-01-01'
    end_dt = None
    pred_length = 4
    future_pred_ext_values = np.array([[100000, 0]] * 4)
    tuning_duration = 30

    model, pred_df, fig = predict_metric(
        spark=spark,
        game=game,
        market=market,
        user_source_type_cd=user_source_type_cd,
        channel=channel,
        country=country,
        target_var=target_var,
        ext_vars=ext_vars,
        m=m,
        by=by,
        start_dt=start_dt,
        end_dt=end_dt,
        pred_length=pred_length,
        future_pred_ext_values=future_pred_ext_values,
        tuning_duration=tuning_duration
    )
    return model, pred_df, fig


if __name__ == "__main__":
    model, pred_df, fig = test_jc_ts_sarimax()
    fig.show()