
from prophet import Prophet
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from pyspark import SparkConf, SparkContext
from databricks.connect import DatabricksSession
from typing import Tuple

from boxiang_pylib.jc.ts.prophet import predict_metric

def test_jc_ts_prophet() -> Tuple[Prophet, pd.DataFrame, Figure]:
    spark = DatabricksSession.builder.getOrCreate()
    spark.sparkContext = None
    game = 'Emoji Blitz'
    market = 'GO'
    user_source_type_cd = 'US'
    target_var = 'INSTALL'
    by = 'MONTH'
    start_dt = '2021-01-01'
    pred_length = 12
    ext_vars = ['FEATURING']
    future_ext_values = np.array([[0]] * 12)
    cap_ratio = 2
    floor_ratio = 0.8
    growth = ['linear', 'logistic', 'flat']
    seasonality_mode = ['additive', 'multiplicative']
    m, pred_df, fig = predict_metric(
        spark=spark,
        game=game,
        market=market,
        user_source_type_cd=user_source_type_cd,
        target_var=target_var,
        by=by,
        start_dt=start_dt,
        pred_length=pred_length,
        ext_vars=ext_vars,
        future_ext_values=future_ext_values,
        cap_ratio=cap_ratio,
        floor_ratio=floor_ratio,
        growth=growth,
        seasonality_mode=seasonality_mode
    )

    return m, pred_df, fig


if __name__ == "__main__":
    model, pred_df, fig = test_jc_ts_prophet()
    fig.show()