from pyspark.sql.session import SparkSession
from typing import Union, Tuple, List, Optional
import pandas as pd
import numpy as np
from prophet import Prophet
from matplotlib.figure import Figure
import sys
from boxiang_pylib.setting import PACKAGE_PATH
if PACKAGE_PATH not in sys.path:
    sys.path.append(PACKAGE_PATH)
from boxiang_pylib.general.ts.prophet import train_prophet_model
from boxiang_pylib.jc.etl.data_loader import get_ts_data

def predict_metric(
    spark: SparkSession,
    game: Union[str, List[str]],
    market: Union[str, List[str]],
    user_source_type_cd: Union[str, List[str]],
    source: Optional[Union[str, List[str]]] = None,
    channel: Optional[Union[str, List[str]]] = None,
    country: Optional[Union[str, List[str]]] = None,
    target_var: str = 'LTV',
    ext_vars: Optional[Union[None, List[str]]] = None,
    by: str = 'DAY',
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
    pred_length: Optional[int] = None,
    future_ext_values: Optional[np.ndarray] = None,
    growth: Union[str, List[str]] = 'linear',
    seasonality_mode: Union[str, List[str]] = 'additive',
    interval_width: float = 0.8,
    cap: Union[None, float] = None,
    cap_ratio: Union[None, float] = None,
    floor: Union[None, float] = None,
    floor_ratio: Union[None, float] = None,
    custom_holidays: Union[None, pd.DataFrame] = None,
    mcmc_samples: int = 0,
    parallelism: Union[None, int] = None,
    max_evals: Union[None, int] = None,
    timeout: Union[None, int] = None
) -> Tuple[Prophet, pd.DataFrame, Figure]:
    # load data
    df = get_ts_data(
        spark=spark,
        game=game,
        market=market,
        user_source_type_cd=user_source_type_cd,
        source=source,
        channel=channel,
        country=country,
        target_var=target_var,
        by=by,
        start_dt=start_dt,
        end_dt=end_dt
    )
    df = df.reset_index(drop=False)

    # build model
    m, pred_df, fig = train_prophet_model(
        spark=spark,
        df=df,
        target_var=target_var,
        by=by,
        pred_length=pred_length,
        ext_vars=ext_vars,
        future_ext_values=future_ext_values,
        growth=growth,
        seasonality_mode=seasonality_mode,
        interval_width=interval_width,
        cap=cap,
        cap_ratio=cap_ratio,
        floor=floor,
        floor_ratio=floor_ratio,
        custom_holidays=custom_holidays,
        mcmc_samples=mcmc_samples,
        parallelism=parallelism,
        max_evals=max_evals,
        timeout=timeout
    )

    # return
    return m, pred_df, fig