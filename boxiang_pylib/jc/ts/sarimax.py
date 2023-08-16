from pyspark.sql.session import SparkSession
from typing import Union, Tuple, List, Optional
import pandas as pd
import numpy as np
import pmdarima.arima as pm
from matplotlib.figure import Figure
import sys
from boxiang_pylib.setting import PACKAGE_PATH
if PACKAGE_PATH not in sys.path:
    sys.path.append(PACKAGE_PATH)
from boxiang_pylib.general.ts.sarimax import build_ts_model
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
    m: Optional[int] = 1,
    by: str = 'DAY',
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None,
    pred_length: Optional[int] = None,
    future_pred_ext_values: Optional[np.ndarray] = None,
    tuning_duration: Optional[int] = 600
) -> Tuple[pm.arima.ARIMA, pd.DataFrame, Figure]:
    """
    Predict future metrics by using SARIMAX

    Args:
        spark: a SparkSession object
        game: can be a single game or a list of games
        market: IT or GO or both
        user_source_type_cd: US, MK or [US, MK]
        source (Optional): SKAN or Non-SKAN or both
        channel (Optional)
        country (Optional): full country name. If market includes IT, do not include country
        target_var: ARPI (ARPI_001, ARPI_003, ARPI_007, ARPI_014, ARPI_030, ARPI_060, ARPI_090, ARPI_180, ARPI_270)
                    LTV
                    CPI
                    INSTALL
                    REVENUE
                    ROAS
        ext_vars: a list of variables, including SPEND, FEATURING
        m (Optional): the seasona length. In case of no season, set m = 1
        by: DAY/WEEK/MONTH
        start_dt (Optional): the start dt of etl loading
        end_dt (Optional): the end dt of etl loading
        pred_length (Optional): number of periods to predict
        future_pred_ext_values (Optional): a 2D array. Can provide either pred_length or future_pred_ext_values. In case both are provided, pred_length must be the same as future_pred_ext_values.shape[0]
        tuning_duration (Optional): the maximum seconds allocated for tuning

    Returns:
        model: pm.arima.ARIMA
        pred_df: the prediction frame that is indexed by date/time and includes target_var_hat, target_var_lower, target_var_upper and ext_vars,
        Figure: this figure includes 2 plots. The left plot is the validation of the model (80% and 20% of df), and the right plot is the prediction result, which includes the confidence interval
    """
    # load etl
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
    if ext_vars is None:
        keep_cols = [target_var]
    else:
        keep_cols = [target_var] + ext_vars
    df = df[keep_cols]

    # build_ts_model
    model, pred_df, fig = build_ts_model(
        df=df,
        target_var=target_var,
        ext_vars=ext_vars,
        m=m,
        by=by,
        pred_length=pred_length,
        future_pred_ext_values=future_pred_ext_values,
        tuning_duration=tuning_duration
    )
    return model, pred_df, fig

# 973-740-1330