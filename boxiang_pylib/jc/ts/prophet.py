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
    """
        Predict future metrics by using Prophet and Hyperopt.
        Please note that the spark_session cannot be a remote session because it does not have sparkContext that is needed by SparkTrial in Hyperopt.

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
            future_ext_values (Optional): a 2D array. Can provide either pred_length or future_pred_ext_values. In case both are provided, pred_length must be the same as future_pred_ext_values.shape[0]
            growth (Optional): linear, logistic, or flat or a list of them. Default is linear. If a list, then it will be included in hyper-parameter search. If growth is logistic, must provide cap or cap_ratio.
            seasonality_mode (Optional): additive or multiplicative or a list of them. Default is additive. If a list, then it will be included in hyper-parameter search.
            interval_width (Optional): a value between 0 and 1. Default is 0.8. 0.95 for 95% confidence interval.
            cap (Optional): the max value to be allowed in prediction. By default of Prophet, cap and floor are only effective for logistic growth. But this function will enforce them regardless of growth type.
            cap_ratio (Optional): set cap = cap_ratio * history_max. Among cap and cap_ratio, cap has higher priority.
            floor (Optional): the min value to be allowed in prediction.
            floor_ratio (Optional): set floor = floor_ratio * history_min. Among floor and floor_ratio, floor has higher priority.
            custom_holidays (Optional): either None or a pandas df with columns: HOLIDAY, DS, LOWER_WINDOW and UPPER_WINDOW. Refer to https://facebook.github.io/prophet/docs/handling_shocks.html#treating-covid-19-lockdowns-as-a-one-off-holidays for a comprehensive example.
            mcmc_samples (Optional): default to 0. Prophet is a daily model. If the training data is at a coarse granularity (i.e., weekly or monthly), we can use MCMC to sample data and provide daily input.
            parallelism (Optional): This is the parallelism for Hyperopt SparkTrial, not for Prophet fit. 1/2 worker number ≤ parellelism ≤ worker number. If None, use Spark worker number. parallelism should be <= 0.1 * max_evals.
            max_evals (Optional): maximum number of trials for Hyperopt hyperparameter search. Should be 10 to 20 times of parameter number. Suggest 60 to 150.
            timeout (Optional): max seconds for Hyperopt search. When the timeout is hit, runs are terminated early and the current results will be returned.

        Returns:
            model: pm.arima.ARIMA
            pred_df: the prediction frame that is indexed by date/time and includes target_var_hat, target_var_lower, target_var_upper and ext_vars,
            Figure: this figure includes 2 plots. The left plot is the validation of the model (80% and 20% of df), and the right plot is the prediction result, which includes the confidence interval
        """

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