import numpy as np
import pandas as pd
from pyspark.sql.session import SparkSession
from typing import Union, Tuple, List, Optional
from matplotlib.figure import Figure
from sklearn.linear_model import Ridge

import sys
from boxiang_pylib.setting import PACKAGE_PATH
if PACKAGE_PATH not in sys.path:
    sys.path.append(PACKAGE_PATH)
from boxiang_pylib.general.regression.ridge import build_ridge_regression_model
from boxiang_pylib.jc.etl.data_loader import get_ts_data


def train_predict(
        spark: SparkSession,
        game: Union[str, List[str]],
        market: Union[str, List[str]],
        user_source_type_cd: Union[str, List[str]],
        source: Optional[Union[str, List[str]]] = None,
        channel: Optional[Union[str, List[str]]] = None,
        country: Optional[Union[str, List[str]]] = None,
        target_var: Optional[str] = 'LTV',
        feature_vars: Optional[List[str]] = ['SPEND', 'FEATURING'],
        by: Optional[str] = 'DAY',
        start_dt: Optional[str] = None,
        end_dt: Optional[str] = None,
        pred_length: Optional[int] = None,
        future_feature_vals: Optional[np.ndarray] = None,
        alpha: Optional[float] = 1.0,
        fit_intercept: Optional[bool] = True,
        positive: Optional[bool] = False,
        max_weight: Optional[float] = 1.0,
        tuning_params: Optional[List[str]] = None,
        parallelism: Optional[int] = None,
        max_evals: Optional[int] = None,
        timeout: Optional[int] = None
) -> Tuple[Ridge, pd.DataFrame, Figure]:
    """
    Train and predict a ridge regression model of a target_var based on feature_vars.
    If pred_length = 0 or future_feature_vars is empty: simply do train and do not predict. In that case, the 2nd return value is None.

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
        feature_vars: a list of variables, including SPEND, FEATURING
        m (Optional): the seasona length. In case of no season, set m = 1
        by: DAY/WEEK/MONTH
        start_dt (Optional): the start dt of etl loading
        end_dt (Optional): the end dt of etl loading
        pred_length (Optional): number of periods to predict
        future_feature_vals (Optional): a 2D array. Can provide either pred_length or future_feature_vals. In case both are provided, pred_length must be the same as future_feature_vals.shape[0]. If future_feature_vals is None, no prediction will be returned.
        alpha (default 1.0): the constant that multiplies the L2 term.
        fit_intercept (default True): Whether to fit the intercept for this model. If set to false, no intercept will be used in calculations (i.e. X and y are expected to be centered).
        positive (default False): When set to True, forces the coefficients to be positive. Only ‘lbfgs’ solver is supported in this case.
        max_weight (default 1.0): the maximum weight for all training samples. If different than 1.0, a recency model is applied (higher weights are assigned to more recent training samples).
        tuning_params: the list of params to tune by using hyperopt. Can be a subset of ['alpha', 'fit_intercept', 'positive', 'max_weight']. If empty or None, it will use the provided parameters to train a ridge model. Otherwise, it will invoke hyperopt to do hyper-param tuning.
        parallelism: This is the parallelism for Hyperopt SparkTrial, not for Prophet fit. 1/2 worker number ≤ parellelism ≤ worker number. If None, use Spark worker number. parallelism should be <= 0.1 * max_evals.
        max_evals: maximum number of trials for Hyperopt hyperparameter search. Should be 10 to 20 times of parameter number. Suggest 60 to 150.
        timeout: max seconds for Hyperopt search. When the timeout is hit, runs are terminated early and the current results will be returned.
    Returns:
        ridge: the ridge model
        pred_df: the prediction frame that is indexed by date/time and includes feature_vars and target_var. If pred_length = 0 or future_feature_vars is empty, pred_df is None.
        Figure: a scatter plot that shows the ground truth and predictions on training data.
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
    last_train_dt = df.index[-1]
    df = df.reset_index(drop=False)

    # train model and generate plot
    X_train = df[feature_vars].values
    y_train = df[target_var].values
    ridge, fig = build_ridge_regression_model(
        spark=spark,
        X=X_train,
        y=y_train,
        alpha=alpha,
        fit_intercept=fit_intercept,
        positive=positive,
        max_weight=max_weight,
        tuning_params=tuning_params,
        parallelism=parallelism,
        max_evals=max_evals,
        timeout=timeout
    )

    # generate predictions if needed
    if future_feature_vals is not None:
        y_pred = ridge.predict(future_feature_vals)
        n = y_pred.shape[0]
        freq_d = {
            'DAY': 'D',
            'WEEK': 'W-MON',
            'MONTH': 'MS'
        }
        freq = freq_d[by]
        dts = pd.date_range(start=last_train_dt, periods=n + 1, freq=freq, inclusive='right').date
        data = np.concatenate([future_feature_vals, np.expand_dims(y_pred, 1)], axis=1)
        pred_df = pd.DataFrame(
            data,
            index=dts,
            columns=feature_vars + [target_var]
        )
    else:
        pred_df = None

    return ridge, pred_df, fig