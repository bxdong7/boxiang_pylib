import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
from datetime import datetime
from prophet import Prophet
from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK
from pyspark.sql import SparkSession
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from boxiang_pylib.general.utils.loss import logcosh_loss
from boxiang_pylib.setting import DATE_FORMAT

def predict_with_prophet_model(
        train_df: pd.DataFrame,
        target_var: str,
        by: str,
        pred_length: int,
        params: Dict,
        ext_vars: Union[None, List[str]] = None,
        future_ext_values: Union[None, np.ndarray] = None
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Fit a prophet model.

    Args:
        train_df: a dataframe includes by, target_var and ext_vars
        target_var: the var to predict
        by: DAY/WEEK/MONTH.
        pred_length: number of periods to predict.
        params: the params that include 'growth', 'seasonality_mode', 'interval_width', 'holidays', 'mcmc_samples', 'changepoint_prior_scale', 'seasonality_prior_scale', 'holidays_prior_scale', 'cap' and 'floor'.
        ext_vars: dependent variables.
        future_ext_values: the future values of ext_vars. shape[0] must be the same as pred_length.

    Returns:
        m: a Prophet model.
        pred_df: a prediction data frame that includes by, f"{target_var}", f"{target_var}_lower", f"{target_var}_upper".
    """
    # split params into data_params and model_params
    model_param_list = ['growth', 'seasonality_mode', 'interval_width', 'holidays', 'mcmc_samples',
                        'changepoint_prior_scale', 'seasonality_prior_scale', 'holidays_prior_scale']
    model_params = {}
    for param in model_param_list:
        if param in params:
            model_params[param] = params[param]
    if 'cap' in params:
        cap = params['cap']
    else:
        cap = None
    if 'floor' in params:
        floor = params['floor']
    else:
        floor = None

    # prepare train data
    train_df = train_df.rename(
        columns={
            by: 'ds',
            target_var: 'y'
        }
    )

    # build model
    m = Prophet(**model_params)
    if ext_vars is not None:
        for ext_var in ext_vars:
            m.add_regressor(ext_var)
    if cap is not None:
        train_df['cap'] = cap
    if floor is not None:
        train_df['floor'] = floor
    m.fit(train_df)

    # predict
    if by == 'DAY':
        freq = 'D'
    elif by == 'WEEK':
        freq = 'W-MON'
    else:
        freq = 'MS'
    pred_df = m.make_future_dataframe(periods=pred_length, freq=freq, include_history=False)
    if ext_vars is not None:
        pred_df[ext_vars] = future_ext_values
    if cap is not None:
        pred_df['cap'] = cap
    if floor is not None:
        pred_df['floor'] = floor

    pred_df = m.predict(pred_df)

    # organize return result
    pred_df = pred_df.rename(
        columns={
            "yhat": f"{target_var}",
            "yhat_lower": f"{target_var}_lower",
            "yhat_upper": f"{target_var}_upper",
            "ds": by
        }
    )
    pred_df = pred_df[[by, f"{target_var}", f"{target_var}_lower", f"{target_var}_upper"]]

    # enforce cap and floor
    if cap is not None:
        pred_df.loc[pred_df[target_var] > cap, target_var] = cap
    if floor is not None:
        pred_df.loc[pred_df[target_var] < floor, target_var] = floor

    # return
    return m, pred_df


def predict_with_prophet_model_for_hyperopt(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_var: str,
        by: str,
        non_tune_params: Dict,
        ext_vars: Union[None, List[str]] = None
):
    def objective(tune_params: Dict) -> Dict:
        # combine params
        params = {**non_tune_params, **tune_params}

        # organize future_ext_values
        pred_length = valid_df.shape[0]
        if ext_vars is not None:
            future_ext_values = valid_df[ext_vars].values
        else:
            future_ext_values = None

        # train and predict
        m, pred_df = predict_with_prophet_model(
            train_df=train_df,
            target_var=target_var,
            by=by,
            pred_length=pred_length,
            params=params,
            ext_vars=ext_vars,
            future_ext_values=future_ext_values
        )

        # calculate loss
        y_test = valid_df[target_var].values
        y_pred = pred_df[target_var].values
        loss = logcosh_loss(y_pred, y_test)

        # organize return result
        return {'loss': loss, 'status': STATUS_OK}

    return objective

def visualize_prophet_prediction(
    m: Prophet,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    by: str,
    target_var: str
) -> Figure:
    # copy data
    train_df = train_df.copy(deep=True)
    pred_df = pred_df.copy(deep=True)
    pred_df[by] = pred_df[by].apply(lambda x: x.date())

    # rename columns
    train_df = train_df.rename(
        columns={
            by: 'ds',
            target_var: 'y'
        }
    )
    pred_df = pred_df.rename(
        columns={
            by: 'ds',
            f"{target_var}": 'yhat',
            f"{target_var}_lower": 'yhat_lower',
            f"{target_var}_upper": 'yhat_upper',
        }
    )
    df = pd.concat([train_df, pred_df], ignore_index=True)
    df['ds'] = pd.to_datetime(df['ds'])

    # plot predictions
    fig, ax = plt.subplots(figsize=(20, 8))
    m.plot(df, ax=ax)
    ax.set_title("Predictions", fontsize=20)
    fig.suptitle(target_var, fontsize=25)
    return fig

def train_prophet_model(
        spark: SparkSession,
        df: pd.DataFrame,
        target_var: str,
        by: str,
        pred_length: int,
        ext_vars: Union[None, List[str]] = None,
        future_ext_values: Union[None, np.ndarray] = None,
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
    Use prophet to predict future values based on time series. Hyperopt is used to finetune the model.
    Use logcosh loss as the metric to find best hyperparameters.

    Args:
        spark: a SparkSession that is used to create SparkTrials for parallel hyperopt search.
        df: training dataset that includes by, target_var and ext_vars if any
        target_var: the variable to predict
        by: DAY/WEEK/MONTH. If not daily, need to provide mcmc_samples like 100, 300.
        target_var: the var to be predicted
        ext_vars: a list of additional features to consider
        pred_length: number of periods to predict
        future_ext_values: either None or a 2D array of future values. Its shape[0] must match pred_length.
        growth: linear, logistic, or flat or a list of them. If a list, then it will be included in hyper-parameter search. If growth is logistic, must provide cap or cap_ratio.
        seasonality_mode: additive or multiplicative or a list of them. If a list, then it will be included in hyper-parameter search.
        interval_width: a value between 0 and 1. Default is 0.8. 0.95 for 95% confidence interval.
        cap: the max value to be allowed in prediction. By default of Prophet, cap and floor are only effective for logistic growth. But this function will enforce them regardless of growth type.
        cap_ratio: set cap = cap_ratio * history_max. Among cap and cap_ratio, cap has higher priority.
        floor: the min value to be allowed in prediction.
        floor_ratio: set floor = floor_ratio * history_min. Among floor and floor_ratio, floor has higher priority.
        custom_holidays: either None or a pandas df with columns: HOLIDAY, DS, LOWER_WINDOW and UPPER_WINDOW. Refer to https://facebook.github.io/prophet/docs/handling_shocks.html#treating-covid-19-lockdowns-as-a-one-off-holidays for a comprehensive example.
        mcmc_samples: default to 0. Prophet is a daily model. If the training data is at a coarse granularity (i.e., weekly or monthly), we can use MCMC to sample data and provide daily input.
        parallelism: This is the parallelism for Hyperopt SparkTrial, not for Prophet fit. 1/2 worker number ≤ parellelism ≤ worker number. If None, use Spark worker number. parallelism should be <= 0.1 * max_evals.
        max_evals: maximum number of trials for Hyperopt hyperparameter search. Should be 10 to 20 times of parameter number. Suggest 60 to 150.
        timeout: max seconds for Hyperopt search. When the timeout is hit, runs are terminated early and the current results will be returned.

    Returns:
        m: a Prophet model.
        pred_df: a prediction data frame that includes by, f"{target_var}", f"{target_var}_lower", f"{target_var}_upper".
        fig: a figure that shows the predictions.
    """
    # split into train and valid set
    n = df.shape[0]
    train_df = df.iloc[:int(n * 0.8)].copy()
    valid_df = df.iloc[int(n * 0.8):].copy()

    # organize params
    params = {}
    if type(growth) == str:
        params['growth'] = growth
    if type(seasonality_mode) == str:
        params['seasonality_mode'] = seasonality_mode
    params['interval_width'] = interval_width
    if custom_holidays is not None:
        params['holidays'] = custom_holidays
    if by != 'DAY':
        if mcmc_samples == 0:
            params['mcmc_samples'] = 100
        else:
            params['mcmc_samples'] = mcmc_samples
    if cap is not None:
        params['cap'] = cap
    elif cap_ratio is not None:
        params['cap'] = train_df[target_var].max() * cap_ratio
    if floor is not None:
        params['floor'] = floor
    elif floor_ratio is not None:
        params['floor'] = train_df[target_var].min() * floor_ratio

    # define search space
    param_search_space = {
        'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', np.log(0.001), np.log(0.5)),
        'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', np.log(0.01), np.log(10)),
        'holidays_prior_scale': hp.loguniform('holidays_prior_scale', np.log(0.01), np.log(10))
    }
    if type(growth) == list:
        param_search_space['growth'] = hp.choice('growth', growth)
    if type(seasonality_mode) == list:
        param_search_space['seasonality_mode'] = hp.choice('seasonality_mode', seasonality_mode)

    # perform tpe search
    algo = tpe.suggest
    optimize_func = predict_with_prophet_model_for_hyperopt(
        train_df=train_df,
        valid_df=valid_df,
        target_var=target_var,
        by=by,
        non_tune_params=params,
        ext_vars=ext_vars
    )
    if max_evals is None:
        max_evals = 60
        if type(growth) == list:
            max_evals += 15 * len(growth)
        if type(seasonality_mode) == list:
            max_evals += 15 * len(seasonality_mode)

    if parallelism is None:
        max_workers = int(spark.conf.get('spark.databricks.clusterUsageTags.clusterMaxWorkers'))
        if max_workers <= 0.1 * max_evals:
            parallelism = max_workers
        else:
            parallelism = int(0.1 * max_evals)

    spark_trials = SparkTrials(parallelism, timeout=timeout, spark_session=spark)

    best_params = fmin(
        fn=optimize_func,
        space=param_search_space,
        algo=algo,
        max_evals=max_evals,
        trials=spark_trials
    )
    if 'growth' in best_params:
        best_params['growth'] = growth[best_params['growth']]
    if 'seasonality_mode' in best_params:
        best_params['seasonality_mode'] = seasonality_mode[best_params['seasonality_mode']]

    # use best params to fit and predict
    m, pred_df = predict_with_prophet_model(
        train_df=df,
        target_var=target_var,
        by=by,
        pred_length=pred_length,
        params={**params, **best_params},
        ext_vars=ext_vars,
        future_ext_values=future_ext_values
    )

    fig = visualize_prophet_prediction(m, train_df, pred_df, by, target_var)

    return m, pred_df, fig