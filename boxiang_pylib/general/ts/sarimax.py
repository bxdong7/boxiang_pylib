import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import pmdarima.arima as pm
from pmdarima.utils import tsdisplay, plot_pacf, decomposed_plot
from pmdarima.metrics import smape
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple, Union, List, Optional

def visualize_ts(s: pd.Series, m: int) -> None:
    """
    Given a time series indexed by time/date, show three plots
        decomposed plot, which includes etl, trend, season and random
        tsdisplay: which includes time series, ACF plot and histgram
        PACF plot

    Args:
        s: a time series indexed on time/date
        m: the season length. If no seasonality is considered, set to s length
    """
    figsize = (12, 12)
    lag_max = min(m, int(0.5 * s.shape[0]))

    decomposed = pm.decompose(s.values, type_='additive', m=m)
    figure_kwargs = {'figsize': figsize}
    decomposed_plot(decomposed, figure_kwargs=figure_kwargs, show=True)

    tsdisplay(s, lag_max=lag_max, figsize=figsize, title='TSDisplay')

    fig, ax = plt.subplots(figsize=figsize)
    plot_pacf(s, lags=lag_max, ax=ax, title='PACF Plot')
    fig.show()


def train_ts_model(train_df: pd.DataFrame, pred_df: pd.DataFrame, target_var: str,
                   ext_vars: Union[None, List[str]] = None, m: int = 1, tuning_duration: int = 600) -> Tuple[
    pm.arima.ARIMA, pd.DataFrame]:
    """
    Train a SARIMAX model and use it to predict target_var, lower and upper

    Args:
        train_df: the etl that is indexed by time/date and includes target_var and ext_vars
        target_var: the variable name to be modeled
        ext_vars: external signals used to predict target_var, can be None
        m: the seasona length. In case of no season, set m = 1
        tuning_duration: the maximum seconds allocated for tuning

    Returns:
        model: pm.arima.ARIMA
        pred_df: the prediction frame that is indexed by date/time and includes target_var_hat, target_var_lower, target_var_upper and ext_vars,
    """
    # calculate d and D
    y = train_df[target_var].values
    kpss_diffs = pm.ndiffs(y, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = pm.ndiffs(y, alpha=0.05, test='adf', max_d=6)
    d = max(adf_diffs, kpss_diffs)

    D_ch = pm.nsdiffs(y, m=m, max_D=12, test='ch')
    D_oscb = pm.nsdiffs(y, m=m, max_D=12, test='ocsb')
    D = max(D_ch, D_oscb)

    # train
    if ext_vars is None:
        X = None
    else:
        X = train_df[ext_vars].values
    with pm.StepwiseContext(max_dur=tuning_duration):
        # if normal distribution, then directly auto_arima
        _, pvalue = normaltest(train_df[target_var].values)
        if pvalue > 0.05:
            model = pm.auto_arima(train_df[target_var], X=X, d=d, D=D, stepwise=True, m=m,
                                  suppress_warnings=True, error_action="ignore",
                                  seasonal=True, max_p=6, max_order=None, trace=False)
        # else build a pipeline
        else:
            model = Pipeline([
                ("boxcox", BoxCoxEndogTransformer(neg_action='ignore')),
                ('arima', pm.AutoARIMA(d=d, D=D, stepwise=True, m=m,
                                       suppress_warnings=True, error_action="ignore",
                                       seasonal=True, max_p=6, max_order=None, trace=False)
                 )
            ])
            model.fit(train_df[target_var].values, X=X)

    # predict
    pred_length = pred_df.shape[0]
    if ext_vars is None:
        X = None
    else:
        X = pred_df[ext_vars].values
    y_preds, conf_int = model.predict(n_periods=pred_length, X=X, return_conf_int=True)

    lower = conf_int[:, 0]
    upper = conf_int[:, 1]
    pred_df[f"{target_var}_hat"] = y_preds
    pred_df[f"{target_var}_lower"] = lower
    pred_df[f"{target_var}_upper"] = upper
    return model, pred_df


def visualize_ts_prediction(train_df: pd.DataFrame, pred_df: pd.DataFrame, target_var: str, ax: Axes) -> Axes:
    """
    Draw a time series plot that shows the training etl and prediction result. If groud truth is available, also show it in scatters.

    Args:
        train_df: a dataframe indexed by date/time and includes target_var
        pred_df: a dataframe indexed by date/time and includes target_var_hat, target_var_lower, target_var_upper. It may also include target_var if the test ground truth is known
        ax: the ax to plot

    Returns:
        ax: the ax
    """
    ax.plot(train_df.index, train_df[target_var], alpha=0.75)
    ax.plot(pred_df.index, pred_df[f'{target_var}_hat'], alpha=0.75)
    ax.fill_between(pred_df.index, pred_df[f"{target_var}_lower"], pred_df[f"{target_var}_upper"], alpha=0.1, color='b')
    if target_var in pred_df.columns:
        ax.scatter(pred_df.index, pred_df[target_var], alpha=0.4, marker='x')
    return ax


def build_ts_model(df: pd.DataFrame, target_var: str, ext_vars: Optional[Union[None, List[str]]] = None,
                   m: Optional[int] = 1, by: str = 'day', pred_length: Optional[int] = None,
                   future_pred_ext_values: Optional[np.ndarray] = None, tuning_duration: Optional[int] = 600) -> Tuple[
    pm.arima.ARIMA, pd.DataFrame, Figure]:
    """
    Build a SARIMAX model to predict target_var,

    Args:
        df: the etl that is indexed by time/date
        target_var: the variable name to be modeled
        ext_vars: external signals used to predict target_var, can be None
        m: the seasona length. In case of no season, set m = 1
        by: the time series index unit, can be day/week/month
        pred_length (Optional): how long to predict. In case of None, it will be 20% of df length
        future_pred_ext_values (Optional): a 2D array, shape[0] must be the same as pred_length
        tuning_duration: the maximum seconds allocated for tuning

    Returns:
        model: pm.arima.ARIMA
        pred_df: the prediction frame that is indexed by date/time and includes target_var_hat, target_var_lower, target_var_upper and ext_vars,
        Figure: this figure includes 2 plots. The left plot is the validation of the model (80% and 20% of df), and the right plot is the prediction result, which includes the confidence interval
    """
    # split df into train and valid
    n = df.shape[0]
    train_length = int(n * 0.8)
    train_df = df.iloc[:train_length]
    valid_df = df.iloc[train_length:]

    # train, predict and visualize
    fig, axs = plt.subplots(1, 2, figsize=(40, 8))
    _, valid_df = train_ts_model(train_df, valid_df, target_var, ext_vars, m, tuning_duration)
    visualize_ts_prediction(train_df, valid_df, target_var, axs[0])

    # calculate SMAPE and visualize on the plot
    smape_error = smape(valid_df[target_var].values, valid_df[f"{target_var}_hat"].values)
    axs[0].set_title(f"Validation SMAPE = {smape_error:.4f}", fontsize=18)

    if pred_length is None:
        pred_length = future_pred_ext_values.shape[0]

    # prepare predict_df
    if by == 'DAY':
        pred_index = [df.index[-1] + timedelta(days=i) for i in range(1, pred_length + 1)]
    elif by == 'WEEK':
        pred_index = [df.index[-1] + timedelta(weeks=i) for i in range(1, pred_length + 1)]
    else:
        pred_index = [df.index[-1] + timedelta(days=i * 30) for i in range(1, pred_length + 1)]

    if ext_vars is None:
        pred_df = pd.DataFrame(index=pred_index)
    else:
        assert pred_length == future_pred_ext_values.shape[0]
        pred_df = pd.DataFrame(
            future_pred_ext_values,
            columns=ext_vars,
            index=pred_index
        )

    # use all df to train, predict and visualize
    model, pred_df = train_ts_model(df, pred_df, target_var, ext_vars, m)
    pred_df = pred_df[[f"{target_var}_hat", f"{target_var}_lower", f"{target_var}_upper"]]
    visualize_ts_prediction(df, pred_df, target_var, axs[1])

    return model, pred_df, fig