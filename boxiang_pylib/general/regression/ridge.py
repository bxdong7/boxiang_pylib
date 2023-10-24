import numpy as np
from pyspark.sql.session import SparkSession
from typing import Union, Tuple, List, Optional, Dict
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK

def train_ridge_model(
        X: np.array,
        y: np.array,
        alpha: Optional[float] = 1.0,
        fit_intercept: Optional[bool] = True,
        positive: Optional[bool] = False,
        max_weight: Optional[float] = 1.0
) -> Ridge:
    """
    Train a ridge regression model based on X and y.

    Args:
        X: 2D array. Both X and y must be sorted based on temporal info from earlist to latest if max_weight != 1.
        y: 1D array
        alpha (default 1.0): the constant that multiplies the L2 term.
        fit_intercept (default True): Whether to fit the intercept for this model. If set to false, no intercept will be used in calculations (i.e. X and y are expected to be centered).
        positive (default False): When set to True, forces the coefficients to be positive. Only ‘lbfgs’ solver is supported in this case.
        max_weight (default 1.0): the maximum weight for all training samples. If different than 1.0, a recency model is applied (higher weights are assigned to more recent training samples).

    Returns:
        model
    """
    # get weights
    if max_weight == 1.0:
        sample_weight = None
    else:
        n = X.shape[0]
        sample_weight_base = max_weight ** (1 / (n - 1))
        sample_weight = np.emath.power(np.array([sample_weight_base]), np.arange(n))

    # fit model
    ridge = Ridge(alpha=alpha, positive=positive, fit_intercept=fit_intercept).fit(X, y, sample_weight=sample_weight)

    return ridge


def train_evaluate_ridge_model(
        X_train: np.array,
        y_train: np.array,
        X_test: np.array,
        y_test: np.array,
        alpha: Optional[float] = 1.0,
        fit_intercept: Optional[bool] = True,
        positive: Optional[bool] = False,
        max_weight: Optional[float] = 1.0
) -> Tuple[Ridge, float]:
    """
    Train a ridge regression model based on X_train and y_train, and evaluate the model's accuracy (R2) on the test set.

    Args:
        X_train: 2D array. Both X and y must be sorted based on temporal info from earlist to latest if max_weight != 1.
        y_train: 1D array.
        X_test: 2D array.
        y_test: 1D array.
        alpha (default 1.0): the constant that multiplies the L2 term.
        fit_intercept (default True): Whether to fit the intercept for this model. If set to false, no intercept will be used in calculations (i.e. X and y are expected to be centered).
        positive (default False): When set to True, forces the coefficients to be positive. Only ‘lbfgs’ solver is supported in this case.
        max_weight (default 1.0): the maximum weight for all training samples. If different than 1.0, a recency model is applied (higher weights are assigned to more recent training samples). In the evaluation, all test samples are equally weighted.

    Returns:
        model:
        r2
    """
    # train model
    ridge = train_ridge_model(X_train, y_train, alpha, fit_intercept, positive, max_weight)

    # calculate loss on test set
    r2 = ridge.score(X_test, y_test)
    return ridge, r2


def visualize_ridge_prediction(
        ridge: Ridge,
        X: np.array,
        y: np.array,
        max_weight: Optional[float] = 1.0
) -> Figure:
    """
    Scatter plot y_pred v.s. y_test. The point size is determined by max_weight.

    Args:
        ridge: a trained ridge model/
        X: 2D array. Both X and y must be sorted based on temporal info from earlist to latest if max_weight != 1.
        y: 1D array.
        max_weight (default 1.0): the maximum weight for all training samples. If different than 1.0, a recency model is applied (higher weights are assigned to more recent training samples).

    Returns:
        The plotted figure.
    """
    # get weights
    n = X.shape[0]
    if max_weight == 1.0:
        sample_weighs = np.array([1.0] * n)
    else:
        sample_weight_base = max_weight ** (1 / (n - 1))
        sample_weighs = np.emath.power(np.array([sample_weight_base]), np.arange(n))
    sample_weighs = sample_weighs * 30

    # get pred
    y_test = y
    y_pred = ridge.predict(X)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(y_test, y_pred, s=sample_weighs, alpha=0.5, c='b')
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Ground Truth", fontsize=20)
    ax.set_ylabel("Prediction", fontsize=20)
    ax.plot([min_val, max_val], [min_val, max_val], c='r', lw=3)
    return fig


def build_ridge_regression_model(
        spark: SparkSession,
        X: np.array,
        y: np.array,
        alpha: Optional[float] = 1.0,
        fit_intercept: Optional[bool] = True,
        positive: Optional[bool] = False,
        max_weight: Optional[float] = 1.0,
        tuning_params: Optional[List[str]] = None,
        parallelism: Optional[int] = None,
        max_evals: Optional[int] = None,
        timeout: Optional[int] = None
) -> Tuple[Ridge, Figure]:
    """
    Train a ridge regression model based on X and y. If tuning_params is not empty, invoke hyperopt for tuning.

    Args:
        spark: SparkSession
        X: 2D array. Both X and y must be sorted based on temporal info from earlist to latest if max_weight != 1.
        y: 1D array.
        alpha (default 1.0): the constant that multiplies the L2 term.
        fit_intercept (default True): Whether to fit the intercept for this model. If set to false, no intercept will be used in calculations (i.e. X and y are expected to be centered).
        positive (default False): When set to True, forces the coefficients to be positive. Only ‘lbfgs’ solver is supported in this case.
        max_weight (default 1.0): the maximum weight for all training samples. If different than 1.0, a recency model is applied (higher weights are assigned to more recent training samples).
        tuning_params: the list of params to tune by using hyperopt. Can be a subset of ['alpha', 'fit_intercept', 'positive', 'max_weight']. If empty or None, it will use the provided parameters to train a ridge model. Otherwise, it will invoke hyperopt to do hyper-param tuning.
        parallelism: This is the parallelism for Hyperopt SparkTrial, not for Prophet fit. 1/2 worker number ≤ parellelism ≤ worker number. If None, use Spark worker number. parallelism should be <= 0.1 * max_evals.
        max_evals: maximum number of trials for Hyperopt hyperparameter search. Should be 10 to 20 times of parameter number. Suggest 60 to 150.
        timeout: max seconds for Hyperopt search. When the timeout is hit, runs are terminated early and the current results will be returned.

    Returns:
        ridge: the trained model
        fig: the plotted prediction figure that shows the ground truth and predictions
    """
    # if no tuning, then simply train
    if tuning_params is None or len(tuning_params) == 0:
        ridge = train_ridge_model(X, y, alpha, fit_intercept, positive, max_weight)
    # else:
    else:
        # organize params
        param_list = ['alpha', 'fit_intercept', 'positive', 'max_weight']
        default_search_space = {
            'alpha': hp.loguniform('alpha', np.log(1e-3), np.log(1e3)),
            'fit_intercept': hp.choice('fit_intercept', [True, False]),
            'positive': hp.choice('positive', [True, False]),
            'max_weight': hp.uniform('max_weight', 1, 30)
        }
        tune_space = {}
        for param in tuning_params:
            tune_space[param] = default_search_space[param]
        pass_params_d = {
            'alpha': alpha,
            'fit_intercept': fit_intercept,
            'positive': positive,
            'max_weight': max_weight
        }
        fix_params = set(param_list).difference(set(tuning_params))
        fix_params_d = {}
        for param in fix_params:
            fix_params_d[param] = pass_params_d[param]

        # split into train and test
        n = X.shape[0]
        train_n = int(n * 0.8)
        X_train, y_train = X[:train_n], y[:train_n]
        X_test, y_test = X[train_n:], y[train_n:]

        # determine max_eval
        if max_evals is None:
            max_evals = 0
            for param in ['alpha', 'max_weight']:
                if param in tune_space:
                    max_evals += 15
            for param in ['fit_intercept', 'positive']:
                if param in tune_space:
                    max_evals += 30

        # determine parallelism
        if parallelism is None:
            max_workers = int(spark.sparkContext.getConf().get('spark.databricks.clusterUsageTags.clusterMaxWorkers'))
            if max_workers <= 0.1 * max_evals:
                parallelism = max_workers
            else:
                parallelism = int(0.1 * max_evals)

        spark_trials = SparkTrials(parallelism, timeout=timeout)

        # perform tpe search
        algo = tpe.suggest

        def train_evaluate_ridge_model_for_hyperopt(
                X_train: np.array,
                y_train: np.array,
                X_test: np.array,
                y_test: np.array,
                fix_params_d: Dict
        ):
            def hyperopt_optimize(tune_params: Dict) -> Dict:
                ridge, r2 = train_evaluate_ridge_model(X_train, y_train, X_test, y_test, **fix_params_d, **tune_params)
                return {'loss': 1 - r2, 'status': STATUS_OK}

            return hyperopt_optimize

        optimize_func = train_evaluate_ridge_model_for_hyperopt(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fix_params_d=fix_params_d
        )
        best_params = fmin(
            fn=optimize_func,
            space=tune_space,
            algo=algo,
            max_evals=max_evals,
            trials=spark_trials
        )
        # retrain with best params
        for param in ['fit_intercept', 'positive']:
            if param in best_params:
                values = [True, False]
                best_params[param] = values[best_params[param]]
        ridge = train_ridge_model(
            X=X,
            y=y,
            **fix_params_d,
            **best_params
        )
        if 'max_weight' in fix_params_d:
            max_weight = fix_params_d['max_weight']
        else:
            max_weight = best_params['max_weight']

    # plot the prediction

    fig = visualize_ridge_prediction(
        ridge=ridge,
        X=X,
        y=y,
        max_weight=max_weight
    )
    return ridge, fig