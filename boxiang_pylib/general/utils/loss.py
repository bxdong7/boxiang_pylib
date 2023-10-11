from typing import Union
import numpy as np

def logcosh(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)

def logcosh_loss(pred_y: np.ndarray, test_y: np.ndarray) -> float:
    """
    Calcualte the log-cosh loss.

    Args:
        pred_y: a 1d array that includes the predictions
        test_y: a 1d array that includes the ground truths

    Returns:
        the loss
    """
    loss = np.mean(logcosh(pred_y - test_y))
    return loss