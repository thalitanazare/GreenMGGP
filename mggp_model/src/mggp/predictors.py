
from __future__ import annotations
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mggp.base import Individual


def miso_FIR_INSTANT(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    FIR instantâneo (mesmo instante):
      y_pred[k] alinhado com y_true[k], iniciando em k = lagMax
    """
    regressors = ind.makeRegressors(y_true, u)
    return np.dot(regressors, ind.theta), y_true[ind.lagMax:]


def mimo_FIR_INSTANT(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    FIR instantâneo (MIMO):
      y_pred[k,:] alinhado com y_true[k,:], iniciando em k = lagMax
    """
    regressors = ind.makeRegressors(y_true, u)
    y_pred = [np.dot(regressor, theta) for regressor, theta in zip(regressors, np.array(ind.theta))]
    return np.array(y_pred).T, y_true[ind.lagMax:]


def mimo_INSTANT(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    regressors = ind.makeRegressors(y_true, u)
    yp = [np.dot(regressor, theta) for regressor, theta in zip(regressors, np.array(ind.theta))]
    y_pred = np.array(yp).T
    y_true = y_true[ind.lagMax:]          # mesmo instante
    return y_pred, y_true


def mimo_CLASSIFY(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    Preditor "mesmo instante" para classificação (MIMO).
    Alinha y_pred[k] com y_true[k], iniciando em k = lagMax.
    """
    regressors = ind.makeRegressors(y_true, u)
    yp = [np.dot(regressor, theta) for regressor, theta in zip(regressors, np.array(ind.theta))]
    return np.array(yp).T, y_true[ind.lagMax:]


def miso_CLASSIFY(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    Preditor "mesmo instante" para classificação (MIMO).
    Alinha y_pred[k] com y_true[k], iniciando em k = lagMax.
    """
    regressors = ind.makeRegressors(y_true, u)
    
    y_pred = np.dot(regressors, ind.theta)
    return np.array(y_pred).T, y_true[ind.lagMax:]


def miso_OSA(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    Implements the One-Step_Ahead predictor for MISO models
    Arguments:
        ind = C_Individual object
        y   = 1-dimensional array with output data
        u   = n-dimensional array with input data
    """
    regressors = ind.makeRegressors(y_true, u)
    y_pred = np.dot(regressors, ind.theta)
    return y_pred, y_true[ind.lagMax + 1:]


def mimo_OSA(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    Implements the One-Step_Ahead predictor for MIMO models
    Arguments:
        ind = C_Individual object
        y   = n-dimensional array with output data
        u   = m-dimensional array with input data
    """
    regressors = ind.makeRegressors(y_true, u)
    y_pred = [np.dot(regressor, theta) for regressor, theta in zip(regressors, np.array(ind.theta))]

    return np.array(y_pred).T, y_true[ind.lagMax + 1:]


def miso_FreeRun(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:

    y_true = y_true.reshape(-1, 1)

    if u.ndim == 1:
        u = u.reshape(-1, 1)

    lag = ind.lagMax
    n_samples = u.shape[0]

    y = y_true[:lag].copy()

    y_pred = []

    for k in range(lag, n_samples):

        y_window = y[k - lag:k].reshape(-1, 1)

        listV = [y_window]

        for v in u.T:
            u_window = v[k - lag:k + 1].reshape(-1, 1)
            listV.append(u_window)

        regressors = [1.0]
        # regressors = []

        for j in range(len(ind)):
            func = ind.funcs[j]
            out = func(*listV)
            # regressors.append(float(out[-1]))
            try:
                val = float(out[-1][0])

            except (IndexError, ValueError):
                val = 0.0

            regressors.append(val)

        yk = np.dot(regressors, ind.theta)
        y = np.vstack((y, [yk]))
        y_pred.append(yk)

    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_trim = y_true[lag:]

    return np.nan_to_num(y_pred, nan=0), np.nan_to_num(y_true_trim, nan=0)


def mimo_FreeRun(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    Free-Run predictor for MIMO models (consistente com u[k] e y[k-1])
    """

    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    lag = ind.lagMax
    n_samples = u.shape[0]
    n_outputs = y_true.shape[1]
    n_inputs = u.shape[1]
    y = y_true[:lag, :].copy()

    y_pred = []

    for k in range(lag, n_samples):

        y_windows = [y[k - lag:k, i:i+1] for i in range(n_outputs)]
        # u_windows = [u[k - lag:k + 1, j:j+1] for j in range(n_inputs)]
        u_windows = [u[k - lag+1:k+1, i:i+1] for i in range(n_inputs)]

        listV = y_windows + u_windows

        yk_all_outputs = []

        for idx_output in range(len(ind)):

            regressors = [1.0]  # bias

            for j in range(len(ind[idx_output])):

                func = ind.funcs[idx_output][j]
                try:
                    out = func(*listV)
                except AttributeError:
                    out = np.asarray([0])
                    print("allert!")
                out2 = out.reshape(-1)
                regressors.append(out2[-1])

            theta_k = ind.theta[idx_output]
            yk = np.dot(regressors, theta_k)

            yk_all_outputs.append(yk)

        yk_all_outputs = np.array(yk_all_outputs).reshape(1, -1)

        y = np.vstack((y, yk_all_outputs))
        y_pred.append(yk_all_outputs)

    y_pred = np.vstack(y_pred)  # (n_samples-lag, n_outputs)
    y_true_trim = y_true[lag:, :]

    return np.nan_to_num(y_pred, nan=0), np.nan_to_num(y_true_trim, nan=0)
    

def mimo_FIR_FreeRun(ind: Individual, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    Implements the Free-Run predictor for MIMO models.
    Args:
        ind: C_Individual object (MIMO).
        y0: Initial conditions (n_outputs x history).
        u: Input data (n_samples x n_inputs).
    Returns:
        y_pred: Predicted outputs (n_samples x n_outputs).
        y_true: Ground truth (trimmed to match y_pred).
    """
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)

    n_samples = u.shape[0]
    n_outputs = y_true.shape[1]
    initial_conditions_size = ind.lagMax
    y_pred = np.zeros((n_samples - initial_conditions_size, n_outputs))
    
    y_history = y_true[:initial_conditions_size + 1 , :].copy()

    for step in tqdm(range(n_samples - initial_conditions_size), desc="Processing iterations in FreeRun"):

        listV = []
        for v in u.T:
            listV.append(v[step:step + initial_conditions_size + 1].reshape(-1, 1))

        for idx_output in range(n_outputs):
            regressors = [1.0]  # bias

            for idx_equation_tree in range(len(ind[idx_output])):

                genetic_programming_term = ind.funcs[idx_output][idx_equation_tree]
                out = genetic_programming_term(*listV)

                out = out.reshape(-1)
                regressors.append(out[-1])
            
            y_pred[step, idx_output] = np.dot(regressors, ind.theta[idx_output])

        y_history = np.column_stack([y_history.T, y_pred[step, :]]).T

    y_true = y_true[initial_conditions_size:, :]
    return y_pred, y_true


def miso_MShooting(ind: Individual, k: int, y: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    Implements the Multiple-Shooting predictor for MISO models
    Arguments:
        ind = C_Individual object
        k   = steps ahead prediction for each 'shooting'
        y   = 1-dimensional array with output data
        u   = n-dimensional array with input data
    """
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)

    initial_conditions_size = ind.lagMax 
    batch_size = initial_conditions_size + k
    n_batchs = int(np.floor(u.shape[0] / batch_size))
    newshape = (n_batchs, batch_size, 1)

    listU = [np.resize(v, newshape) for v in u.T]
    y_true = np.resize(y, newshape)
    y_pred = y_true[:, :initial_conditions_size , :]

    for shooting in range(k):

        regressors = []
        out = np.ones((n_batchs, 1, 1))
        regressors.append(out)

        for idx_output_equation in range(len(ind)):

            genetic_programming_term = ind.funcs[idx_output_equation]

            y_window = [y_pred[:, shooting:shooting + initial_conditions_size, :]]
            u_window = [v[:, shooting:shooting + initial_conditions_size + 1, :] for v in listU]

            listV = y_window + u_window

            out = genetic_programming_term(*listV)
            out = out[:, -1:, :]
            regressors.append(out)

        regressors = np.concatenate(regressors, axis=2)
        y_pred = np.concatenate((y_pred, np.dot(regressors, ind.theta)), axis=1)
    
    return np.nan_to_num(y_pred.reshape(-1, 1), nan=0), np.nan_to_num(y_true.reshape(-1, 1), nan=0)


def mimo_MShooting(ind: Individual, k: int, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """
    Multiple-Shooting predictor for MIMO models
    """

    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)

    initial_conditions_size = ind.lagMax
    batch_size = initial_conditions_size + k
    n_batchs = int(np.floor(u.shape[0] / batch_size))

    newshape_u = (n_batchs, batch_size, 1)
    listU = [np.resize(v, newshape_u) for v in u.T]

    y_true = np.resize(y_true, (n_batchs, batch_size, y_true.shape[1]))
    y_pred = y_true[:, :initial_conditions_size, :]

    for shooting in range(k):

        predictions_for_output_equation = []

        for idx_output_equation in range(len(ind)):

            regressors = []

            # bias
            out = np.ones((n_batchs, 1, 1))
            regressors.append(out)

            y_window = [y_pred[:, shooting:shooting + initial_conditions_size, i:i+1] for i in range(y_pred.shape[2])]
            u_window = [v[:, shooting:shooting + initial_conditions_size + 1, :] for v in listU]

            listV = y_window + u_window

            for idx_tree in range(len(ind[idx_output_equation])):

                genetic_programming_term = ind.funcs[idx_output_equation][idx_tree]
                out = genetic_programming_term(*listV)
                out = out[:, -1:, :]

                regressors.append(out)

            regressors = np.concatenate(regressors, axis=2)

            theta_k = ind.theta[idx_output_equation] 
            output_pred = np.dot(regressors, theta_k.T).reshape(n_batchs, 1, 1)

            predictions_for_output_equation.append(output_pred)

        predictions_for_output_equation = np.concatenate(predictions_for_output_equation, axis=2)
        y_pred = np.concatenate((y_pred, predictions_for_output_equation), axis=1)

    return np.nan_to_num(y_pred.reshape(-1, y_pred.shape[1]), nan=0), np.nan_to_num(y_true.reshape(-1, y_true.shape[1]), nan=0)


def mimo_FIR_MShooting(ind: Individual, k: int, y_true: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:

    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)
    
    initial_conditions_size = ind.lagMax 
    batch_size = initial_conditions_size + 1 + k
    n_batchs = int(np.floor(u.shape[0] / batch_size))
    newshape = (n_batchs, batch_size, 1)

    listU = []
    for v in u.T:
        listU.append(np.resize(v, newshape))

    y_true = np.resize(y_true, (n_batchs, batch_size, y_true.shape[1]))
    y_pred = y_true[:, :initial_conditions_size + 1, :]

    for shooting in range(k):

        listV = []
        for v in listU:
            listV.append(v[:, shooting:shooting + initial_conditions_size + 1, 0].reshape(n_batchs, -1, 1))

        predictions_for_output_equation = []
        for idx_output_equation in range(len(ind)):

            regressors = []
            out = np.ones((n_batchs, 1, 1))
            regressors.append(out)

            for idx_equation_tree in range(len(ind[idx_output_equation])):

                func = ind.funcs[idx_output_equation][idx_equation_tree]
                out = func(*listV)
                out = out[:, initial_conditions_size:, :]
                regressors.append(out)

            regressors = np.concatenate(regressors, axis=2)
            output_pred = np.dot(regressors, ind.theta[idx_output_equation].T).reshape(-1, 1, 1)
            predictions_for_output_equation.append(output_pred)
        
        predictions_for_output_equation = np.concatenate(predictions_for_output_equation, axis=2)
        y_pred = np.concatenate((y_pred, predictions_for_output_equation), axis=1)

    return np.nan_to_num(y_pred.reshape(-1, y_true.shape[1]), nan=0), np.nan_to_num(y_true.reshape(-1, y_true.shape[1]), nan=0)
