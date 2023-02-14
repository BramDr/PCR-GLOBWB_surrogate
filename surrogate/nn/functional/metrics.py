from typing import Union

import numpy as np
import torch


def SE(y_pred: Union[torch.Tensor, np.ndarray],
       y_true: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    err = (y_true - y_pred) ** 2
    return err


def RSE_T(y_pred: torch.Tensor,
          y_true: torch.Tensor) -> torch.Tensor:
    err = SE(y_pred=y_pred, y_true=y_true)
    err = torch.sqrt(err)
    return err


def RSE(y_pred: np.ndarray,
        y_true: np.ndarray) -> np.ndarray:
    err = SE(y_pred=y_pred, y_true=y_true)
    err = np.sqrt(err)
    return err


def NSE_T(y_pred: torch.Tensor,
          y_true: torch.Tensor) -> torch.Tensor:
    y_true_mean = torch.mean(input = y_true, dim = 1, keepdim = True)
    y_true_mean = y_true_mean.expand_as(y_true)

    err_sim = SE(y_pred=y_pred, y_true=y_true)
    err_sim = torch.sum(err_sim, dim=1, keepdim=True)

    err_mean = SE(y_pred=y_true_mean, y_true=y_true)
    err_mean = torch.sum(err_mean, dim=1, keepdim=True)

    err = 1 - err_sim / err_mean
    err = torch.where(err_mean == 0, 0, err)
    return err


def NSE(y_pred: np.ndarray,
        y_true: np.ndarray) -> np.ndarray:
    y_true_mean = np.mean(y_true, axis=1, keepdims=True)
    y_true_mean = np.broadcast_to(y_true_mean, shape=y_true.shape)

    err_sim = SE(y_pred=y_pred, y_true=y_true)
    err_sim = np.sum(err_sim, axis=1, keepdims=True)

    err_mean = SE(y_pred=y_true_mean, y_true=y_true)
    err_sim = np.sum(err_sim, axis=1, keepdims=True)

    err = 1 - err_sim / err_mean
    err = np.where(err_mean == 0, 0, err)
    return err
