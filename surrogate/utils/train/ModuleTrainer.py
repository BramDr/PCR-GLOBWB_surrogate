from typing import Sequence, Optional, Union

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from surrogate.nn import SurrogateModel
from surrogate.utils.data import DataLoader
from surrogate.utils.data import PreFetcher

from .Callback import Callback


class ModuleTrainer():
    def __init__(self,
                 train_loader: Union[DataLoader, PreFetcher],
                 model: SurrogateModel,
                 optimizer: optim.Optimizer,
                 validation_loader: Optional[Union[DataLoader, PreFetcher]] = None,
                 test_loader: Optional[Union[DataLoader, PreFetcher]] = None,
                 scheduler: Optional[lr_scheduler._LRScheduler] = None,
                 callbacks: Optional[Sequence[Callback]] = None,
                 transformers: Optional[np.ndarray] = None) -> None:

        if callbacks is None:
            callbacks = []

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.model = model
        self.optimizer = optimizer
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.transformers = transformers

        param = next(iter(model.parameters()))
        
        if train_loader is not None:
            x = train_loader.dataset.x_items[0]
        elif validation_loader is not None:
            x = validation_loader.dataset.x_items[0]
        elif test_loader is not None:
            x = test_loader.dataset.x_items[0]
        else:
            raise ValueError("No dataloader provided")

        self.cuda = False
        if param.device.type == "cuda" and not x.device.type == "cuda":
            self.cuda = True
            
        self.epoch = None
        self.train_loss = None
        self.validation_loss = None
        self.test_losses = None
        self.best_loss = None

    def __str__(self):
        callback_strings = ["{}".format(type(callback).__name__)
                            for callback in self.callbacks]
        callback_string = ", ".join(callback_strings)

        string = "{}:\n".format(type(self).__name__)
        string += "\tOptimizer: {}\n".format(type(self.optimizer).__name__)
        string += "\tCallbacks: {}\n".format(callback_string)
        return string

    def _model_forward(self,
                       x: torch.Tensor) -> torch.Tensor:
        if self.cuda:
            x = x.cuda()
        y_pred, _ = self.model.forward(x)
        return y_pred

    def _evaluate_model_batch(self,
                              y_pred: torch.Tensor,
                              y_true: torch.Tensor,
                              metric: nn.Module) -> torch.Tensor:
        if self.cuda:
            y_true = y_true.cuda()
        
        metric_value = metric.forward(input=y_pred,
                                      target=y_true)
        return metric_value

    def _optimize_model_batch(self,
                              loss_value: torch.Tensor,
                              optimizer: optim.Optimizer) -> None:
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    def _run_training_batch(self,
                            x: torch.Tensor,
                            y_true: torch.Tensor,
                            loss: nn.Module) -> torch.Tensor:
        y_pred = self._model_forward(x=x)
        loss_value = self._evaluate_model_batch(y_pred=y_pred,
                                                y_true=y_true,
                                                metric=loss)
        self._optimize_model_batch(loss_value=loss_value,
                                    optimizer=self.optimizer)
        return loss_value

    def _run_validation_batch(self,
                                x: torch.Tensor,
                                y_true: torch.Tensor,
                                loss: nn.Module) -> torch.Tensor:
        with torch.inference_mode():
            y_pred = self._model_forward(x=x)
            loss_value = self._evaluate_model_batch(y_pred=y_pred,
                                                    y_true=y_true,
                                                    metric=loss)
        return loss_value

    def _run_test_batch(self,
                        x: torch.Tensor,
                        y_true: torch.Tensor,
                        metrics: Sequence[nn.Module]) -> list[torch.Tensor]:
        
        with torch.inference_mode():
            y_pred = self._model_forward(x=x)
            
            if self.transformers is not None:
                y_pred = y_pred.clone()
                y_true = y_true.clone()
                for index, transformer in enumerate(self.transformers):
                    y_pred[..., [index]] = transformer.detransform_normalize(y_pred[..., [index]])
                    y_true[..., [index]] = transformer.detransform_normalize(y_true[..., [index]])
            
            metric_values = []
            for metric in metrics:
                metric_value = self._evaluate_model_batch(y_pred=y_pred,
                                                          y_true=y_true,
                                                          metric=metric)
                metric_values.append(metric_value)
        return metric_values

    def test(self,
             metrics: Sequence[nn.Module],
             verbose: int = 1) -> list[float]:
                
        # Test
        if self.test_loader is None:
            raise ValueError("No test data loader provided")
        
        self.test_losses = {}
        
        test_number = 0
        test_metric_values = [0. for _ in range(len(metrics))]

        self.model.train(False)
        for x, y_true in self.test_loader:
            
            metric_values = self._run_test_batch(x=x,
                                                y_true=y_true,
                                                metrics=metrics)
            
            test_number += 1
            for index, metric_value in enumerate(metric_values):
                test_metric_values[index] += metric_value.item()

        test_adj_metric_values = []
        for test_metric_value in test_metric_values:
            test_metric_value /= test_number
            test_adj_metric_values.append(test_metric_value)
        
        for metric, test_metric_value in zip(metrics, test_adj_metric_values):
            self.test_losses[type(metric).__name__] = test_metric_value
            if np.isnan(test_metric_value) or np.isinf(test_metric_value):
                raise ValueError("Test metric value is {}".format(test_metric_value))
            
        if verbose > 0:
            print(" - ".join(["{} {: .2e} ".format(k, v) for k, v in self.test_losses.items()]))
        
        return list(self.test_losses.values())
    
    def run(self,
            epochs: int,
            loss: nn.Module,
            verbose: int = 1,
            seed: int = 19920223) -> float:

        random.seed(seed)
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

        self.best_loss = float("inf")

        for callback in self.callbacks:
            callback.start_callback(trainer=self)

        for self.epoch in range(epochs):

            # Training
            if self.train_loader is None:
                raise ValueError("No train data loader provided")
            
            train_number = 0
            train_loss_value = 0

            self.model.train(True)
            for x, y_true in self.train_loader:
                loss_value = self._run_training_batch(x=x,
                                                      y_true=y_true,
                                                      loss=loss)
                
                train_number += 1
                train_loss_value += loss_value.item()
            self.model.train(False)

            train_loss_value /= train_number
            self.train_loss = train_loss_value
        
            if np.isnan(train_loss_value) or np.isinf(train_loss_value):
                raise ValueError("Train loss value is {}".format(train_loss_value))

            # validation
            if self.validation_loader is not None:
                
                validation_number = 0
                validation_loss_value = 0
                
                for x, y_true in self.validation_loader:
                    loss_value = self._run_validation_batch(x=x,
                                                            y_true=y_true,
                                                            loss=loss)
                    
                    validation_number += 1
                    validation_loss_value += loss_value.item()

                validation_loss_value /= validation_number
                self.validation_loss = validation_loss_value
        
                if np.isnan(validation_loss_value) or np.isinf(validation_loss_value):
                    raise ValueError("Validation loss value is {}".format(validation_loss_value))

            if self.validation_loss is not None:
                loss_value = self.validation_loss
            else:
                loss_value = self.train_loss
            if loss_value < self.best_loss:
                self.best_loss = loss_value

            if verbose > 0:
                statistics = "epoch {:04d} - train loss {: .2e} - validation loss {: .2e} - best loss {: .2e}".format(self.epoch,
                                                                                                                        self.train_loss,
                                                                                                                        self.validation_loss,
                                                                                                                        self.best_loss)
                if self.scheduler is not None:
                    statistics += " - learning rate {: .2e}".format(self.scheduler.get_last_lr()[0])
                print(statistics)
            
            if self.scheduler is not None:
                self.scheduler.step()

            for callback in self.callbacks:
                callback.epoch_end_callback(trainer=self)

        for callback in self.callbacks:
            callback.end_callback(trainer=self)

        return self.best_loss
