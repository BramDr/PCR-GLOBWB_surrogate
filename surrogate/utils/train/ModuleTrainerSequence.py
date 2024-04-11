from typing import Sequence, Optional, Union

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from surrogate.nn import SurrogateModel
from surrogate.utils.data import DataLoaderSequence
from surrogate.utils.data import PreFetcherSequence

from .ModuleTrainer import ModuleTrainer
from .Callback import Callback


class ModuleTrainerSequence(ModuleTrainer):
    def __init__(self,
                 train_loader: Union[DataLoaderSequence, PreFetcherSequence],
                 model: SurrogateModel,
                 optimizer: optim.Optimizer,
                 validation_loader: Optional[Union[DataLoaderSequence, PreFetcherSequence]] = None,
                 test_loader: Optional[Union[DataLoaderSequence, PreFetcherSequence]] = None,
                 scheduler: Optional[lr_scheduler._LRScheduler] = None,
                 callbacks: Optional[Sequence[Callback]] = None,
                 transformers: Optional[np.ndarray] = None) -> None:

        super(ModuleTrainerSequence, self).__init__(train_loader = train_loader,
                                                    validation_loader=validation_loader,
                                                    model = model,
                                                    optimizer=optimizer,
                                                    test_loader=test_loader,
                                                    scheduler=scheduler,
                                                    callbacks=callbacks,
                                                    transformers=transformers)

    def _model_forward(self,
                       x: torch.Tensor,
                       hidden: Optional[torch.Tensor] = None,
                       cell: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                
        if self.cuda:
            x = x.cuda()
            
        y_pred, (hidden, cell) = self.model.forward(x, hidden, cell)
        return y_pred, (hidden, cell)

    def _run_training_batch(self,
                            x: torch.Tensor,
                            y_true: torch.Tensor,
                            loss: nn.Module,
                            hidden: Optional[torch.Tensor] = None,
                            cell: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        y_pred, (hidden, cell) = self._model_forward(x=x,
                                                     hidden=hidden,
                                                     cell=cell)
                    
        loss_value = self._evaluate_model_batch(y_pred=y_pred,
                                                y_true=y_true,
                                                metric=loss)
        self._optimize_model_batch(loss_value=loss_value,
                                    optimizer=self.optimizer)
        return loss_value, (hidden, cell)
    
    def _run_training_sequence(self,
                               xs: Sequence[torch.Tensor],
                               y_trues: Sequence[torch.Tensor],
                               loss: nn.Module,
                               spinup_sequences: int = 0,
                               hidden: Optional[torch.Tensor] = None,
                               cell: Optional[torch.Tensor] = None) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:

        train_number = 0
        train_loss_value = 0
        for seq_index, (x, y_true) in enumerate(zip(xs, y_trues)):
            
            # Spinup
            if seq_index < spinup_sequences:
                with torch.no_grad():
                    _, (hidden, cell) = self._model_forward(x=x,
                                                            hidden=hidden,
                                                            cell=cell)
                hidden = hidden.detach()
                cell = cell.detach()
                continue
            
            # Training
            loss_value, (hidden, cell) = self._run_training_batch(x=x,
                                                                    y_true=y_true,
                                                                    loss=loss,
                                                                    hidden = hidden,
                                                                    cell = cell)
            hidden = hidden.detach()
            cell = cell.detach()
            
            train_number += 1
            train_loss_value += loss_value.item()
        
        train_loss_value /= train_number
        
        return train_loss_value, (hidden, cell)
            
    def _run_validation_batch(self,
                           x: torch.Tensor,
                           y_true: torch.Tensor,
                           loss: nn.Module,
                           hidden: Optional[torch.Tensor] = None,
                           cell: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        
        with torch.inference_mode():
            y_pred, (hidden, cell) = self._model_forward(x=x,
                                                        hidden=hidden,
                                                        cell=cell)
                    
            loss_value = self._evaluate_model_batch(y_pred=y_pred,
                                                    y_true=y_true,
                                                    metric=loss)
        return loss_value, (hidden, cell)

    def _run_validation_sequence(self,
                                 xs: Sequence[torch.Tensor],
                                 y_trues: Sequence[torch.Tensor],
                                 loss: nn.Module,
                                 spinup_sequences: int = 0,
                                 hidden: Optional[torch.Tensor] = None,
                                 cell: Optional[torch.Tensor] = None) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
        
        validation_number = 0
        validation_loss_value = 0
        for seq_index, (x, y_true) in enumerate(zip(xs, y_trues)):
        
            # Spinup
            if seq_index < spinup_sequences:
                with torch.no_grad():
                    _, (hidden, cell) = self._model_forward(x=x,
                                                            hidden=hidden,
                                                            cell=cell)
                hidden = hidden.detach()
                cell = cell.detach()
                continue
                
            # Validation
            loss_value, (hidden, cell) = self._run_validation_batch(x=x,
                                                                    y_true=y_true,
                                                                    loss=loss,
                                                                    hidden = hidden,
                                                                    cell = cell)
            hidden = hidden.detach()
            cell = cell.detach()
            
            validation_number += 1
            validation_loss_value += loss_value.item()
        
        validation_loss_value /= validation_number
        
        return validation_loss_value, (hidden, cell)
    
    def _run_test_batch(self,
                        x: torch.Tensor,
                        y_true: torch.Tensor,
                        metrics: Sequence[nn.Module],
                        hidden: Optional[torch.Tensor] = None,
                        cell: Optional[torch.Tensor] = None) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        
        with torch.inference_mode():
            y_pred, (hidden, cell) = self._model_forward(x=x,
                                                         hidden=hidden,
                                                         cell=cell)
            
            if self.transformers is not None:
                y_pred = y_pred.clone()
                y_true = y_true.clone()
                for index, transformer in enumerate(self.transformers):
                    y_pred[..., [index]] = transformer.detransform(y_pred[..., [index]])
                    y_true[..., [index]] = transformer.detransform(y_true[..., [index]])
            
            metric_values = []
            for metric in metrics:
                metric_value = self._evaluate_model_batch(y_pred=y_pred,
                                                          y_true=y_true,
                                                          metric=metric)
                metric_values.append(metric_value)
        return metric_values, (hidden, cell)
    
    def _run_test_sequence(self,
                           xs: Sequence[torch.Tensor],
                           y_trues: Sequence[torch.Tensor],
                           metrics: Sequence[nn.Module],
                           spinup_sequences: int = 0,
                           hidden: Optional[torch.Tensor] = None,
                           cell: Optional[torch.Tensor] = None) -> tuple[list[float], tuple[torch.Tensor, torch.Tensor]]:
        
        test_number = 0
        test_metric_values = [0. for _ in range(len(metrics))]
        for seq_index, (x, y_true) in enumerate(zip(xs, y_trues)):
                
            # Spinup
            if seq_index < spinup_sequences:
                with torch.no_grad():
                    _, (hidden, cell) = self._model_forward(x=x,
                                                            hidden=hidden,
                                                            cell=cell)
                hidden = hidden.detach()
                cell = cell.detach()
                continue
            
            # Test
            metric_values, (hidden, cell) = self._run_test_batch(x=x,
                                                                 y_true=y_true,
                                                                 metrics=metrics,
                                                                 hidden = hidden,
                                                                 cell = cell)
            
            hidden = hidden.detach()
            cell = cell.detach()
            
            test_number += 1
            for index, metric_value in enumerate(metric_values):
                test_metric_values[index] += metric_value.item()
                
        test_adj_metric_values = []
        for test_metric_value in test_metric_values:
            test_metric_value /= test_number
            test_adj_metric_values.append(test_metric_value)
                
        return test_adj_metric_values, (hidden, cell)

    def test(self,
             metrics: Sequence[nn.Module],
             spinup_sequences: int = 1,
             verbose: int = 1) -> list[float]:
                
        # Test
        if self.test_loader is None:
            raise ValueError("No test data loader provided")
        
        self.test_losses = {}
        
        test_number = 0
        test_metric_values = [0. for _ in range(len(metrics))]
        
        self.model.train(False)
        for xs, y_trues in self.test_loader:
                    
            metric_values, _ = self._run_test_sequence(xs = xs,
                                                       y_trues=y_trues,
                                                       metrics = metrics,
                                                       spinup_sequences=spinup_sequences)
            
            test_number += 1
            for index, metric_value in enumerate(metric_values):
                test_metric_values[index] += metric_value
        
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
            spinup_sequences: int = 1,
            verbose: int = 1,
            seed: int = 19920223):

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
            for xs, y_trues in self.train_loader:
                
                loss_value, _ = self._run_training_sequence(xs = xs,
                                                            y_trues=y_trues,
                                                            loss = loss,
                                                            spinup_sequences=spinup_sequences)
                    
                train_number += 1
                train_loss_value += loss_value
            
            train_loss_value /= train_number
            self.train_loss = train_loss_value
        
            if np.isnan(train_loss_value) or np.isinf(train_loss_value):
                raise ValueError("Train loss value is {}".format(train_loss_value))

            # validation
            if self.validation_loader is not None:
                
                validation_number = 0
                validation_loss_value = 0
                
                self.model.train(False)
                for xs, y_trues in self.validation_loader:
                            
                    loss_value, _ = self._run_validation_sequence(xs = xs,
                                                                  y_trues=y_trues,
                                                                  loss = loss,
                                                                  spinup_sequences=spinup_sequences)
                        
                    validation_number += 1
                    validation_loss_value += loss_value

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
