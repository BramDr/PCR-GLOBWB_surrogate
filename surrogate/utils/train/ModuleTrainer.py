from typing import Sequence, Optional, Union

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils.clip_grad as clip

from surrogate.nn import SurrogateModel
from surrogate.nn.functional import Metric
from surrogate.utils.data import SequenceDataLoader
from surrogate.utils.data import SequencePreFetcher

from .Callback import Callback

class ModuleTrainer():
    def __init__(self,
                 train_loader: Union[SequenceDataLoader, SequencePreFetcher],
                 model: SurrogateModel,
                 optimizer: Optional[optim.Optimizer] = None,
                 test_loaders: Optional[Sequence[Union[SequenceDataLoader, SequencePreFetcher]]] = None,
                 callbacks: Optional[list[Callback]] = None) -> None:
        
        if test_loaders is None:
            test_loaders = []
        if callbacks is None:
            callbacks = []
        
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.test_loaders = test_loaders
        self.callbacks = callbacks

        param = next(iter(model.parameters()))
        x, _ = next(iter(train_loader))

        self.cuda = False
        if param.device.type == "cuda" and not x.device.type == "cuda":
            self.cuda = True

    def __str__(self):
        callback_strings = ["{}".format(type(callback).__name__) for callback in self.callbacks]
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
                              metric: Metric) -> torch.Tensor:
        if self.cuda:
            y_true = y_true.cuda()
            
        metric_value = metric.calculate_reduce(pred=y_pred,
                                               true=y_true)
        return metric_value

    def _optimize_model_batch(self,
                              loss_value: torch.Tensor,
                              optimizer: optim.Optimizer) -> None:

        optimizer.zero_grad()
        loss_value.sum().backward()
        optimizer.step()
        clip.clip_grad_norm_(parameters=self.model.parameters(), max_norm=2.0)

    def _run_training_batch(self,
                            x: torch.Tensor,
                            y_true: torch.Tensor,
                            loss: Optional[Metric] = None) -> torch.Tensor:
        
        #forward_start = time.perf_counter()
        y_pred = self._model_forward(x = x)
        #print(x[0, 0, :])
        #print(y_pred[0, 0, :])
        #print(y_true[0, 0, :])
        #forward_time = time.perf_counter() - forward_start
        #print("Forward time {: .2e}".format(forward_time))
        
        loss_value = torch.Tensor([0])
        if loss is not None:
            #loss_start = time.perf_counter()
            loss_value = self._evaluate_model_batch(y_pred=y_pred,
                                                    y_true=y_true,
                                                    metric=loss)
            #loss_time = time.perf_counter() - loss_start
            #print("Loss time {: .2e}".format(loss_time))
            
            if self.optimizer is not None:
                #optimize_start = time.perf_counter()
                self._optimize_model_batch(loss_value=loss_value,
                                            optimizer=self.optimizer)
                #optimize_time = time.perf_counter() - optimize_start
                #print("Optimize time {: .2e}".format(optimize_time))
                
        return loss_value

    def _run_testing_batch(self,
                           x: torch.Tensor,
                           y_true: torch.Tensor,
                           loss: Optional[Metric] = None,
                           metrics: Optional[Sequence[Metric]] = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        
        if metrics is None:
            metrics = []
        
        with torch.inference_mode():
            y_pred = self._model_forward(x = x)
            
            loss_value = torch.Tensor([0])
            if loss is not None:
                loss_value = self._evaluate_model_batch(y_pred=y_pred,
                                                        y_true=y_true,
                                                        metric=loss)
            
            metric_values = []
            for metric in metrics:
                metric_value = self._evaluate_model_batch(y_pred=y_pred,
                                                          y_true=y_true,
                                                          metric=metric)
                metric_values.append(metric_value)
                
        return loss_value, metric_values
    
    def run(self,
            epochs: int,
            loss: Optional[Metric] = None,
            metrics: Optional[Sequence[Metric]] = None,
            verbose: int = 1,
            seed: int = 19920223):
        
        random.seed(seed)
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)
            
        if metrics is None:
            metrics = []
            
        best_loss = float("inf")
        
        for callback in self.callbacks:
            callback.start_callback(trainer = self)

        for epoch in range(epochs):
            statistics_dict = {}
                        
            # Training
            train_loss_value = 0
            train_loss_number = 0
            
            self.model.train(True)
            for x, y_true in self.train_loader:
                    loss_value = self._run_training_batch(x=x,
                                                          y_true=y_true,
                                                          loss=loss)
                    train_loss_number += len(loss_value)
                    train_loss_value += loss_value.sum().item()
            self.model.train(False)
            
            if train_loss_number > 0:
                train_loss_value /= train_loss_number
            else:
                train_loss_value = float("-inf")
            statistics_dict["train_loss"] = train_loss_value
            
            # Testing
            if len(self.test_loaders) > 0:
                test_loss_value = 0
                test_loss_number = 0
                test_metric_values = [0] * len(metrics)
                test_metric_numbers = [0] * len(metrics)
                
                for test_loader in self.test_loaders:
                    for x, y_true in test_loader:
                            loss_value, metric_values = self._run_testing_batch(x=x,
                                                                                y_true=y_true,
                                                                                loss=loss,
                                                                                metrics=metrics)
                            test_loss_number += len(loss_value)
                            test_loss_value += loss_value.sum().item()
                            
                            for index, metric_value in enumerate(metric_values):
                                test_metric_numbers[index] += len(metric_value)
                                test_metric_values[index] += metric_value.sum().item()
                
                if test_loss_number > 0:
                    test_loss_value /= test_loss_number
                else:
                    test_loss_value = float("-inf")
                statistics_dict["test_loss"] = test_loss_value
                
                for metric, test_metric_value, test_metric_number in zip(metrics, test_metric_values, test_metric_numbers):
                    if test_metric_number > 0:
                        test_metric_value /= test_metric_number
                    else:
                        test_metric_value = float("-inf")                    
                    statistics_dict["test_{}".format(type(metric).__name__)] = test_metric_value
            
            try:
                loss_value = statistics_dict["test_loss"]
            except KeyError:
                loss_value = statistics_dict["train_loss"]
            if loss_value < best_loss:
                best_loss = loss_value
            statistics_dict["best_loss"] = best_loss
            
            if verbose > 0:
                statistics_str = ""
                statistics_str += "epoch {:04d}".format(epoch)
                statistics_str += "".join([" - {} {: .2e}".format(k, v) for k, v in statistics_dict.items()])
                print(statistics_str, flush=True)
                
            for callback in self.callbacks:
                callback.epoch_end_callback(trainer=self,
                                            epoch=epoch, 
                                            statistics=statistics_dict)
                
        for callback in self.callbacks:
            callback.end_callback(trainer = self)

        return best_loss
