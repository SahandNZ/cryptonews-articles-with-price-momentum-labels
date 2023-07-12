import os
from abc import abstractmethod
from datetime import datetime
from typing import Tuple, List, Callable, Dict

import torch
from torch import nn
from tqdm.auto import tqdm

from definitions import DEEP_LEARNING_MODELS_DIR
from source.data_loader.data_loader import DataLoader
from source.dataset.dataset import Dataset
from source.performace_measure.deep_learning import DeepLearningPerformanceMeasure
from source.utils.directory import create_directory_recursively
from source.utils.function_call import call_from_dict


class Model(nn.Module):
    def __init__(self, activation_fn: nn.Module, prediction_fn: Callable, show_tqdm: bool):
        super().__init__()
        self.activation_fn: nn.Module = activation_fn
        self.prediction_fn: Callable = prediction_fn
        self.show_tqdm: bool = show_tqdm

        self.train_properties: Dict = None
        self.epochs: int = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.validation_fn: Callable = None
        self.gradiant_clipping_threshold: float = None

    def set_train_properties(self, properties_dict: Dict):
        self.train_properties: Dict = properties_dict
        self.epochs: int = properties_dict['epochs']
        self.loss_fn = call_from_dict(properties_dict['loss_fn'])
        self.validation_fn = properties_dict['validation_fn']
        self.gradiant_clipping_threshold: float = properties_dict['gradiant_clipping_threshold']

    @abstractmethod
    def forward(self, **args):
        raise NotImplemented()

    def train_dataset(self, dataset: Dataset, data_loader: DataLoader):
        data_loader.set_dataset(dataset)

        bar = range(self.epochs)
        if self.show_tqdm:
            bar = tqdm(bar)
            bar.set_description("Training")

        train_pm_dict = {}
        for epoch in bar:
            epoch_pm = DeepLearningPerformanceMeasure(name='Train', epoch=epoch)
            for X, y in data_loader:
                X, y = data_loader.load(X, y)
                output = self.process_dataset(X=X, y=y, optimizer=self.optimizer)
                epoch_pm.append_output(*output)
                data_loader.unload(X, y)

            if self.scheduler is not None:
                self.scheduler.step()

            model_id = self.save()
            train_pm_dict[model_id] = epoch_pm

            if self.show_tqdm:
                learning_rate = self.optimizer.param_groups[0]['lr']
                bar.set_postfix_str("{} lr: {:.4f}".format(epoch_pm, learning_rate))

        return train_pm_dict

    def validate_dataset(self, dataset: Dataset, data_loader: DataLoader, models_id: List[int]):
        bar = models_id
        if self.show_tqdm:
            bar = tqdm(bar)
            bar.set_description("Validating")

        X, y = data_loader.load(dataset.X, dataset.y)
        dev_pm_dict = {}
        for model_id in bar:
            self.load(model_id)
            output = self.process_dataset(X=X, y=y, optimizer=None)
            dev_set_pm = DeepLearningPerformanceMeasure(name='Validation')
            dev_set_pm.append_output(*output)
            dev_pm_dict[model_id] = dev_set_pm

        data_loader.unload(X, y)

        sorted_items = sorted(dev_pm_dict.items(), key=lambda i: self.validation_fn(i[1]))
        dev_pm_dict = {k: v for k, v in sorted_items}

        return dev_pm_dict

    def test_dataset(self, dataset: Dataset, data_loader: DataLoader, model_id: int):
        X, y = data_loader.load(dataset.X, dataset.y)

        self.load(model_id)
        output = self.process_dataset(X=X, y=y, optimizer=None)
        test_pm = DeepLearningPerformanceMeasure(name='Test')
        test_pm.append_output(*output)

        data_loader.unload(X, y)

        return test_pm

    def process_dataset(self, X: torch.Tensor, y: torch.Tensor, optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
        # specify a mode
        if optimizer:
            self.train()
        else:
            self.eval()

        # forward prop
        if optimizer is not None:
            optimizer.zero_grad()

        if optimizer is not None:
            y_hat = self.forward(x=X)
        else:
            with torch.no_grad():
                y_hat = self.forward(x=X)

        loss = self.loss_fn(y_hat, y)

        # backward prop
        if optimizer is not None:
            loss.backward()
            if self.gradiant_clipping_threshold:
                torch.nn.utils.clip_grad_norm(self.parameters(), self.gradiant_clipping_threshold)
            optimizer.step()

        if self.prediction_fn is None:
            actual = None
            prediction = None
        else:
            if 1 == y.size(-1):
                actual = y
            else:
                actual = torch.argmax(y, dim=1)
                prediction = self.prediction_fn(y_hat)

        return y, y_hat, loss, actual, prediction

    def reset(self, xavier: bool = True):
        for layer in self.children():
            if xavier and (type(layer) == nn.Linear or type(layer) == nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        optimizer_dict = self.train_properties['optimizer']
        optimizer_dict['params'] = self.parameters()
        self.optimizer = call_from_dict(optimizer_dict)

        if self.train_properties['scheduler'] is not None:
            scheduler_dict = self.train_properties['scheduler']
            scheduler_dict['optimizer'] = self.optimizer
            self.scheduler = call_from_dict(scheduler_dict)

    def save(self) -> int:
        model_id = int(datetime.now().timestamp() * 1_000_00)
        create_directory_recursively(DEEP_LEARNING_MODELS_DIR)
        path = os.path.join(DEEP_LEARNING_MODELS_DIR, str(model_id))
        torch.save(self.state_dict(), path)
        return model_id

    def load(self, model_id: int, device: str = None):
        path = os.path.join(DEEP_LEARNING_MODELS_DIR, str(model_id))
        if device is not None:
            if 'cuda' == device and torch.cuda.is_available():
                self.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
            else:
                self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(path))
