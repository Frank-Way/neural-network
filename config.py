"""
Модуль с описанием класса, хранящего настройки для обучения
"""
import json
from typing import Tuple
from os.path import exists

import numpy as np

from data_loaders import MNISTDataLoader, ApproximationDataLoader, DataLoader
from exceptions import OptimizerNotFoundException, \
    DataLoaderNotFoundException, ConfigNotFoundException, \
    TooManyInputsException, NotImplementedDataLoaderGraphException
from layers import Dense
from losses import MeanSquaredError, SoftmaxCrossEntropy
from networks import NeuralNetwork
from operations import Sigmoid, Linear, LeakyReLU, ReLU, Tanh
from optimizers import SGD, SGDMomentum, Optimizer
from trainers import Trainer
from utils import calc_accuracy_model, show_results, show_results3d


# noinspection PyTypeChecker,PyArgumentList
class Configuration(object):
    """
    Класс, представляющий обёртку вокруг словаря, хранящем
    конфигурации сети, обучения и входных данных
    """
    decoder = {"dense": Dense, "sigmoid": Sigmoid, "linear": Linear,
               "leakyrelu": LeakyReLU, "relu": ReLU, "tanh": Tanh,
               "mse": MeanSquaredError, "sce": SoftmaxCrossEntropy,
               "sgd": SGD, "sgdm": SGDMomentum, "t": Trainer,
               "mnist": MNISTDataLoader, "ap": ApproximationDataLoader}
    layers: dict
    inputs: dict
    train: dict
    values: dict
    model: NeuralNetwork
    trainer: Trainer
    optimizer: Optimizer
    dl: DataLoader

    def __init__(self, path: str):
        """
        Конструктор
        Parameters
        ----------
        path: Путь к файлу с настройками
        """
        self.path = path

    def load(self, path: str = None) -> None:
        """
        Загрузка настроек из файла
        """
        if path is not None:
            self.path = path
        if exists(self.path):
            with open(self.path, 'r') as file:
                values = json.load(file)
                self.values = values
                self.__dict__.update(values)
        else:
            raise ConfigNotFoundException(self.path)

    def save(self, path: str = None) -> None:
        """
        Созранение настроек в файл
        Parameters
        ----------
        path: Путь к файлу с настройками
        """
        if path is not None:
            self.path = path
        with open(self.path, 'w') as file:
            json.dump(self.values, file, indent=4)

    def update(self, values: dict) -> None:
        self.values = values
        self.__dict__.update(values)

    def _get_net(self) -> NeuralNetwork:
        """
        Генерация модели по заданным настройкам
        Returns
        -------
        NeuralNetwork: Модель
        """
        nn = NeuralNetwork(
            layers=[self.decoder[self.layers["config"][ll]["class"]](
                neurons=self.layers["config"][ll]["neurons"],
                activation=self.decoder[self.layers["config"][ll]["activation"]](),
                dropout=self.layers["config"][ll]["dropout"],
                weight_init=self.layers["config"][ll]["weight_init"]
            )
                for ll in range(self.layers["count"])],
            loss=self.decoder[self.train["loss"]]()
        )
        return nn

    def _get_optimizer(self) -> Optimizer:
        """
        Генерация оптимизатора по заданным настройкам
        Returns
        -------
        Optimizer: Оптимизатор
        """
        if self.train["optimizer"] == "sgd":
            optimizer = self.decoder["sgd"](
                lr=self.train["lr"],
                final_lr=self.train["final_lr"],
                decay_type=self.train["decay_type"]
            )
        elif self.train["optimizer"] == "sgdm":
            optimizer = self.decoder["sgdm"](
                lr=self.train["lr"],
                final_lr=self.train["final_lr"],
                decay_type=self.train["decay_type"],
                momentum=self.train["momentum"]
            )
        else:
            raise OptimizerNotFoundException(self.train["optimizer"])
        return optimizer

    def get_trainer(self) -> Trainer:
        """
        Генерация тренера по заданным настройкам
        Returns
        -------
        Trainer: Тренер
        """
        self.model = self._get_net()
        self.optimizer = self._get_optimizer()
        self.trainer = self.decoder[self.train["trainer"]](self.model,
                                                           self.optimizer)
        return self.trainer

    def get_data(self) -> dict:
        """
        Генерация входных и выходных данных для обучения и тестов в
        соответствии с заданными настройками
        Returns
        -------
        dict
            "x_train": Массив входных значений для обучения
            "x_test":  Массив входных значений для тестов
            "y_train": Массив выходных значений для обучения
            "y_test":  Массив выходных значений для тестов
        """
        if self.inputs["data_loader"] == "ap":
            self.dl = self.decoder["ap"](
                inputs=self.inputs["count"],
                fun_name=self.inputs["function"],
                size=self.inputs["size"],
                scale_inputs=self.inputs["scale_inputs"],
                limits=self.inputs["limits"],
                p_to_test=self.inputs["p_to_test"],
                p_to_extend=self.inputs["p_to_extend"]
            )
        elif self.inputs["data_loader"] == "mnist":
            self.dl = self.decoder["mnist"](self.inputs["path"])
        else:
            raise DataLoaderNotFoundException(self.inputs["data_loader"])
        x_train, x_test, y_train, y_test = self.dl.load()

        return {"x_train": x_train, "x_test": x_test,
                "y_train": y_train, "y_test": y_test}

    def get_fit_params(self, data: dict) -> dict:
        """
        Генерация набора параметров для вызова функции обучения нейросети
        Parameters
        ----------
        data: Данные для обучения и тестов
        Returns
        -------
        dict: Параметры
        """
        fit_params = {
            "x_train": data["x_train"], "x_test": data["x_test"],
            "y_train": data["y_train"], "y_test": data["y_test"],
            "epochs": self.train["epochs"],
            "query_every": self.train["epochs"] // self.train["query_times"],
            "batch_size": self.train["batch_size"],
            "early_stopping": self.train["early_stopping"],
            "seed": self.train["seed"] if self.train["use_seed"] else None,
            "print_results": self.train["print_results"]
        }
        return fit_params

    def get_str_results(self, results: Tuple) -> str:
        """
        Формирование строки с результатами обучения
        Parameters
        ----------
        results: Потеря, макс. ошибка, потери, тестовые входы, выходы и
                 результаты
        Returns
        -------
        str: Результат обучения
        """
        loss, delta, losses, x_test, y_test, test_preds = results
        layers = [self.layers["config"][ll]["neurons"]
                  for ll in range(self.layers["count"])]
        if self.inputs["data_loader"] == "ap":
            function = str(self.dl.simplified_expression)
            size = self.inputs['size']
            rel_delta = delta / np.abs(np.max(y_test) - np.min(y_test)) * 100
            avg_delta = np.average(np.abs(y_test - test_preds))
            result = f"слои: {layers[:-1]};\tфункция: {function};" \
                     f"\tразмер выборки: {size}"
            result += f"\nмакс. абс. ошибка: {delta:e};" \
                      f"\tмакс. относ. ошибка: {rel_delta:e} %;" \
                      f"\tсред. абс. ошибка: {avg_delta:e}"
        elif self.inputs["data_loader"] == "mnist":
            result = f"слои: {layers[:-1]}\tобучение распознаванию " \
                     f"рукописных цифр MNIST"
            result += calc_accuracy_model(self.model, x_test, y_test)
        else:
            raise DataLoaderNotFoundException
        return result

    def get_graph_results(self, results: Tuple) -> Tuple:
        """
        Подготовка функции отрисовки графиков и параметров её вызова
        Parameters
        ----------
        results: Потеря, макс. ошибка, потери, тестовые входы, выходы и
                 результаты
        Returns
        -------
        Tuple: Функция и параметры для её вызова
        """
        if self.inputs["data_loader"] == "ap":
            if self.train["show_plots"]:
                loss, delta, losses, x_test, y_test, test_preds = results
                layers = [self.layers["config"][ll]["neurons"]
                          for ll in range(self.layers["count"])]
                params = {"losses": losses, "x_test": x_test,
                          "pred_test": test_preds, "y_test": y_test,
                          "function_name": self.inputs["function"],
                          "neurons": layers}
                if self.inputs["count"] == 1:
                    function = show_results
                elif self.inputs["count"] == 2:
                    function = show_results3d
                else:
                    raise TooManyInputsException
                return function, params
            else:
                return
        else:
            raise NotImplementedDataLoaderGraphException(self.inputs["data_loader"])
