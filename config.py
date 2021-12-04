"""
Модуль с описанием класса, хранящего настройки для обучения
"""
import json
import os
from typing import Tuple, Any
from os.path import exists

import numpy as np
from numpy import ndarray

from data_loaders import MNISTDataLoader, ApproximationDataLoader, DataLoader
from exceptions import OptimizerNotFoundException, \
    DataLoaderNotFoundException, ConfigNotFoundException, \
    TooManyInputsException, NotImplementedDataLoaderGraphException, NoPathToConfigSpecifiedException
from layers import Dense
from losses import MeanSquaredError, SoftmaxCrossEntropy
from networks import NeuralNetwork
from operations import Sigmoid, Linear, LeakyReLU, ReLU, Tanh
from optimizers import SGD, SGDMomentum, Optimizer
from trainers import Trainer
from utils import calc_accuracy_model, show_results, show_results3d, show_results_losses


# noinspection PyTypeChecker,PyArgumentList
class Configuration(object):
    """
    Класс, представляющий обёртку вокруг словаря, хранящем
    конфигурации сети, обучения и входных данных
    """
    # декодировщик значений из конфигурации в нужные классы
    decoder = {"dense": Dense, "sigmoid": Sigmoid, "linear": Linear,
               "leakyrelu": LeakyReLU, "relu": ReLU, "tanh": Tanh,
               "mse": MeanSquaredError, "sce": SoftmaxCrossEntropy,
               "sgd": SGD, "sgdm": SGDMomentum, "t": Trainer,
               "mnist": MNISTDataLoader, "ap": ApproximationDataLoader}
    layers: dict  # конфигурация слоёв
    inputs: dict  # конфигурация входных данных
    train: dict   # конфигурация обучения
    values: dict  # объединение конфигураций
    model: NeuralNetwork  # нейросеть
    trainer: Trainer  # тренер
    optimizer: Optimizer  # оптимизатор
    dl: DataLoader  # загрузчик данных
    deltas: Tuple[float, float, float]  # результаты обучения

    def __init__(self, path: str = None):
        """
        Конструктор
        Parameters
        ----------
        path: Путь к файлу с настройками
        """
        if path is not None:
            self.path = path
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load(self, path: str = None) -> None:
        """
        Загрузка настроек из файла. Если путь не был задан ранее, то
        генерируется исключение. Если путь задан, но файл отсутствует, то
        генерируется конфигурация по умолчанию
        Parameters
        ----------
        path: Путь к файлу с настройками
        """
        if path is not None:
            self.path = path
        if self.path is not None:
            if exists(self.path):
                with open(self.path, 'r') as file:
                    values = json.load(file)
                    self.update(values)
            else:
                self.generate_default_config()
        else:
            raise NoPathToConfigSpecifiedException()

    def save(self, path: str = None) -> None:
        """
        Сохранение настроек в файл. Если путь не указан, то генерируется
        соответствующее исключение
        Parameters
        ----------
        path: Путь к файлу с настройками
        """
        if path is not None:
            self.path = path
        if self.path is not None:
            with open(self.path, 'w') as file:
                json.dump(self.values, file, indent=4)
        else:
            raise NoPathToConfigSpecifiedException()

    def generate_default_config(self) -> None:
        """
        Генерация конфигурации по умолчанию и обновление экземпляра в
        соответствии с ней
        """
        values = {"inputs": {"count": 1,
                             "path": "",
                             "data_loader": "ap",
                             "function": "sin(pi*x1)",
                             "limits": [[0.0, 1.0]],
                             "size": 256,
                             "scale_inputs": False,
                             "p_to_test": 0.3,
                             "p_to_extend": 0.1},
                  "train": {"seed": 0,
                            "use_seed": False,
                            "lr": 0.1,
                            "final_lr": 0.001,
                            "decay_type": "lin",
                            "epochs": 5000,
                            "query_times": 10,
                            "batch_size": 64,
                            "early_stopping": True,
                            "print_results": True,
                            "show_plots": False,
                            "momentum": 0.8,
                            "loss": "mse",
                            "optimizer": "sgd",
                            "trainer": "t",
                            "restarts": 1},
                  "layers": {"count": 2,
                             "config":
                                 [
                                     {
                                         "neurons": 8,
                                         "class": "dense",
                                         "activation": "tanh",
                                         "dropout": 1.0,
                                         "weight_init": "glorot"},
                                     {
                                         "neurons": 1,
                                         "class": "dense",
                                         "activation": "linear",
                                         "dropout": 1.0,
                                         "weight_init": "glorot"}
                                 ]
                             }
                  }
        self.update(values)

    def update(self, values: dict) -> None:
        """
        Обновление экземпляра в соответствии с полученной конфигурацией
        Parameters
        ----------
        values: Конфигуарция
        """
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
            self.deltas = (delta, rel_delta, avg_delta)
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
                    function = show_results_losses
                return function, params
            else:
                return
        else:
            raise NotImplementedDataLoaderGraphException(self.inputs["data_loader"])

    def export_model(self, result_type: str = "text") -> Any:
        """
        Экспорт модели в заданный тип
        Parameters
        ----------
        result_type: Тип результата экспорта (text - строка)
        Returns
        -------
        Результат экспорта, тип которого определяется параметром result_type
        """
        if result_type != "text":
            raise NotImplementedError()

        activations_decoder = {"linear": "Линейная", "sigmoid": "Сигмоида",
                               "tanh": "Гиперболичесий тангенс", "relu": "ReLU",
                               "leakyrelu": "Leaky ReLU"}

        msg = ""
        delim = "#" * 80
        msg += delim + "\n"

        msg += "Нейронная сеть, обученная для воспроизведения зависимости\n"
        if self.inputs["count"] == 1:
            msg += "    F(x1)"
        elif self.inputs["count"] == 2:
            msg += "    F(x1, x2)"
        else:
            msg += f"    F(x1, ..., x{self.inputs['count']})"
        msg += f" = {self.dl.str_expression}\n"

        msg += "в пределах изменения входных переменных:\n"
        for ii in range(self.inputs["count"]):
            msg += f"    x{ii + 1}: [{self.inputs['limits'][ii][0]}; " \
                   f"{self.inputs['limits'][ii][1]}]\n"

        if self.deltas is not None:
            msg += "\n".join(("с точностью:",
                              f"    макс. абс. ошибка = {self.deltas[0]:e}",
                              f"    макс. отн. ошибка = {self.deltas[1]:e} %",
                              f"    сред. абс. ошибка = {self.deltas[2]:e}"))
        msg += "\n" + delim + "\n"

        msg += "Конфигурация слоёв\n"
        for ii in range(self.layers["count"]):
            neurons = self.layers['config'][ii]['neurons']
            act = activations_decoder[self.layers['config'][ii]['activation']]
            dropout = self.layers['config'][ii]['dropout']
            layer_cfg = "\n".join((f"{ii + 1})",
                                   f"    нейронов - {neurons}",
                                   f"    активация - {act}",
                                   f"    dropout - {dropout}"))
            msg += layer_cfg + "\n"
        msg += delim + "\n"

        msg += "Параметры модели\n"
        for ii in range(self.layers["count"]):
            msg += f"{ii + 1})\n"
            msg += "Веса\n"
            msg += str(self.model.layers[ii].params[0]) + "\n\n"
            msg += "Смещения\n"
            msg += str(self.model.layers[ii].params[1]) + "\n\n"
        msg += delim
        return msg
