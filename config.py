"""
Модуль с описанием класса, хранящего настройки для обучения
"""
import json
import os
from os.path import exists
from typing import Tuple, Any, List, Callable

import numpy as np
import sympy

from data_loaders import MNISTDataLoader, ApproximationDataLoader, DataLoader
from exceptions import OptimizerNotFoundException, \
    DataLoaderNotFoundException, NotImplementedDataLoaderGraphException, NoPathToConfigSpecifiedException
from layers import Dense
from losses import MeanSquaredError, SoftmaxCrossEntropy
from networks import NeuralNetwork
from operations import Sigmoid, Linear, LeakyReLU, ReLU, Tanh
from optimizers import SGD, SGDMomentum, Optimizer
from trainers import Trainer
from utils import calc_accuracy_model, show_results, show_results3d, show_results_losses


class InputsConfig:
    count: int
    path: str
    data_loader: str
    function: str
    limits: List[List[float]]
    size: int
    scale_inputs: bool
    p_to_test: float
    p_to_extend: float

    def __init__(self, count: int, path: str, data_loader: str, function: str,
                 limits: List[List[float]], size: int, scale_inputs: bool,
                 p_to_test: float, p_to_extend: float) -> None:
        self.count = count
        self.path = path
        self.data_loader = data_loader
        self.function = function
        self.limits = limits
        self.size = size
        self.scale_inputs = scale_inputs
        self.p_to_test = p_to_test
        self.p_to_extend = p_to_extend

    @staticmethod
    def from_json(json_dct: dict):
        return InputsConfig(**json_dct)

    def to_json(self) -> dict:
        return self.__dict__


class LayerConfig:
    neurons: int
    layer_class: str
    activation: str
    dropout: float
    weight_init: str

    def __init__(self, neurons: int, layer_class: str, activation: str,
                 dropout: float, weight_init: str) -> None:
        self.neurons = neurons
        self.layer_class = layer_class
        self.activation = activation
        self.dropout = dropout
        self.weight_init = weight_init

    @staticmethod
    def from_json(json_dct: dict):
        return LayerConfig(**json_dct)

    def to_json(self) -> dict:
        return self.__dict__


class LayersConfig:
    count: int
    layers_config: List[LayerConfig]

    def __init__(self, count: int, layers_config: List[LayerConfig]) -> None:
        self.count = count
        self.layers_config = layers_config

    @staticmethod
    def from_json(json_dct: dict):
        return LayersConfig(
            json_dct["count"], [LayerConfig.from_json(inner_json_dct)
                                for inner_json_dct in json_dct["layers_config"]])

    def to_json(self) -> dict:
        return {"count": self.count,
                "layers_config": [layer_config.to_json()
                                  for layer_config in self.layers_config]}




class TrainConfig:
    seed: int
    use_seed: bool
    lr: float
    final_lr: float
    decay_type: str
    epochs: int
    query_times: int
    batch_size: int
    early_stopping: bool
    print_results: bool
    show_plots: bool
    momentum: float
    loss: str
    optimizer: str
    trainer: str
    restarts: int

    def __init__(self, seed: int, use_seed: bool, lr: float, final_lr: float,
                 decay_type: str, epochs: int, query_times: int, batch_size: int,
                 early_stopping: bool, print_results: bool, show_plots: bool,
                 momentum: float, loss: str, optimizer: str, trainer: str,
                 restarts: int) -> None:
        self.seed = seed
        self.use_seed = use_seed
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.epochs = epochs
        self.query_times = query_times
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.print_results = print_results
        self.show_plots = show_plots
        self.momentum = momentum
        self.loss = loss
        self.optimizer = optimizer
        self.trainer = trainer
        self.restarts = restarts

    @staticmethod
    def from_json(json_dct: dict):
        return TrainConfig(**json_dct)

    def to_json(self) -> dict:
        return self.__dict__

class Config:
    inputs: InputsConfig
    train: TrainConfig
    layers: LayersConfig
    default_path: str = "settings/config.json"

    def __init__(self, inputs: InputsConfig, train: TrainConfig,
                 layers: LayersConfig):
        self.inputs = inputs
        self.train = train
        self.layers = layers

    @staticmethod
    def from_json(json_dct: dict):
        return Config(InputsConfig.from_json(json_dct["inputs"]),
                      TrainConfig.from_json(json_dct["train"]),
                      LayersConfig.from_json(json_dct["layers"]))

    def to_json(self) -> dict:
        return {"inputs": self.inputs.to_json(),
                "train": self.train.to_json(),
                "layers": self.layers.to_json()}

    @staticmethod
    def load(path: str = None):
        config: Config = None
        if path is not None:
            if exists(path):
                with open(path, "r") as file:
                    config = Config.from_json(json.load(file))
        if config is None:
            config = Config.generate_default()
        return config

    def save(self, path: str) -> None:
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as file:
                json.dump(self.to_json(), file, indent=4)
        else:
            raise NoPathToConfigSpecifiedException()

    @staticmethod
    def generate_default():
        config = Config(
            inputs=InputsConfig(1, "", "ap", "sin(pi*x1)", [[0.0, 1.0]], 256,
                                False, 0.3, 0.1),
            train=TrainConfig(0, False, 0.1, 0.001, "lin", 5000, 10, 64, True,
                              True, False, 0.8, "mse", "sgd", "t", 1),
            layers=LayersConfig(2, [
                LayerConfig(8, "dense", "tanh", 1.0, "glorot"),
                LayerConfig(1, "dense", "linear", 1.0, "glorot")
            ]))
        return config





class ConfigHandler(object):
    config: Config  # объединение конфигураций
    model: NeuralNetwork  # нейросеть
    trainer: Trainer  # тренер
    optimizer: Optimizer  # оптимизатор
    dl: DataLoader  # загрузчик данных
    deltas: Tuple[float, float, float]  # результаты обучения
    decoder: dict  # декодировщик значений из конфигурации в нужные классы
    default_path: str = "settings/config.json"

    def __init__(self, path: str, config: Config) -> None:
        self.config = config
        self.path = path
        self.decoder = dict(dense=Dense, sigmoid=Sigmoid, linear=Linear,
                            leakyrelu=LeakyReLU, relu=ReLU, tanh=Tanh,
                            mse=MeanSquaredError, sce=SoftmaxCrossEntropy,
                            sgd=SGD, sgdm=SGDMomentum, t=Trainer,
                            mnist=MNISTDataLoader, ap=ApproximationDataLoader)
        self.deltas = None

    def _get_net(self) -> NeuralNetwork:
        """
        Генерация модели по заданным настройкам
        Returns
        -------
        NeuralNetwork
            Модель
        """
        nn = NeuralNetwork(
            layers=[self.decoder[self.config.layers.layers_config[ll].layer_class](
                neurons=self.config.layers.layers_config[ll].neurons,
                activation=self.decoder[self.config.layers.layers_config[ll].activation](),
                dropout=self.config.layers.layers_config[ll].dropout,
                weight_init=self.config.layers.layers_config[ll].weight_init
            )
                for ll in range(self.config.layers.count)],
            loss=self.decoder[self.config.train.loss]()
        )
        return nn

    def _get_optimizer(self) -> Optimizer:
        """
        Генерация оптимизатора по заданным настройкам
        Returns
        -------
        Optimizer
            Оптимизатор
        """
        if self.config.train.optimizer == "sgd":
            optimizer = self.decoder["sgd"](
                lr=self.config.train.lr,
                final_lr=self.config.train.final_lr,
                decay_type=self.config.train.decay_type
            )
        elif self.config.train.optimizer == "sgdm":
            optimizer = self.decoder["sgdm"](
                lr=self.config.train.lr,
                final_lr=self.config.train.final_lr,
                decay_type=self.config.train.decay_type,
                momentum=self.config.train.momentum
            )
        else:
            raise OptimizerNotFoundException(self.config.train.optimizer)
        return optimizer

    def get_trainer(self) -> Trainer:
        """
        Генерация тренера по заданным настройкам
        Returns
        -------
        Trainer
            Тренер
        """
        self.model = self._get_net()
        self.optimizer = self._get_optimizer()
        self.trainer = self.decoder[self.config.train.trainer](self.model,
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
        if self.config.inputs.data_loader == "ap":
            self.dl = self.decoder["ap"](
                inputs=self.config.inputs.count,
                fun_name=self.config.inputs.function,
                size=self.config.inputs.size,
                scale_inputs=self.config.inputs.scale_inputs,
                limits=self.config.inputs.limits,
                p_to_test=self.config.inputs.p_to_test,
                p_to_extend=self.config.inputs.p_to_extend
            )
        elif self.config.inputs.data_loader == "mnist":
            self.dl = self.decoder["mnist"](self.config.inputs.path)
        else:
            raise DataLoaderNotFoundException(self.config.inputs.data_loader)
        x_train, x_test, y_train, y_test = self.dl.load()

        return {"x_train": x_train, "x_test": x_test,
                "y_train": y_train, "y_test": y_test}

    def get_fit_params(self, data: dict) -> dict:
        """
        Генерация набора параметров для вызова функции обучения нейросети
        Parameters
        ----------
        data: dict
            Данные для обучения и тестов
        Returns
        -------
        dict
            Параметры
        """
        fit_params = {
            "x_train": data["x_train"], "x_test": data["x_test"],
            "y_train": data["y_train"], "y_test": data["y_test"],
            "epochs": self.config.train.epochs,
            "query_every": self.config.train.epochs // self.config.train.query_times,
            "batch_size": self.config.train.batch_size,
            "early_stopping": self.config.train.early_stopping,
            "seed": self.config.train.seed if self.config.train.use_seed else None,
            "print_results": self.config.train.print_results
        }
        return fit_params

    def get_str_results(self, results: Tuple) -> str:
        """
        Формирование строки с результатами обучения
        Parameters
        ----------
        results: Tuple
            Потеря, макс. ошибка, потери, тестовые входы, выходы и результаты
        Returns
        -------
        str
            Результат обучения
        """
        loss, delta, losses, x_test, y_test, test_preds = results
        layers = [self.config.layers.layers_config[ll].neurons
                  for ll in range(self.config.layers.count)]
        if self.config.inputs.data_loader == "ap":
            function = self.config.inputs.function
            size = self.config.inputs.size
            rel_delta = delta / np.abs(np.max(y_test) - np.min(y_test)) * 100
            avg_delta = np.average(np.abs(y_test - test_preds))
            result = f"слои: {layers[:-1]};\tфункция: {function};" \
                     f"\tразмер выборки: {size}"
            result += f"\nмакс. абс. ошибка: {delta:e};" \
                      f"\tмакс. относ. ошибка: {rel_delta:e} %;" \
                      f"\tсред. абс. ошибка: {avg_delta:e}"
            self.deltas = (delta, rel_delta, avg_delta)
        elif self.config.inputs.data_loader == "mnist":
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
        results: Tuple
            Потеря, макс. ошибка, потери, тестовые входы, выходы и результаты
        Returns
        -------
        Tuple
            Функция и параметры для её вызова
        """
        if self.config.inputs.data_loader == "ap":
            if self.config.train.show_plots:
                loss, delta, losses, x_test, y_test, test_preds = results
                layers = [self.config.layers.layers_config[ll].neurons
                          for ll in range(self.config.layers.count)]
                params = {"losses": losses, "x_test": x_test,
                          "pred_test": test_preds, "y_test": y_test,
                          "function_name": self.config.inputs.function,
                          "neurons": layers}
                if self.config.inputs.count == 1:
                    function = show_results
                elif self.config.inputs.count == 2:
                    function = show_results3d
                else:
                    function = show_results_losses
                return function, params
            else:
                return
        else:
            raise NotImplementedDataLoaderGraphException(
                self.config.inputs.data_loader)

    def export_model(self, result_type: str = "text") -> Any:
        """
        Экспорт модели в заданный тип
        Parameters
        ----------
        result_type: str
            Тип результата экспорта (text - строка)
        Returns
        -------
        Any
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
        if self.config.inputs.count == 1:
            msg += "    F(x1)"
        elif self.config.inputs.count == 2:
            msg += "    F(x1, x2)"
        else:
            msg += f"    F(x1, ..., x{self.config.inputs.count})"
        msg += f" = {self.dl.str_expression}\n"

        msg += "в пределах изменения входных переменных:\n"
        for ii in range(self.config.inputs.count):
            msg += f"    x{ii + 1}: [{self.config.inputs.limits[ii][0]}; " \
                   f"{self.config.inputs.limits[ii][1]}]\n"

        if self.deltas is not None:
            msg += "\n".join(("с точностью:",
                              f"    макс. абс. ошибка = {self.deltas[0]:e}",
                              f"    макс. отн. ошибка = {self.deltas[1]:e} %",
                              f"    сред. абс. ошибка = {self.deltas[2]:e}"))
        msg += "\n" + delim + "\n"

        msg += "Конфигурация слоёв\n"
        for ii in range(self.config.layers.count):
            neurons = self.config.layers.layers_config[ii].neurons
            act = activations_decoder[self.config.layers.layers_config[ii].activation]
            dropout = self.config.layers.layers_config[ii].dropout
            layer_cfg = "\n".join((f"{ii + 1})",
                                   f"    нейронов - {neurons}",
                                   f"    активация - {act}",
                                   f"    dropout - {dropout}"))
            msg += layer_cfg + "\n"
        msg += delim + "\n"

        msg += "Параметры модели\n"
        for ii in range(self.config.layers.count):
            msg += f"{ii + 1})\n"
            msg += "Веса\n"
            msg += str(self.model.layers[ii].params[0]) + "\n\n"
            msg += "Смещения\n"
            msg += str(self.model.layers[ii].params[1]) + "\n\n"
        msg += delim
        return msg


class ConfigUtils:
    @staticmethod
    def lambidify_function(fn: str, inputs_count: int,
                           base_symbol: str = "x") -> Callable:
        input_symbols = [sympy.Symbol(f"{base_symbol}{ii + 1}")
                         for ii in range(inputs_count)]
        return sympy.lambdify(input_symbols, fn)

    @staticmethod
    def simplify_function(fn: str) -> str:
        return str(sympy.simplify(fn.lower()))

    @staticmethod
    def is_function_valid(fn: Callable, inputs_count: int) -> bool:
        valid = True
        try:  # попытка вычислить значение функции от случайных аргументов
            res = fn(*(10 * np.random.random(inputs_count) - 5))
        # если функция задана не правильно, function генерирует это исключение
        except NameError:
            valid = False  # функция задана не правильно
        # проверка типа возвращаемого значения
        if valid and type(res) != np.float64:  # если было возвращено не число
            valid = False  # функция задана не правильно
        return valid

    @staticmethod
    def validate(config: Config) -> str:
        message = ""
        # проверка различных значений
        valid_inputs = config.inputs.count != 0
        message += "" if valid_inputs else "Число входов должно быть" \
                                           " отлично от нуля\n"

        valid_function = ConfigUtils.is_function_valid(
            ConfigUtils.lambidify_function(
                ConfigUtils.simplify_function(config.inputs.function),
                config.inputs.count),
            config.inputs.count)
        message += "" if valid_function else "Функция задана неверно\n"

        valid_limits = all(map(lambda limit: limit[0] < limit[1],
                               config.inputs.limits))
        message += "" if valid_limits else "Левая граница для входа должна " \
                                           "быть строго меньше правой\n"

        valid_size = config.inputs.size != 0
        message += "" if valid_size else "Размер выборки должен быть" \
                                         " отличен от нуля\n"

        valid_batch = config.train.batch_size != 0
        message += "" if valid_batch else "Размер пакета должен быть" \
                                          " отличен от нуля\n"

        valid_test = config.inputs.p_to_test != 0.0
        message += "" if valid_test else "Размер тестовой выборки должен " \
                                         "быть отличен от нуля\n"

        valid_lr = 0.0 < config.train.lr <= 2
        message += "" if valid_lr else "Начальная скорость обучения должна " \
                                       "быть в пределах (0; 2]\n"

        valid_final_lr = 0.0 < config.train.final_lr <= 2
        message += "" if valid_final_lr else "Конечная скорость обучения " \
                                             "должна быть в пределах (0; 2]\n"

        valid_decay = config.train.decay_type is not None
        message += "" if valid_decay else "Не выбран способ снижения " \
                                          "скорости обучения\n"

        valid_loss = config.train.loss is not None
        message += "" if valid_loss else "Не выбрана функция потерь\n"

        valid_optimizer = config.train.optimizer is not None
        message += "" if valid_optimizer else "Не выбран оптимизатор\n"

        valid_epochs = config.train.epochs > 1
        message += "" if valid_epochs else "Количество эпох обучения " \
                                           "должно быть больше 1\n"

        valid_queries = config.train.query_times > 0
        message += "" if valid_queries else "Недостаточное количество эпох " \
                                            "обучения для реализации " \
                                            "заданного числа опросов\n"

        valid_momentum = True
        if config.train.optimizer == "sgdm":
            valid_momentum = 0.0 < config.train.momentum <= 1.0
            message += "" if valid_momentum else "Инертность должна быть" \
                                                 " в пределах (0; 1]\n"

        valid_restarts = config.train.restarts > 0
        message += "" if valid_restarts else "Количество запусков должно " \
                                             "быть больше 0"

        valid_layers = config.layers.count != 0
        message += "" if valid_layers else "Число слоёв должно быть" \
                                           " отлично от нуля\n"

        valid_layers_config_neurons = all(
            map(lambda cfg: cfg.neurons > 0, config.layers.layers_config))
        if not valid_layers_config_neurons:
            message += "Число нейронов должно быть отлично от нуля\n"

        valid_layers_config_activations = all(
            map(lambda cfg: cfg.activation is not None,
                config.layers.layers_config))
        if not valid_layers_config_activations:
            message += "Не выбрана функция активации\n"

        valid_layers_config_dropout = all(
            map(lambda cfg: 0.0 < cfg.dropout <= 1.0,
                config.layers.layers_config))
        if not valid_layers_config_dropout:
            message += "Drop-out должен быть в пределах (0; 1]\n"

        # объединение всех проверок по И
        valid = all((valid_restarts, valid_queries, valid_epochs, valid_inputs,
                     valid_size, valid_lr,
                     valid_test, valid_layers, valid_final_lr, valid_function,
                     valid_optimizer, valid_momentum, valid_limits, valid_loss,
                     valid_layers_config_neurons, valid_layers_config_dropout,
                     valid_layers_config_activations, valid_batch, valid_decay))
        if not valid:  # при наличии ошибок
            return message  # возвращается сообщение об ошибках
