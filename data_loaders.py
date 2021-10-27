"""
Модуль с описанием классов, загружающих данные для обучения и тестов
"""
from os.path import join, exists
from typing import Tuple, List

from urllib import request
import gzip
import pickle
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler

import sympy

from utils import to_2d, mnist_labels_to_y, replace_chars, cartesian


class DataLoader(object):
    """
    Базовый класс
    """
    def __init__(self):
        pass

    def load(self):
        """
        Загрузка реализуется в наследниках
        """
        raise NotImplementedError


class MNISTDataLoader(DataLoader):
    """
    Класс для загрузки набора рукописных цифр MNIST
    """
    filename = [
        ["training_images", "train-images-idx3-ubyte.gz"],
        ["test_images", "t10k-images-idx3-ubyte.gz"],
        ["training_labels", "train-labels-idx1-ubyte.gz"],
        ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]

    def __init__(self, path: str):
        """
        Конструктор
        Parameters
        ----------
        path: Путь к выборке
        """
        self.path = path

    def download(self) -> None:
        """
        Загрузка выборки в виде сжатого архива
        """
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in self.filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], join(self.path, name[1]))
        print("Download complete.")

    def save(self) -> None:
        """
        Распаковка выборки и её сохранение в формате, удобном для дальнейшей
        работы
        """
        mnist = {}
        for name in self.filename[:2]:
            with gzip.open(join(self.path, name[1]), 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8,
                                               offset=16).reshape(-1, 28 * 28)
        for name in self.filename[-2:]:
            with gzip.open(join(self.path, name[1]), 'rb') as f:
                mnist[name[0]] = mnist_labels_to_y(np.frombuffer(f.read(),
                                                                 np.uint8,
                                                                 offset=8))
        with open(join(self.path, "mnist.pkl"), 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    def load(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Чтение выборки
        Returns
        -------
        Tuple[ndarray, ndarray, ndarray, ndarray]:
            Входные значения для обучения, входные значения для тестов,
            выходные значения для обучения, выходные значения для тестов
        """
        if not exists(join(self.path, "mnist.pkl")):
            self.download()
            self.save()
        with open(join(self.path, "mnist.pkl"), 'rb') as f:
            mnist = pickle.load(f)
        return (mnist["training_images"], mnist["test_images"],
                mnist["training_labels"], mnist["test_labels"])


class ApproximationDataLoader(DataLoader):
    """
    Класс для загрузки данных для обучения для задачи аппроксимации
    функции
    """
    sep = "/"

    def __init__(self, inputs: int,
                 fun_name: str,
                 size: int,
                 limits: List[List[float]],
                 scale_inputs: bool = False,
                 p_to_test: float = 0.3,
                 p_to_extend: float = 0.0) -> None:
        """
        Конструктор
        Parameters
        ----------
        inputs: Количество входов
        fun_name: Символическая запись функции
        size: Размер выборки
        limits: Границы для входных переменных
        scale_inputs: Скалировать входы к [0;1]?
        p_to_test: Сколько процентов составляет тестовая выборка
        p_to_extend: На сколько процентов расширять входные значения в выборке
        """
        self.limits = limits
        self.scale_inputs = scale_inputs
        self.expression = fun_name

        self.p_to_test = p_to_test
        self.p_to_extend = p_to_extend

        self.inputs = inputs
        self.input_symbols = [sympy.Symbol(f"x{ii + 1}")
                              for ii in range(self.inputs)]

        self.simplified_expression = sympy.simplify(self.expression)
        self.str_expression = replace_chars(str(self.simplified_expression))

        self.function = sympy.lambdify(self.input_symbols,
                                       self.simplified_expression)

        self.train_size = size
        self.test_size = int(self.train_size * self.p_to_test)

    def load(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Загружает (или генерирует) обучающую выборку
        Returns
        -------
        Tuple[ndarray, ndarray, ndarray, ndarray]:
            Входные значения для обучения, входные значения для тестов,
            выходные значения для обучения, выходные значения для тестов
        """
        data = self.generate()

        if self.scale_inputs:
            scaler = MinMaxScaler()
            merged = to_2d(np.vstack((data["x_train"], data["x_test"])))
            scaler.fit(merged)
            data["x_train"] = scaler.transform(to_2d(data["x_train"]))
            data["x_test"] = scaler.transform(to_2d(data["x_test"]))

        return (to_2d(data["x_train"]),
                to_2d(data["x_test"]),
                to_2d(data["y_train"]),
                to_2d(data["y_test"]))

    def generate(self) -> dict:
        """
        Функция генерации выборки
        Returns
        -------
        dict
            "x_train": Массив входных значений для обучения
            "x_test":  Массив входных значений для тестов
            "y_train": Массив выходных значений для обучения
            "y_test":  Массив выходных значений для тестов
        """
        x_train_list = []
        x_train_extended_list = []
        x_test_list = []

        for ii in range(self.inputs):
            left = self.limits[ii][0]
            right = self.limits[ii][1]
            x_train_list.append(np.linspace(left, right, num=self.train_size))
            x_test_list.append(np.linspace(left, right, num=self.test_size))

            delta = np.abs(right - left)
            lleft = left - self.p_to_extend * delta
            rright = right + self.p_to_extend * delta
            x_train_extended_list.append(np.linspace(lleft, rright,
                                                     num=self.train_size))

        x_train = cartesian(tuple(x_train_list))
        x_train_extended = cartesian(tuple(x_train_extended_list))
        x_test = cartesian(tuple(x_test_list))
        y_train = self.function(*[x_train[:, ii] for ii in range(self.inputs)])
        y_test = self.function(*[x_test[:, ii] for ii in range(self.inputs)])

        extended_valid = True
        try:
            y_train_extended = self.function(*[x_train_extended[:, ii]
                                               for ii in range(self.inputs)])
        except ValueError:
            extended_valid = False

        if extended_valid:
            x_train = x_train_extended
            y_train = y_train_extended

        data = {"x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test}

        return data
