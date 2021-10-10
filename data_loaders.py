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


class MNISTDataLoader(object):
    """
    Класс для загрузки набора рукописных цифр MNIST
    """
    filename = [
        ["training_images", "train-images-idx3-ubyte.gz"],
        ["test_images", "t10k-images-idx3-ubyte.gz"],
        ["training_labels", "train-labels-idx1-ubyte.gz"],
        ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]

    def __init__(self, path, scale_inputs=None):
        self.path = path

    def download(self):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in self.filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], join(self.path, name[1]))
            # request.urlretrieve(base_url + name[1], name[1])
        print("Download complete.")

    def save(self):
        mnist = {}
        for name in self.filename[:2]:
            with gzip.open(join(self.path, name[1]), 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        for name in self.filename[-2:]:
            with gzip.open(join(self.path, name[1]), 'rb') as f:
                mnist[name[0]] = mnist_labels_to_y(np.frombuffer(f.read(), np.uint8, offset=8))
        with open(join(self.path, "mnist.pkl"), 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    def load(self):
        if not exists(join(self.path, "mnist.pkl")):
            self.download()
            self.save()
        with open(join(self.path, "mnist.pkl"), 'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["test_images"], mnist["training_labels"], mnist["test_labels"]


class ApproximationDataLoader(object):
    """
    Класс для загрузки данных для обучения для задачи аппроксимации
    функции
    """
    p_to_test = 0.30
    p_to_extend = 0.15
    sep = "/"

    def __init__(self, path: str,
                 limits: List[List[float]],
                 scale_inputs: bool = False):
        """
        Конструктор

        Parameters
        ----------
        limits : List[List[float, float]]
            Список, каждое значение которого представляет собой список
            минимального и максимального значения для переменной
        path: str
            Путь к файлу с выборкой
        scale_inputs: bool
            Скалировать ли входные данные?
        """
        self.path = path
        self.limits = limits
        self.scale_inputs = scale_inputs

        splitted_by_sep_path = self.path.split(self.sep)

        self.folder = join(*(splitted_by_sep_path[:-1]))
        self.filename = splitted_by_sep_path[-1:][0]
        self.extension = self.filename.split(".")[-1]

        self.inputs = int(self.folder.split(self.sep)[-1])
        self.input_symbols = [sympy.Symbol(f"x{ii + 1}") for ii in range(self.inputs)]

        self.expression = self.filename.split('_')[0]
        self.simplified_expression = sympy.simplify(self.expression)
        self.str_expression = replace_chars(str(self.simplified_expression))

        self.function = sympy.lambdify(self.input_symbols, self.simplified_expression)

        self.train_size = int(self.filename.split("_")[-1].split(".")[0])
        # self.train_size = np.power(self.size, self.inputs)
        self.test_size = int(self.train_size * self.p_to_test)

        self.modified_path = join(self.folder, f"{self.str_expression}_{self.train_size}.{self.extension}")

    def load(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Загружает (или генерирует) обучающую выборку

        Returns
        -------
        Tuple
            0: ndarray
                Массив входных значений для обучения
            1: ndarray
                Массив входных значений для тестов
            2: ndarray
                Массив выходных значений для обучения
            3: ndarray
                Массив выходных значений для тестов
        """

        # if exists(self.path):
        #     with open(self.path, 'rb') as f:
        #         data = pickle.load(f)
        # elif exists(self.modified_path):
        #     with open(self.modified_path, 'rb') as f:
        #         data = pickle.load(f)
        # else:
        #     data = self.generate()

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
            "x_train": ndarray
                Массив входных значений для обучения
            "x_test": ndarray
                Массив входных значений для тестов
            "y_train": ndarray
                Массив выходных значений для обучения
            "y_test": ndarray
                Массив выходных значений для тестов
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
            x_train_extended_list.append(np.linspace(lleft, rright, num=self.train_size))

        x_train = cartesian(tuple(x_train_list))
        x_train_extended = cartesian(tuple(x_train_extended_list))
        x_test = cartesian(tuple(x_test_list))
        y_train = self.function(*[x_train[:, ii] for ii in range(self.inputs)])
        y_test = self.function(*[x_test[:, ii] for ii in range(self.inputs)])

        extended_valid = True
        try:
            y_train_extended = self.function(*[x_train_extended[:, ii] for ii in range(self.inputs)])
        except ValueError:
            extended_valid = False

        if extended_valid:
            x_train = x_train_extended
            y_train = y_train_extended

        data = {"x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test}

        with open(self.modified_path, 'wb') as f:
            pickle.dump(data, f)

        return data
