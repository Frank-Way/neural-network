"""
Модуль с описанием классов слоёв нейросети
"""
from typing import List

import numpy as np
from numpy import ndarray

from operations import Operation, ParamOperation, Dropout, WeightMultiply, BiasAdd, Linear
from utils import assert_same_shape


class Layer(object):
    """
    Слой нейронов
    """
    input_: ndarray
    output: ndarray

    def __init__(self, neurons: int, seed: int = None):
        """
        Конструктор слоя
        Parameters
        ----------
        neurons: neurons
            Количество нейронов в слое
        seed: int
            Сид для инициализации рандомайзера
        """
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []
        self.seed = seed

    def _setup_layer(self, input_: ndarray) -> None:
        """
        Метод настройки слоя реализуется в наследниках
        """
        raise NotImplementedError()

    def forward(self,
                input_: ndarray,
                inference: bool) -> ndarray:
        """
        Обработка входа набором операций
        Parameters
        ----------
        inference: inference
            Обратный проход?
        input_: ndarray
            Массив входов
        Returns
        -------
        ndarray
            Выход
        """
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиенты в обратном направлении через набор операций
        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе
        Returns
        -------
        ndarray
            Градиент на входе
        """
        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()
        return input_grad

    def _param_grads(self) -> None:
        """
        Извлечение градиентов по параметрам из операций в слое
        """
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:
        """
        Извлечение параметров операций из слоя
        """
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    """
    Полносвязный слой
    """
    def __init__(self,
                 neurons: int,
                 activation: Operation = Linear(),
                 dropout: float = 1.0,
                 weight_init: str = "standard"):
        """
        Конструктор полносвязного слоя
        Parameters
        ----------
        weight_init: str
            Способ инициализации весов
        dropout: float
            Вероятность исключения нейронов из слоя
        neurons: int
            Количество нейронов в слое
        activation: Operation
            Функция активации
        """
        super().__init__(neurons)
        self.activation = activation
        self.dropout = dropout
        self.weight_init = weight_init

    def _setup_layer(self, input_: ndarray) -> None:
        """
        Определение операций и их параметров для полносвязного слоя
        Parameters
        ----------
        input_: ndarray
            Массив входных значений
        """
        if self.seed:
            np.random.seed(self.seed)

        num_in = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2 / (num_in + self.neurons)
        else:
            scale = 1.0

        self.params = []

        # веса
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(num_in, self.neurons)))

        # смещения
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(1, self.neurons)))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))
