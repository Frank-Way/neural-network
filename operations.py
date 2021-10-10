import numpy as np
from numpy import ndarray

from utils import assert_same_shape


# # # # # # # # # # # # # # # # # # # # # # # # # #
# Базовые классы представления операций           #
# # # # # # # # # # # # # # # # # # # # # # # # # #


class Operation(object):
    """
    Базовый класс операции в нейронной сети
    """
    input_: ndarray
    output: ndarray
    input_grad: ndarray

    def __init__(self):
        pass

    def forward(self,
                input_: ndarray,
                inference: bool = False) -> ndarray:
        """
        Получение результата обработки входа
        Вход хранится в атрибуте экземпляра _input
        Выход получен вызовом соответствующего методоа _output

        Parameters
        ----------
        input_: ndarray
            Вход для операции
        inference: bool
            Обратный проход

        Returns
        -------
        ndarray
            Результат выполнения операции
        """
        self.input_ = input_

        self.output = self._output(inference)

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Получение градиента при обратном проходе

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе операции (исходный градиент при обратном
            проходе)

        Returns
        -------
        ndarray
            Градиент на входе операции (результирующий градиент при
            обратном проходе)
        """
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self, inference: bool = False) -> ndarray:
        """
        Метод вычисления выхода реализуется в наследниках

        Returns
        -------
        ndarray
            Результат выполнения операции
        """
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Метод вычисления градиента реализуется в наследниках

        Parameters
        ----------
        output_grad
            Градиент, полученный от следующей операции

        Returns
        -------
        ndarray
            Результат вычисления градиента при обратном проходе
        """
        raise NotImplementedError()


class ParamOperation(Operation):
    """
    Базовый класс для операций с параметрами
    """
    param_grad: ndarray

    def __init__(self, param: ndarray):
        """
        Конструктор операции с параметром

        Parameters
        ----------
        param: ndarray
            Параметр операции
        """
        super().__init__()

        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Получение градиента при обратрном проходе для
        операции с параметром

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

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Метод вычисления градиента для операции с параметром реализуется
        в наследниках

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе операции с параметром

        Returns
        -------
        ndarray
            Градиент на входе операции с параметром
        """
        raise NotImplementedError()

    def _output(self, inference: bool = False) -> ndarray:
        """
        Метод вычисления выхода реализуется в наследниках

        Returns
        -------
        ndarray
            Результат выполнения операции
        """
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Метод вычисления градиента реализуется в наследниках

        Parameters
        ----------
        output_grad
            Градиент, полученный от следующей операции

        Returns
        -------
        ndarray
            Результат вычисления градиента при обратном проходе
        """
        raise NotImplementedError()


# # # # # # # # # # # # # # # # # # # # # # # # # #
# Классы, реализующие выполнения операций         #
# # # # # # # # # # # # # # # # # # # # # # # # # #


class WeightMultiply(ParamOperation):
    """
    Операция умножения на веса
    """
    def __init__(self, W: ndarray):
        """
        Конструктор операции с параметром, где self.param = W

        Parameters
        ----------
        W: ndarray
            Массив весов
        """
        super().__init__(W)

    def _output(self, inference: bool = False) -> ndarray:
        """
        Умножение (вычисление выхода)

        Returns
        -------
        ndarray
            Результат умножения (выход)
        """
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиента на входе

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент на входе
        """
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиента по параметру

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент на входе
        """
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    """
    Операция добавления смещений
    """
    def __init__(self, B: ndarray):
        """
        Конструктор операции с параметром, где self.param = B

        Parameters
        ----------
        B: ndarray
            Массив смещений
        """
        assert B.shape[0] == 1

        super().__init__(B)

    def _output(self, inference: bool = False) -> ndarray:
        """
        Добавление смещений (получение выхода)

        Returns
        -------
        ndarray
            Результат добавления
        """
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиента на входе

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент на выходе
        """
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиента по параметру на входе

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент по параметру на входе
        """
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Dropout(Operation):
    """
    Операция прореживания
    """
    mask: ndarray

    def __init__(self,
                 keep_prob: float = 0.8):
        """
        Конструктор

        Parameters
        ----------
        keep_prob: float
            Вероятность того, что нейрон не будет выключен
        """
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self, inference: bool = False) -> ndarray:
        """
        Прореживание

        Parameters
        ----------
        inference: bool
            Обратный проход?

        Returns
        -------
        ndarray
            Прореженный массив
        """
        if inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob,
                                           size=self.input_.shape)
            return self.input_ * self.mask

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиента на входе

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент на входе
        """
        return output_grad * self.mask


# # # # # # # # # # # # # # # # # # # # # # # # # #
# Классы, реализующие функции активации           #
# # # # # # # # # # # # # # # # # # # # # # # # # #


class Sigmoid(Operation):
    """
    Сигмоидная функция активации
    """
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = None) -> ndarray:
        """
        Вычисление сигмоиды (выход)

        Parameters
        ----------
        inference: bool
            Обратный проход?

        Returns
        -------
        ndarray
            Массив активаций
        """
        return 1.0 / (1.0 + np.exp(-self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиента на входе

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент на входе
        """
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Linear(Operation):
    """
    Линейная функция активации
    """
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = None) -> ndarray:
        """
        Вход передаётся на выход без изменений

        Parameters
        ----------
        inference: bool
            Обратный проход

        Returns
        -------
        ndarray
            Массив активаций
        """
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Градиент передаётся на вход без изменений

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент на входе
        """
        return output_grad


class Tanh(Operation):
    """
    Функция активации - гиперболический тангенс
    """
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = None) -> ndarray:
        """
        Вычисление гиперболического тангенса

        Parameters
        ----------
        inference: bool
            Обратный проход

        Returns
        -------
        ndarray
            Массив активаций
        """
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиента на входе

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент на выходе
        """
        return output_grad * (1 - self.output * self.output)


class LeakyReLU(Operation):
    """
    Функция активации -  выпрямитель: max(0.2x, x)
    """
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = None) -> ndarray:
        """
        Вычисление LeakyReLU

        Parameters
        ----------
        inference: bool
            Обратный проход

        Returns
        -------
        ndarray
            Массив активаций
        """
        return np.where(self.input_ > 0.0, self.input_, 0.2 * self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиента на входе

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент на входе
        """
        leakyrelu_backward = np.where(self.output > 0.0, 1.0, 0.2)
        input_grad = leakyrelu_backward * output_grad
        return input_grad


class ReLU(Operation):
    """
    Функция активации -  выпрямитель: max(0, x)
    """
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = None) -> ndarray:
        """
        Вычисление LeakyReLU

        Parameters
        ----------
        inference: bool
            Обратный проход

        Returns
        -------
        ndarray
            Массив активаций
        """
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Вычисление градиента на входе

        Parameters
        ----------
        output_grad: ndarray
            Градиент на выходе

        Returns
        -------
        ndarray
            Градиент на входе
        """
        mask = self.output >= 0
        return output_grad * mask
