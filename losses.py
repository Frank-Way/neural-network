import numpy as np
from numpy import ndarray

from utils import softmax, assert_same_shape, normalize, unnormalize


class Loss(object):
    """
    Потери при обучении
    """
    prediction: ndarray
    target: ndarray
    output: float
    input_grad: ndarray

    def __init__(self):
        pass

    def forward(self,
                prediction: ndarray,
                target: ndarray) -> float:
        """
        Вычисление потерь

        Parameters
        ----------
        prediction: ndarray
            Предсказанные значения
        target: ndarray
            Целевые значения

        Returns
        -------
        float
            Потери
        """
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        self.output = self._output()

        return self.output

    def backward(self) -> ndarray:
        """
        Вычисление градиента потерь по входам функции потерь

        Returns
        -------
        ndarray
            Градиент потерь
        """
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        """
        Вычисление потерь реализуется в наследниках
        """
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        """
        Вычисление градиента потерь реализуется в наследниках
        """
        raise NotImplementedError()


class MeanSquaredError(Loss):
    """
    Среднеквадратическая ошибка
    """
    def __init__(self,
                 needs_normalization: bool = False):
        super().__init__()
        self.normalize = needs_normalization

    def _output(self) -> float:
        """
        Вычисление среднего квадрата ошибки

        Returns
        -------
        float
            Ошибка
        """
        if self.normalize:
            self.prediction = self.prediction / self.prediction.sum(axis=1, keepdims=True)
        loss = np.sum(np.power(self.prediction - self.target, 2))
        loss /= self.prediction.shape[0]

        return loss

    def _input_grad(self) -> ndarray:
        """
        Вычисление ошибка градиента функции потерь по входу

        Returns
        -------
        ndarray
            Градиент потерь
        """
        input_grad = 2.0 * (self.prediction - self.target)
        input_grad /= self.prediction.shape[0]

        return input_grad


class SoftmaxCrossEntropy(Loss):
    """
    Функция потерь softmax + cross entropy.
    Softmax позволяет привести выходы сети к диапозону [0; 1]
    """
    def __init__(self, eps: float = 1e-9):
        """
        Конструктор функции потерь

        Parameters
        ----------
        eps: float
            Погрешность
        """
        super().__init__()
        self.eps = eps
        self.single_class = False

    def _output(self) -> float:
        """
        Вычисление потерь

        Returns
        -------
        float
            Потеря
        """
        # если сеть выдаёт вероятность принадлежности к одному классу
        if self.target.shape[1] == 0:
            self.single_class = True

        # если single_class, применяется нормализация
        if self.single_class:
            self.prediction, self.target = \
                normalize(self.prediction), normalize(self.target)

        # применение softmax
        softmax_preds = softmax(self.prediction, axis=1)

        # ограничение softmax до интервала [eps; 1 - eps]
        self.softmax_preds = np.clip(softmax_preds,
                                     self.eps,
                                     1 - self.eps)

        # вычисление потерь
        softmax_cross_entropy_loss = -(
                self.target * np.log(self.softmax_preds) +
                (1.0 - self.target) * np.log(1 - self.softmax_preds)
        )
        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:
        """
        Вычисление градиента на входе

        Returns
        -------
        ndarray
            Градиент на входе
        """
        if self.single_class:
            return unnormalize(self.softmax_preds - self.target)
        else:
            return (self.softmax_preds - self.target) / self.prediction.shape[0]
