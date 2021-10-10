from typing import List

from numpy import ndarray

from layers import Layer
from losses import Loss, MeanSquaredError


class NeuralNetwork(object):
    """
    Нейронная сеть
    """
    def __init__(self,
                 layers: List[Layer],
                 loss: Loss = MeanSquaredError,
                 seed: int = None):
        """
        Конструктор нейронной сети

        Parameters
        ----------
        layers: List[Layer]
            Набор слоёв
        loss: Loss
            Функция потерь
        seed: int
            Сид для инициализации рандомайзера
        """
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self,
                x_batch: ndarray,
                inference: bool = False) -> ndarray:
        """
        Передача входа через набор слоёв

        Parameters
        ----------
        inference: bool
            Обратный проход?
        x_batch: ndarray
            Пакет входных данных

        Returns
        -------
        ndarray
            Выход нейросети
        """
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out, inference)

        return x_out

    def backward(self, loss_grad: ndarray) -> ndarray:
        """
        Передача ошибки назад через набор слоёв

        Parameters
        ----------
        loss_grad: ndarray
            Градиент функции потерь

        Returns
        -------
        None
        """
        grad = loss_grad

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def forward_loss(self,
                     x_batch: ndarray,
                     y_batch: ndarray,
                     inference: bool = False) -> float:
        """
        Вычисление потерь

        Parameters
        ----------
        x_batch: ndarray
            Массив входных значений
        y_batch: ndarray
            Массив выходных значений
        inference: bool
            Обратный проход?

        Returns
        -------
        float
            Потеря
        """
        prediction = self.forward(x_batch, inference)
        return self.loss.forward(prediction, y_batch)

    def train_batch(self,
                    x_batch: ndarray,
                    y_batch: ndarray,
                    inference: bool = False) -> float:
        """
        Обработка пакета входных данных (вычисление выхода,
        вычисление потерь, вычисление градиента)

        Parameters
        ----------
        inference: bool
            Обратный проход?
        x_batch: ndarray
            Пакет входных данных из обучающей выборки
        y_batch: ndarray
            Пакет выходных данных из обучающей выборки

        Returns
        -------
        float
            Потеря
        """
        predictions = self.forward(x_batch, inference)

        batch_loss = self.loss.forward(predictions, y_batch)
        loss_grad = self.loss.backward()

        self.backward(loss_grad)

        return batch_loss

    def params(self):
        """
        Получение параметров нейросети
        """
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        """
        Получение градиента потерь по отношению к параметрам нейросети
        """
        for layer in self.layers:
            yield from layer.param_grads
