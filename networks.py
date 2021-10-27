"""
Модуль с описанием класса-нейросети
"""
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
        layers: Набор слоёв
        loss: Функция потерь
        seed: Сид для инициализации рандомайзера
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
        inference: Обратный проход?
        x_batch: Пакет входных данных
        Returns
        -------
        ndarray: Выход нейросети
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
        loss_grad: Градиент функции потерь
        Returns
        -------
        ndarray: Градиент на входе нейросети
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
        x_batch: Массив входных значений
        y_batch: Массив выходных значений
        inference: Обратный проход?
        Returns
        -------
        float: Потеря
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
        inference: Обратный проход?
        x_batch: Пакет входных данных из обучающей выборки
        y_batch: Пакет выходных данных из обучающей выборки
        Returns
        -------
        float: Потеря
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
        Получение градиента по параметрам нейросети
        """
        for layer in self.layers:
            yield from layer.param_grads
