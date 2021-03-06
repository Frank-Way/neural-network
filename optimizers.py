"""
Модуль с описанием классов-оптимизаторов нейросети
"""
from typing import List

import numpy as np

from networks import NeuralNetwork


class Optimizer(object):
    """
    Базовый класс оптимизатора нейронной сети
    """
    max_epochs: int  # максимальное количество эпох обучения
    net: NeuralNetwork  # модель

    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0.0,
                 decay_type: str = "lin"):
        """
        Конструктор оптимизатора
        Parameters
        ----------
        decay_type: str
            Способ уменьшения скорости обучения
        final_lr: float
            Финальная скорость обучения
        lr: float
            Скорость обучения
        """
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True

    def _setup_decay(self) -> None:
        """
        Установка правила уменьшения скорости обучения
        """
        if not self.decay_type:
            return
        elif self.decay_type == "exp":
            self.decay_per_epoch = np.power(self.final_lr / self.lr,
                                            1.0 / (self.max_epochs - 1))
        elif self.decay_type == "lin":
            self.decay_per_epoch = ((self.lr - self.final_lr) /
                                    (self.max_epochs - 1))

    def _decay_lr(self) -> None:
        """
        Уменьшение скорости обучения
        """
        if not self.decay_type:
            return

        if self.decay_type == 'exp':
            self.lr *= self.decay_per_epoch

        elif self.decay_type == 'lin':
            self.lr -= self.decay_per_epoch

    def step(self,
             epoch: int = 0) -> None:
        """
        Шаг подстройки параметров модели
        Parameters
        ----------
        epoch: int
            Номер эпохи обучения
        """
        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            self._update_rule(param=param,
                              grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        """
        Правило подстройки параметров модели реализуется в наследниках
        """
        raise NotImplementedError()


class SGD(Optimizer):
    """
    Стохастический градиентный спуск
    """
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = "lin"):
        """
        Конструктор оптимизатора
        Parameters
        ----------
        decay_type: str
            Способ уменьшения скорости обучения
        final_lr: float
            Финальная скорость обучения
        lr: float
            Скорость обучения
        """
        super().__init__(lr, final_lr, decay_type)

    def _update_rule(self, **kwargs) -> None:
        """
        Каждый параметр корректируется в соответствии с вычисленными
        соответсвующими градиентами с учётом скорости обучения
        """
        update = self.lr * kwargs['grad']
        kwargs['param'] -= update


class SGDMomentum(Optimizer):
    """
    Интертный стохастический градиентный спуск
    """
    velocities: List[np.ndarray]

    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = "lin",
                 momentum: float = 0.9) -> None:
        """
        Конструктор оптимизатора
        Parameters
        ----------
        decay_type: str
            Способ уменьшения скорости обучения
        final_lr: float
            Финальная скорость обучения
        lr: float
            Скорость обучения
        momentum: float
            Величина инертности
        """
        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum

    def step(self, epoch: int = 0) -> None:
        """
        Реализация шага подстройки параметров модели с учётом
        инертности
        Parameters
        ----------
        epoch: int
            Эпоха обучения
        """
        if self.first:
            self.velocities = [np.zeros_like(param)
                               for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(self.net.params(),
                                                 self.net.param_grads(),
                                                 self.velocities):
            self._update_rule(param=param,
                              grad=param_grad,
                              velocity=velocity)

    def _update_rule(self, **kwargs: dict) -> None:
        """
        Правило подстройки параметров модели

        Parameters
        ----------
        kwargs: dict
            Параметры, градиенты, скорости
        Returns
        -------
        None
        """
        # обновление инертности
        kwargs['velocity'] *= self.momentum
        kwargs['velocity'] += self.lr * kwargs['grad']

        # приенение инертности для подстройки
        kwargs['param'] -= kwargs['velocity']
