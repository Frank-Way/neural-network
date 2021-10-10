from copy import deepcopy
from typing import Tuple, List

import numpy as np
from numpy import ndarray

from networks import NeuralNetwork
from optimizers import Optimizer
from utils import permute_data, batches_generator


class Trainer(object):
    """
    Класс, обучающий нейросеть
    """

    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer):
        """
        Конструктор тренера

        Parameters
        ----------
        net: NeuralNetwork
            Нейронная сеть
        optim: Optimizer
            Оптимизатор
        """
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def fit(self,
            x_train: ndarray, y_train: ndarray,
            x_test: ndarray, y_test: ndarray,
            epochs: int = 100,
            query_every: int = 10,
            batch_size: int = 32,
            seed: int = None,
            restart: bool = True,
            # single_output: bool = False,
            early_stopping: bool = True,
            print_results: bool = False) -> Tuple[float, float, List[float], ndarray, ndarray, ndarray]:
        """
        Коррекция параметров нейросети

        Parameters
        ----------
        print_results
        x_train: ndarray
            Входные значения
        y_train: ndarray
            Требуемые выходные значения
        x_test: ndarray
            Тестовые входные значения (они не участвуют при обучении)
        y_test: ndarray
            Тестовые требуемые выходные значения (аналогично)
        epochs: int
            Количество эпох обучения
        query_every: int
            Частота опроса/оценки
        batch_size: int
            Размер пакета для обучения
        seed: int
            Сид для инициализации рандомайзера
        restart: int
            Если True, то повторный вызов метода заново инициализирует
            параметры нейросети
        # single_output: bool
            Один выход?
        early_stopping: bool
            Останавливать при ухудшении результатов?

        Returns
        -------
        Tuple[float, float, List[float], ndarray, ndarray, ndarray]
        """
        setattr(self.optim, 'max_epochs', epochs)

        self.optim._setup_decay()

        if seed:
            np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        losses = []
        last_model = None
        loss = None
        test_preds = None

        for e in range(epochs):
            if (e + 1) % query_every == 0:
                last_model = deepcopy(self.net)
            x_train, y_train = permute_data(x_train, y_train)

            batch_generator = batches_generator(x_train, y_train, batch_size)

            for ii, (x_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(x_batch, y_batch)
                self.optim.step()
            if (e + 1) % query_every == 0:
                test_preds = self.net.forward(x_test, inference=True)
                loss = self.net.loss.forward(test_preds, y_test)
                losses.append(loss)
                if early_stopping:
                    if loss < self.best_loss:
                        if print_results:
                            print(f"Validation loss after {e + 1} epochs is "
                                  f"{loss:e} and max delta is "
                                  f"{np.max(np.abs(test_preds - y_test)):e}")
                        self.best_loss = loss
                    else:
                        if print_results:
                            print(
                                f"Loss increased after epoch {e + 1}, "
                                f"final loss was {self.best_loss:e}, using the model "
                                f"from epoch {e + 1 - query_every}")
                        self.net = last_model
                        setattr(self.optim, 'net', self.net)
                        break
                else:
                    if print_results:
                        print(
                            f"Validation loss after {e + 1} epochs is "
                            f"{loss:e} and max delta is "
                            f"{np.max(np.abs(test_preds - y_test)):e}")
            if self.optim.final_lr:
                self.optim._decay_lr()

        return loss, np.max(np.abs(test_preds - y_test)), losses, x_test, y_test, test_preds
